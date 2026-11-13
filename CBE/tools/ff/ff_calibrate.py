"""ff_calibrate.py — fulcrum rig calibration + solve (consolidated).

Drives a live TouchDesigner session over the ws bridge (tools/td/td.mjs).
Self-contained: no imports from the legacy fulcrum_track/fulcrum_check
modules — the small shared helpers it needs are defined here.

Phases (run in order):
  0. Inputs          — fulcrum_spec, webcam K (cheb_view), frozen projector prior
  1. Webcam solve    — physical ff markers 10-14 -> webcam->ff pose
  2. Correspondences — project a 6x4 ArUco grid, detect it in the webcam,
                       back-project each corner onto ff faces / the table plane
  3. Projector solve — solvePnP seed -> least-squares (rvec,tvec,f,cy),
                       held-out selection, frozen prior as regularizer
  4. Face-dot sweep  — project individual dots at each face, diff-detect them
                       in the webcam, drift check (target must not move)
  5. Final fit       — weighted least-squares on all points + validation gates
  6. Kinect          — (--kinect) SDK factory mapper -> Kinect K, paired
                       webcam+kinect pose solves -> webcam_from_kinect
  7. Write + report  — smoke/ff_{ts}_solve.json archive + update
                       tools/pipe/ff/calibration_current.json +
                       tools/pipe/ff/td_kinect_registration.json

Usage:
    python tools/ff/ff_calibrate.py              # webcam+projector+kinect
    python tools/ff/ff_calibrate.py --no-kinect  # webcam+projector only

Requires: TD running with the ff network (camLogi, kinectC, final_switch,
ff_calibration_pattern inputs available) and the ff target still on the table.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from ff_calibration_io import PIPE, camera_record, publish_documents

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'smoke'
TAG = time.strftime('ff_%Y%m%d_%H%M%S')
TD = ['node', str(ROOT / 'tools/td/td.mjs')]

# ----------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------
SPEC_PATH    = ROOT / 'tools/pipe/ff/fulcrum_spec.json'  # measured ff geometry + markers 10-14
WEBCAM_K_SRC = ROOT / 'tools/pipe/ff/cheb_view.json'      # webcam f_px (640x480, cx=320 cy=240)
PROJ_PRIOR   = ROOT / 'tools/pipe/ff/aruco_solve_frozen.json'  # frozen projector K prior
CALIB_OUT    = ROOT / 'tools/pipe/ff/calibration_current.json'     # canonical pointer the runtime reads
KINECT_OUT   = ROOT / 'tools/pipe/ff/td_kinect_registration.json'  # webcam_from_kinect registration

FUL_IDS = set(range(10, 15))       # physical ff markers
PATT_IDS = list(range(20, 44))     # projected calibration markers (6x4)
MIN_PROJ_GROUPS = 6                # projected markers needed for the solve
MIN_DOTS, MIN_DOT_FACES = 9, 2     # accepted face dots needed
HELD_OUT_MEAN, HELD_OUT_MAX = 5.0, 12.0   # validation gates (px)
OUTLINE_COVERAGE = 0.9             # wireframe check: >90% within 4px
DRIFT_PX = 2.0                     # ff must not move during the dot sweep

# face name -> physical marker id, and the real DICT_4X4_50 4x4 data bits
# (bit=1 is a WHITE printed cell -> reliable dot target; same tables as the shaders)
FACE_ID = {'+y': 11, '+x': 12, '-y': 13, '-x': 14, 'top': 10}
BITS = {
    11: [0,0,0,1, 0,0,0,1, 1,0,1,0, 0,1,1,1],
    12: [0,0,0,0, 1,1,1,0, 1,0,1,1, 0,1,1,1],
    13: [0,0,1,0, 1,0,1,0, 0,0,0,0, 1,1,1,1],
    14: [0,0,1,0, 0,1,0,0, 1,0,1,1, 0,0,0,1],
    10: [1,1,1,1, 1,0,0,1, 1,0,0,1, 0,0,0,1],
}


# ----------------------------------------------------------------------
# Shared helpers (self-contained — replaces fulcrum_track/fulcrum_check)
# ----------------------------------------------------------------------
_spec = json.loads(SPEC_PATH.read_text())
WORLD = {int(i): np.float64(m['corners_TL_TR_BR_BL'])
         for i, m in _spec['markers'].items()}
_m = _spec['measured']


_PAR = cv2.aruco.DetectorParameters()
_PAR.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
_PAR.minOtsuStdDev = 2.0
_PAR.errorCorrectionRate = 0.8
_DET = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), _PAR)


def detect(gray):
    """DICT_4X4_50 marker detect -> {id: (4,2) TL,TR,BR,BL}."""
    corners, ids, _ = _DET.detectMarkers(gray)
    return {} if ids is None else {int(i): corners[k].reshape(4, 2)
                                   for k, i in enumerate(ids.ravel())}


def face_polys():
    """Fulcrum-frame 3D polys: 4 side-face trapezoids + top square."""
    ht, hb, H = _m['top_edge'] / 2, _m['base_edge'] / 2, _m['height']
    faces = {}
    for name, n in (('+y', (0, 1)), ('+x', (1, 0)), ('-y', (0, -1)), ('-x', (-1, 0))):
        t = np.array([-n[1], n[0], 0.0])
        top, base = np.array([n[0] * ht, n[1] * ht, H]), np.array([n[0] * hb, n[1] * hb, 0])
        faces[name] = np.float64([top + t * ht, base + t * hb,
                                  base - t * hb, top - t * ht])
    faces['top'] = np.float64([[-ht, ht, H], [ht, ht, H], [ht, -ht, H], [-ht, -ht, H]])
    return faces


def rt(rv, tv):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.float64(rv).reshape(3))[0]
    T[:3, 3] = np.float64(tv).reshape(3)
    return T


def solve_fulcrum(world, found, K):
    """Non-planar solvePnP against the physical ff markers -> (rv,tv,mean,max), found."""
    ful = {i: c for i, c in found.items() if i in world}
    if not ful:
        return None, ful
    obj = np.vstack([world[i] for i in ful])
    img = np.vstack(list(ful.values()))
    ok, rv, tv = cv2.solvePnP(obj, img, K, np.zeros(5), flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None, ful
    rv, tv = cv2.solvePnPRefineLM(obj, img, K, np.zeros(5), rv, tv)
    pred = cv2.projectPoints(obj, rv, tv, K, np.zeros(5))[0].reshape(-1, 2)
    e = np.linalg.norm(pred - img, axis=1)
    return (rv, tv, float(e.mean()), float(e.max())), ful


def marker_frame(marker_id):
    """Marker frame from spec corners: centre, u/v unit axes, half-sizes."""
    c = WORLD[marker_id]                     # (4,3) TL,TR,BR_BL in ff frame
    MC = c.mean(axis=0)
    u = (c[1] - c[0]) + (c[2] - c[3])
    v = (c[3] - c[0]) + (c[2] - c[1])
    hu, hv = np.linalg.norm(u) / 4, np.linalg.norm(v) / 4
    return MC, u / np.linalg.norm(u), v / np.linalg.norm(v), hu, hv


def face_targets(face_name, q):
    """Sweep targets on this face, anchored on its ArUco marker:
    marker corners + white-cell centres (the tracked structure itself),
    plus four bare-face fill points.  Returns [(point3d, kind, weight)]."""
    mid = FACE_ID[face_name]
    MC, MU, MV, hu, hv = marker_frame(mid)
    targets = [(MC + (c - MC) * 0.97, 'corner', 3.0) for c in WORLD[mid]]
    for i, b in enumerate(BITS[mid]):
        if not b:
            continue
        u, v = ((i % 4) - 1.5) / 3.0, ((i // 4) - 1.5) / 3.0
        targets.append((MC + u * hu * MU + v * hv * MV, 'cell', 3.0))
    for u, v in ((0.06, 0.5), (0.94, 0.5), (0.5, 0.06), (0.5, 0.94)):
        pt = (1 - v) * ((1 - u) * q[0] + u * q[3]) + \
             v * ((1 - u) * q[1] + u * q[2])
        targets.append((pt, 'face', 1.5))
    return targets


def ray_surface(origin, direction, polys):
    """First surface a ray pierces: an ff face poly, else 'table', else None."""
    best, bs = None, np.inf
    for name, q in polys.items():
        normal = np.cross(q[1] - q[0], q[2] - q[0])
        denom = direction @ normal
        if abs(denom) < 1e-9:
            continue
        dist = ((q[0] - origin) @ normal) / denom
        pt = origin + dist * direction
        sides = np.array([np.cross(q[(i + 1) % 4] - q[i], pt - q[i]) @ normal
                          for i in range(4)])
        if 0 < dist < bs and (np.all(sides >= -1e-5) or np.all(sides <= 1e-5)):
            bs, best = dist, name
    if best is not None:
        return best
    if abs(direction[2]) > 1e-9:
        dist = -origin[2] / direction[2]
        pt = origin + dist * direction
        if dist > 0 and abs(pt[0]) < 320 and abs(pt[1]) < 320:
            return 'table'
    return None


def hit(uv, K, rv, tv, polys):
    """Back-project a webcam px: ray -> nearest ff face, else table plane z=0."""
    R = cv2.Rodrigues(rv)[0]
    origin = -R.T @ tv.reshape(3)
    direction = R.T @ np.linalg.solve(K, [*uv, 1.0])
    hits = []
    for name, q in polys.items():
        normal = np.cross(q[1] - q[0], q[2] - q[0])
        denom = direction @ normal
        if abs(denom) < 1e-9:
            continue
        dist = ((q[0] - origin) @ normal) / denom
        pt = origin + dist * direction
        sides = np.array([np.cross(q[(i + 1) % 4] - q[i], pt - q[i]) @ normal
                          for i in range(4)])
        if dist > 0 and (np.all(sides >= -1e-5) or np.all(sides <= 1e-5)):
            hits.append((dist, pt, name))
    if hits:
        _, pt, name = min(hits, key=lambda x: x[0])
        return pt, name
    dist = -origin[2] / direction[2]
    if dist <= 0:
        raise RuntimeError('Ray misses table')
    return origin + dist * direction, 'table'


# ----------------------------------------------------------------------
# TD bridge + capture
# ----------------------------------------------------------------------
def td(code, mode='--exec'):
    p = subprocess.run(TD + [mode, code], cwd=ROOT, capture_output=True,
                       text=True, timeout=25)
    if p.returncode:
        raise RuntimeError(p.stderr or p.stdout)
    return p.stdout.strip()


def capture(name, label):
    path = OUT / f'{TAG}_{label}_{name}.png'
    image = wire_client().capture_color('/project1/' + name)
    OUT.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f'Could not save diagnostic frame: {path}')
    return image


def wire_client():
    sys.path.insert(0, str(ROOT / 'tools/hg'))
    from hg import TdClient
    return TdClient(helper=str(HERE / 'ff_td.py'))


def set_pattern(image):
    sys.path.insert(0, str(ROOT / 'tools/hg'))
    import hg_wire
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    wire_client().rpc({
        'action': 'pattern', 'owner': _ff_session_owner,
        'pattern': hg_wire.encode_array(image)})


# ======================================================================
# Phase 1 — webcam -> ff pose from the physical markers
# ======================================================================
def phase_webcam(K, capture_kinect=True):
    td("op('/project1/cam_and_kinect').par.index=0")
    td("op('/project1/final_switch').par.index=0")
    time.sleep(0.7)
    # the camera's auto-exposure needs a moment after the bright projection
    # turns off — retry a few captures and keep the best detection
    best = None
    for attempt in range(4):
        clean = capture('camLogi', 'clean')
        if attempt == 0 and capture_kinect:
            capture('kinectC', 'clean')
        if clean.shape[:2] != (480, 640):
            raise RuntimeError('Webcam resolution differs from intrinsic estimate')
        found = detect(cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY))
        ful = {i: c for i, c in found.items() if i in WORLD}
        if best is None or len(ful) > len(best[1]):
            best = (clean, ful)
        if len(ful) >= 3:
            break
        time.sleep(0.6)
    solution, found = solve_fulcrum(WORLD, best[1], K)
    if solution is None or len(found) < 3:
        raise RuntimeError(f'Need at least three physical ff markers (best attempt: {sorted(best[1])})')
    rv, tv, mean, maximum = solution
    if mean > 3 or maximum > 8:
        raise RuntimeError(f'ff pose residual too high: {mean:.2f}/{maximum:.2f}px')
    print(f'  webcam: markers {sorted(found)}  reproj {mean:.2f}/{maximum:.2f}px')
    # the ff markers ARE the calibration target: refine the webcam intrinsics
    # against their known 3D corners with soft priors toward the estimate
    obj = np.vstack([WORLD[i] for i in found])
    img = np.vstack(list(found.values()))
    initial = np.r_[rv.ravel(), tv.ravel(), K[0, 0], K[0, 2], K[1, 2]]

    def residual(p):
        k = np.float64([[p[6], 0, p[7]], [0, p[6], p[8]], [0, 0, 1]])
        pred = cv2.projectPoints(obj, p[:3], p[3:6], k,
                                 np.zeros(5))[0].reshape(-1, 2)
        return np.r_[(pred - img).ravel(),
                     (p[6] - initial[6]) / 25.0,
                     (p[7] - initial[7]) / 10.0,
                     (p[8] - initial[8]) / 10.0]
    fit = least_squares(residual, initial, loss='soft_l1', f_scale=2,
                        bounds=([-np.inf] * 6 + [400, 290, 210],
                                [np.inf] * 6 + [680, 350, 270]), max_nfev=250)
    K2 = np.float64([[fit.x[6], 0, fit.x[7]], [0, fit.x[6], fit.x[8]], [0, 0, 1]])
    rv, tv = fit.x[:3].reshape(3, 1), fit.x[3:6].reshape(3, 1)
    pred = cv2.projectPoints(obj, rv, tv, K2, np.zeros(5))[0].reshape(-1, 2)
    e2 = np.linalg.norm(pred - img, axis=1)
    print(f'  webcam K refined: f {K[0,0]:.1f}->{K2[0,0]:.1f}  '
          f'cx {K[0,2]:.1f}->{K2[0,2]:.1f}  cy {K[1,2]:.1f}->{K2[1,2]:.1f}  '
          f'reproj {mean:.2f}->{e2.mean():.2f}px')
    return rv, tv, mean, maximum, found, K2


# ======================================================================
# Phase 2 — projected ArUco grid -> 3D<->projector-px correspondences
# ======================================================================
def project_pattern():
    """6x4 grid of id20-43 markers at known canvas positions -> pattern image + corners."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    pattern = np.full((720, 1280), 255, np.uint8)
    corners = {}
    for row in range(4):
        for col in range(6):
            i = 20 + row * 6 + col
            x, y, size = 45 + col * 205, 30 + row * 175, 130
            pattern[y:y + size, x:x + size] = cv2.aruco.generateImageMarker(dictionary, i, size)
            corners[i] = np.float64([[x, y], [x + size - 1, y],
                                     [x + size - 1, y + size - 1], [x, y + size - 1]])
    path = OUT / f'{TAG}_pattern.png'
    cv2.imwrite(str(path), pattern)
    set_pattern(pattern)
    time.sleep(1.2)
    return corners


def phase_correspondences(K, rv, tv, polys):
    corners = project_pattern()
    # autofocus/exposure can still be settling on the first frame — keep the
    # capture with the most decoded projected markers
    projected, detected = None, {}
    for _ in range(3):
        frame = capture('camLogi', 'pattern')
        found = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if len(found) > len(detected):
            projected, detected = frame, found
        time.sleep(0.3)
    capture('kinectC', 'pattern')
    obj, img, groups, surfaces = [], [], [], []
    corrs = []
    for i, cam_corners in detected.items():
        if i not in corners:
            continue
        for uv, proj_uv in zip(cam_corners, corners[i]):
            pt, surface = hit(uv, K, rv, tv, polys)
            obj.append(pt)
            img.append(proj_uv)
            groups.append(i)
            surfaces.append(surface)
            # raw correspondence for the joint fit: camera px <-> projector px
            # table corrs carry reliable surface labels — they anchor the fit
            corrs.append((uv.copy(), proj_uv.copy(), surface, 2.5, i))
    print(f'  correspondences: {len(obj)} pts, {len(set(groups))} markers, '
          f'surfaces {{{", ".join(f"{s}:{surfaces.count(s)}" for s in sorted(set(surfaces)))}}}')
    if len(set(groups)) < MIN_PROJ_GROUPS:
        raise RuntimeError('Insufficient projected marker coverage')
    return np.float64(obj), np.float64(img), groups, surfaces, corrs


# ======================================================================
# Phase 3 — projector solve (seed PnP -> least-squares, held-out gate)
# ======================================================================
def _frozen_projector_pos_table():
    """Projector origin in the frozen table frame, derived from its pose.

    Do not use projector.cam_pos_world_mm: the stored convenience field is
    sign-inconsistent with the rvec/tvec in aruco_solve_frozen.json.
    """
    frozen = json.loads(PROJ_PRIOR.read_text())['projector']
    R = cv2.Rodrigues(np.float64(frozen['rvec']))[0]
    return -R.T @ np.float64(frozen['tvec'])


def _proj_pos_prior(rv, tv):
    """Expected projector position in ff frame, from the frozen table solve:
    the rig is fixed, so the projector's table position is known and only
    the ff's placement on the table varies between runs."""
    frozen = json.loads(PROJ_PRIOR.read_text())
    T_table_cam = np.linalg.inv(rt(frozen['webcam']['rvec'], frozen['webcam']['tvec']))
    T_table_ff = T_table_cam @ rt(rv, tv)
    return np.linalg.inv(T_table_ff) @ np.r_[_frozen_projector_pos_table(), 1.0]


def _pos_prior_residual(p, pos_ff_prior):
    R = cv2.Rodrigues(p[:3])[0]
    pos_ff = -R.T @ p[3:6]
    return (pos_ff - pos_ff_prior[:3]) / 40.0


def phase_projector(obj, img, groups, rv, tv):
    prior = json.loads(PROJ_PRIOR.read_text())['projector']
    KP = np.float64([[prior['f_px'], 0, prior['cx']],
                     [0, prior['f_px'], prior['cy']], [0, 0, 1]])
    pos_ff_prior = _proj_pos_prior(rv, tv)
    train = np.array([i % 3 != 0 for i in groups])
    if train.sum() < 8 or (~train).sum() < 4:
        raise RuntimeError('Insufficient marker groups for held-out validation')
    ok, pr, pt = cv2.solvePnP(obj[train], img[train], KP, np.zeros(5),
                              flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise RuntimeError('Projector solvePnP failed')
    pr, pt = cv2.solvePnPRefineLM(obj[train], img[train], KP, np.zeros(5), pr, pt)
    pred = cv2.projectPoints(obj, pr, pt, KP, np.zeros(5))[0].reshape(-1, 2)
    errors = np.linalg.norm(pred - img, axis=1)

    initial = np.r_[pr.ravel(), pt.ravel(), KP[0, 0], KP[1, 2]]
    def residual(p):
        k = KP.copy()
        k[0, 0] = k[1, 1] = p[6]
        k[1, 2] = p[7]
        pred = cv2.projectPoints(obj[train], p[:3], p[3:6], k,
                                 np.zeros(5))[0].reshape(-1, 2)
        return np.r_[(pred - img[train]).ravel(),
                     (p[6] - initial[6]) / 500, (p[7] - initial[7]) / 300,
                     _pos_prior_residual(p, pos_ff_prior)]
    refined = least_squares(residual, initial, loss='soft_l1', f_scale=3,
                            bounds=([-np.inf] * 6 + [800, -1500],
                                    [np.inf] * 6 + [2600, 720]), max_nfev=300)
    cand_K = KP.copy()
    cand_K[0, 0] = cand_K[1, 1] = refined.x[6]
    cand_K[1, 2] = refined.x[7]
    cand_pred = cv2.projectPoints(obj, refined.x[:3], refined.x[3:6], cand_K,
                                  np.zeros(5))[0].reshape(-1, 2)
    cand_errors = np.linalg.norm(cand_pred - img, axis=1)
    refined_used = cand_errors[~train].mean() < errors[~train].mean()
    if refined_used:
        KP, pr, pt, errors = cand_K, refined.x[:3], refined.x[3:6], cand_errors
    print(f'  projector: held-out {errors[~train].mean():.2f}/{errors[~train].max():.2f}px  '
          f'f={KP[0,0]:.0f}')
    return KP, pr, pt, errors, train, refined_used


# ======================================================================
# Phase 4 — face-dot sweep (extra correspondences + drift check)
# ======================================================================
def phase_face_dots(K, rv, tv, KP, pr, pt, polys, label='dots', gid_base=100):
    observations, survey = [], []
    black = OUT / f'{TAG}_{label}_black.png'
    cv2.imwrite(str(black), np.zeros((720, 1280), np.uint8))
    set_pattern(np.zeros((720, 1280), np.uint8))
    time.sleep(0.5)
    baseline = cv2.cvtColor(capture('camLogi', f'{label}_black'), cv2.COLOR_BGR2GRAY)
    for face_name, q in polys.items():
        for point, kind, weight in face_targets(face_name, q):
            puv = cv2.projectPoints(point.reshape(1, 3), pr, pt, KP,
                                    np.zeros(5))[0].reshape(2)
            cuv = cv2.projectPoints(point.reshape(1, 3), rv, tv, K,
                                    np.zeros(5))[0].reshape(2)
            if not (12 < puv[0] < 1268 and 12 < puv[1] < 708
                    and 12 < cuv[0] < 628 and 12 < cuv[1] < 468):
                continue
            # No seed-model self-consistency pruning: wherever the dot really
            # lands, the (camera-px, projector-px) pair is still an honest
            # correspondence on its MEASURED surface.
            index = len(survey)
            command = np.round(puv).astype(int)
            pattern = np.zeros((720, 1280), np.uint8)
            cv2.circle(pattern, tuple(command), 8, 255, -1)
            path = OUT / f'{TAG}_{label}_{index:02d}.png'
            cv2.imwrite(str(path), pattern)
            set_pattern(pattern)
            time.sleep(0.35)
            frame = capture('camLogi', f'{label}_{index:02d}')
            delta = cv2.subtract(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), baseline)
            yy, xx = np.indices(delta.shape)
            # Wide ROI for a bad seed, but isolate the bright core: a white
            # dot on a white marker cell can merge into a >2k px component at
            # the base threshold.  High adaptive thresholds split it back out;
            # fall back for dim dots.  Never cap area — the cap selected tiny
            # edge artifacts and produced 20-40 mm false correspondences.
            roi = (xx - cuv[0]) ** 2 + (yy - cuv[1]) ** 2 < 32 ** 2
            peak = float(delta[roi].max())
            chosen = None
            for fraction in (0.82, 0.65, 0.45, 0.0):
                threshold = max(28.0, fraction * peak)
                mask = ((delta > threshold) & roi).astype(np.uint8)
                count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
                cands = [j for j in range(1, count)
                         if stats[j, cv2.CC_STAT_AREA] >= 3]
                if cands:
                    j = min(cands, key=lambda c: np.hypot(*(centers[c] - cuv)))
                    chosen = (j, labels, stats, centers)
                    break
            entry = dict(face=face_name, kind=kind,
                         projector_px=command.tolist(),
                         expected_cam_px=cuv.tolist())
            if chosen is not None:
                j, labels, stats, centers = chosen
                weights = delta * (labels == j)
                centroid = np.array([(weights * xx).sum(),
                                     (weights * yy).sum()]) / weights.sum()
                measured, measured_face = hit(centroid, K, rv, tv, polys)
                dist_mm = float(np.linalg.norm(measured - point))
                # label reliability: require the measured point to sit >=5mm
                # inside the measured face's polygon (edge-near hits carry
                # ambiguous surface labels and poison the fit)
                margin = -1.0
                if measured_face != '?' and measured_face != 'table':
                    q = polys[measured_face]
                    n = np.cross(q[1] - q[0], q[2] - q[0])
                    n = n / np.linalg.norm(n)
                    side = np.array([
                        np.cross(q[(k + 1) % 4] - q[k], measured - q[k]) @ n
                        / np.linalg.norm(q[(k + 1) % 4] - q[k])
                        for k in range(4)])
                    if np.all(side >= 0):
                        margin = float(side.min())
                    elif np.all(side <= 0):
                        margin = float(-side.max())
                entry.update(camera_px=centroid.tolist(),
                             measured_face=measured_face,
                             distance_mm=round(dist_mm, 1),
                             margin_mm=round(margin, 1),
                             area=int(stats[j, cv2.CC_STAT_AREA]))
                reliable = (measured_face == 'table' or margin >= 5.0)
                if measured_face != '?' and dist_mm < 60 and reliable:
                    w_eff = weight * (1.0 if measured_face == face_name
                                      and dist_mm < 14 else 0.7)
                    observations.append((centroid.copy(), command, gid_base + index,
                                         measured_face, w_eff))
                    entry['accepted'] = True
            survey.append(entry)
            print('    dot', index, face_name, kind,
                  'accepted' if entry.get('accepted') else 'rejected', flush=True)
    set_pattern(np.zeros((720, 1280), np.uint8))
    time.sleep(0.4)
    final = capture('camLogi', f'{label}_end')
    return observations, survey, final


# ======================================================================
# Phase 5 — final weighted fit + validation gates
# ======================================================================
def _plane_of(surface, polys):
    """Plane (unit normal, d) with n.x + d = 0 for an ff face or the table."""
    if surface == 'table':
        return np.array([0.0, 0.0, 1.0]), 0.0
    q = polys[surface]
    n = np.cross(q[1] - q[0], q[2] - q[0])
    n = n / np.linalg.norm(n)
    return n, -n @ q[0]


def phase_joint_fit(K, rv, tv, KP, pr, pt, marker_obj, marker_img,
                    corrs, polys, pos_ff_prior, prior_fcy):
    """Projector solve anchored on the ff markers + weak rig priors.

    The webcam (pose AND intrinsics) is frozen at the spec-anchored marker
    solve — the spec is trusted, so the marker-anchored K is the camera.
    The projector is seeded from the phase-3 correspondence solve; the frozen
    projector only supplies weak f/cy/position regularization.

      p = [proj_rv, proj_tv, proj_f, proj_cy]   (8 dof)

    Each correspondence residual is cam-px -> ray -> surface plane -> 3D ->
    projector px; 3D points are ray-cast per eval, never precomputed.
    """
    uc = np.float64([c[0] for c in corrs])           # webcam px
    up = np.float64([c[1] for c in corrs])           # projector px
    surf = np.array([c[2] for c in corrs], dtype=object)
    w = np.float64([c[3] for c in corrs])
    gid = np.array([c[4] for c in corrs])
    train = np.array([g % 3 != 0 for g in gid])
    valid = np.ones(len(corrs), bool)
    fit_mask = train.copy()
    Rc = cv2.Rodrigues(rv)[0]
    oc = -Rc.T @ tv.ravel()
    invK = np.linalg.inv(K)
    # Seed with the live phase-3 solve, not the frozen table solve: the frozen
    # pose is only a weak physical prior and may be stale/sign-inconsistent.
    initial = np.r_[pr.ravel(), pt.ravel(), KP[0, 0], KP[1, 2]]

    def corr_residuals(p, idx):
        planes = {s: _plane_of(s, polys) for s in set(surf[idx])}
        Rp = cv2.Rodrigues(p[:3])[0]
        tp = p[3:6]
        rays = (Rc.T @ (invK @ np.c_[uc[idx], np.ones(idx.sum())].T)).T
        res = np.zeros((int(idx.sum()), 2))
        order = np.nonzero(idx)[0]
        for s in set(surf[idx]):
            sel_local = np.nonzero(surf[idx] == s)[0]
            n, d = planes[s]
            denom = rays[sel_local] @ n
            dist = np.where(np.abs(denom) > 1e-9,
                            -(n @ oc + d) / np.where(np.abs(denom) > 1e-9, denom, 1.0),
                            0.0)
            pts = oc + dist[:, None] * rays[sel_local]
            q = (Rp @ pts.T).T + tp
            qz = np.where(np.abs(q[:, 2]) > 1e-6, q[:, 2], 1.0)
            uv = np.c_[p[6] * q[:, 0] / qz + 640.0,
                       p[6] * q[:, 1] / qz + p[7]]
            bad = (np.abs(denom) <= 1e-9) | (np.abs(q[:, 2]) <= 1e-6)
            uv[bad] = up[order[sel_local][bad]]
            res[sel_local] = uv - up[order[sel_local]]
        return res

    def residual(p):
        rb = corr_residuals(p, fit_mask) * w[fit_mask, None]
        pos_ff = -cv2.Rodrigues(p[:3])[0].T @ p[3:6]
        pri = np.r_[(p[6] - prior_fcy[0]) / 300.0,
                    (p[7] - prior_fcy[1]) / 400.0,
                    (pos_ff - pos_ff_prior[:3]) / 150.0]
        return np.r_[rb.ravel(), pri]

    # Loose bounds around the live seed.  The frozen mount prior regularizes
    # the fit but must not hard-bound it: stale/sign-inconsistent pose fields
    # otherwise drag every correspondence to the boundary.
    rv0, tv0, f0, cy0 = initial[:3], initial[3:6], initial[6], initial[7]
    lower = np.r_[rv0 - 0.35, tv0 - 300.0, 500.0, -1800.0]
    upper = np.r_[rv0 + 0.35, tv0 + 300.0, 5000.0, 720.0]
    x0 = np.clip(initial, lower, upper)
    # diagnostic: per-surface residual under the seed before fitting
    e0 = np.linalg.norm(corr_residuals(x0, np.ones(len(uc), bool)), axis=1)
    for s in sorted(set(surf)):
        sel = surf == s
        print(f'    seed residual {s}: n={sel.sum()} '
              f'median={np.median(e0[sel]):.1f}px max={e0[sel].max():.1f}px')
    fit = least_squares(residual, x0, loss='soft_l1', f_scale=3,
                        bounds=(lower, upper), max_nfev=700)
    p = fit.x
    all_err = np.linalg.norm(corr_residuals(p, np.ones(len(uc), bool)), axis=1)
    # Reject correspondences that remain extreme after each robust pass.
    # A first trim can expose one more inconsistent point in the refit.
    trim_cap = max(4, int(len(corrs) * 0.25))
    for _ in range(2):
        bad = (all_err > 12.0) & valid
        if not bad.any() or (~valid).sum() + bad.sum() > trim_cap:
            break
        print(f'    rejected {bad.sum()} correspondences >12px')
        valid[bad] = False
        fit_mask = train & valid
        fit = least_squares(residual, p, loss='soft_l1', f_scale=3,
                            bounds=(lower, upper), max_nfev=700)
        p = fit.x
        all_err = np.linalg.norm(corr_residuals(p, np.ones(len(uc), bool)), axis=1)
    KP2 = np.float64([[p[6], 0, 640.0], [0, p[6], p[7]], [0, 0, 1]])
    pr2, pt2 = p[:3], p[3:6]
    mk = np.linalg.norm(
        cv2.projectPoints(marker_obj, rv, tv, K, np.zeros(5))[0].reshape(-1, 2)
        - marker_img, axis=1)
    heldout_eval = (~train) & valid
    dot_test = heldout_eval & (gid >= 100) & (surf != 'table')
    if not dot_test.any():
        raise RuntimeError('No withheld face dots')
    print(f'  joint fit: markers {mk.mean():.2f}/{mk.max():.2f}px  '
          f'corr held-out {all_err[heldout_eval].mean():.2f}/{all_err[heldout_eval].max():.2f}px  '
          f'proj f={p[6]:.0f} cy={p[7]:.0f}')
    return K, rv, tv, KP2, pr2, pt2, all_err, train, dot_test, valid


def phase_outline_assert(K, rv, tv, KP, pr, pt, polys):
    """Project face wireframes, capture webcam, assert overlap."""
    overlay = np.zeros((720, 1280, 3), np.uint8)
    for face in polys.values():
        uv = cv2.projectPoints(face, pr, pt, KP, np.zeros(5))[0].reshape(-1, 2)
        cv2.polylines(overlay, [np.round(uv).astype(np.int32)], True, (0, 255, 0), 2)
    path = OUT / f'{TAG}_outline.png'
    cv2.imwrite(str(path), overlay)
    set_pattern(overlay)
    time.sleep(0.8)
    validation = capture('camLogi', 'validation')
    capture('kinectC', 'validation')
    expected = np.zeros(validation.shape[:2], np.uint8)
    for face in polys.values():
        uv = cv2.projectPoints(face, rv, tv, K, np.zeros(5))[0].reshape(-1, 2)
        cv2.polylines(expected, [np.round(uv).astype(np.int32)], True, 255, 1)
    b, g, r = [c.astype(np.int16) for c in cv2.split(validation)]
    observed = ((g - r > 25) & (b - r > 15) & (g > 100)).astype(np.uint8)
    observed[cv2.dilate(expected, np.ones((25, 25), np.uint8)) == 0] = 0
    distance = cv2.distanceTransform((observed == 0).astype(np.uint8), cv2.DIST_L2, 3)
    residual = distance[expected > 0]
    return dict(expected_pixels=int(len(residual)),
                observed_pixels=int(observed.sum()),
                mean_distance_cam_px=float(residual.mean()),
                coverage_within_4px=float((residual <= 4).mean()))


# ======================================================================
# Phase 6 — Kinect registration (optional)
# ======================================================================
def depth_factory_intrinsics(sensor=None):
    """Read the SDK depth model, without claiming a measured FF/world pose."""
    owned = sensor is None
    if owned:
        sys.path.insert(0, str(ROOT / 'tools/kinect'))
        from kincap import Kin
        sensor = Kin()
    try:
        intrinsics = sensor.mapper.GetDepthCameraIntrinsics()
        fields = ('FocalLengthX', 'FocalLengthY', 'PrincipalPointX',
                  'PrincipalPointY', 'RadialDistortionSecondOrder',
                  'RadialDistortionFourthOrder', 'RadialDistortionSixthOrder')
        values = {name: float(getattr(intrinsics, name)) for name in fields}
        if not np.isfinite(list(values.values())).all() or min(
                values['FocalLengthX'], values['FocalLengthY']) <= 0:
            raise RuntimeError('Kinect SDK returned invalid depth intrinsics')
        return dict(
            schema='ff.camera_calibration', schema_version=1, camera_id=2,
            timestamp=TAG, source_top='/project1/kinectD', resolution=[512, 424],
            method='kinect_v2_sdk_factory_intrinsics',
            frame='native_sdk_depth_pixels', factory_intrinsics=values,
            K=[[values['FocalLengthX'], 0, values['PrincipalPointX']],
               [0, values['FocalLengthY'], values['PrincipalPointY']], [0, 0, 1]],
            pose_calibrated=False,
            note='Depth is not an RGB marker camera. These are factory intrinsics, '
                 'not a newly measured depth-to-FF pose or a static color-depth map.')
    finally:
        if owned:
            sensor._sensor.Close()


def kinect_factory_models():
    """Reuse the existing SDK color pinhole fit and read depth factory data."""
    sys.path.insert(0, str(ROOT / 'tools/kinect'))
    try:
        from kincap import Kin
    except Exception as e:
        raise RuntimeError(f'Kinect SDK unavailable: {e}') from e
    sensor = Kin()
    try:
        time.sleep(2)
        pts = np.float32([(x, y, z) for z in (0.65, 0.9, 1.3, 1.8)
                          for y in np.linspace(-0.3, 0.3, 9)
                          for x in np.linspace(-0.5, 0.5, 13)])
        pixels = sensor.camera_pts_to_color(pts * np.float32([1, -1, 1]))
        good = np.isfinite(pixels).all(axis=1) & (pixels[:, 0] > 0) & \
            (pixels[:, 0] < 1920) & (pixels[:, 1] > 0) & (pixels[:, 1] < 1080)
        pts, pixels = pts[good], pixels[good]
        K0 = np.float64([[1060, 0, 1004], [0, 1060, 548], [0, 0, 1]])
        rms, K_k, D, _, _ = cv2.calibrateCamera(
            [pts], [pixels], (1920, 1080), K0, np.zeros(5),
            flags=cv2.CALIB_USE_INTRINSIC_GUESS)
        if rms > 0.5 or len(pts) < 200:
            raise RuntimeError(f'Factory mapper pinhole fit failed: rms={rms}, pts={len(pts)}')
        K_k[:2] *= 2 / 3
        print(f'  kinect: factory mapper rms={rms:.3f}px')
        return K_k, D, rms, depth_factory_intrinsics(sensor)
    finally:
        sensor._sensor.Close()


def phase_kinect(calibration, spec_world):
    """Fresh paired poses; return data without publishing a partial rig."""
    K_k, D, rms, depth = kinect_factory_models()
    td("op('/project1/cam_and_kinect').par.index=1")
    td("op('/project1/final_switch').par.index=0; op('/project1/OUT').cook(force=True); "
       "op('/project1/kinectC').cook(force=True)")
    time.sleep(0.7)
    webcam = capture('camLogi', 'kinref')
    kinect = capture('kinectC', 'kinref')
    if kinect.shape[:2] != (720, 1280):
        raise RuntimeError('Kinect resolution must be 1280x720')

    KW = np.float64(calibration['webcam']['K'])
    WC, ws, wfound = _paired_pose(webcam, spec_world, KW, np.zeros(5))
    KC, ks, kfound = _paired_pose(kinect, spec_world, K_k, D)
    _, _, after = _paired_pose(
        cv2.imread(str(capture_path('camLogi', 'kinref_after'))),
        spec_world, KW, np.zeros(5))
    drift = max(np.linalg.norm(wfound[i] - after[i], axis=1).max()
                for i in set(wfound) & set(after))
    if drift > 1.5:
        raise RuntimeError(f'ff moved during registration: {drift:.2f}px; keep target still')
    result = dict(timestamp=TAG, source='kinectC', resolution=[1280, 720],
                  K=K_k.tolist(), D=D.ravel().tolist(),
                  sdk_mapper_fit_rms_px=rms, webcam=ws, kinect=ks,
                  reference_drift_px=float(drift),
                  webcam_from_kinect=(WC @ np.linalg.inv(KC)).tolist(),
                  camera_from_reference=KC.tolist(),
                  calibration_used=str(CALIB_OUT), depth_factory=depth)
    print(f'  kinect: registration measured (drift {drift:.2f}px)')
    return result


def capture_path(name, label):
    path = OUT / f'{TAG}_{label}_{name}.png'
    capture(name, label)
    return path


def _paired_pose(image, world, K, distortion):
    found = {i: c for i, c in detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)).items()
             if i in world}
    if len(found) < 3:
        raise RuntimeError(f'Need three ff markers, found {sorted(found)}')
    obj, img = np.vstack([world[i] for i in found]), np.vstack(list(found.values()))
    ok, rv, tv = cv2.solvePnP(obj, img, K, distortion, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise RuntimeError('Reference pose failed')
    rv, tv = cv2.solvePnPRefineLM(obj, img, K, distortion, rv, tv)
    pred = cv2.projectPoints(obj, rv, tv, K, distortion)[0].reshape(-1, 2)
    e = np.linalg.norm(pred - img, axis=1)
    if e.mean() > 3 or e.max() > 8:
        raise RuntimeError(f'Reference residual too high: {e.mean():.2f}/{e.max():.2f}')
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = cv2.Rodrigues(rv)[0], tv.ravel()
    return T, dict(markers=sorted(found), mean_px=float(e.mean()),
                   max_px=float(e.max())), found


# ======================================================================
# Phase 7 — viewer export: solved rig in the frozen table frame
# ======================================================================
def export_viewer(T_cam_ff, K, KP, pr, pt, registration=None):
    """Write tools/pipe/ff/calib_geometry.json: every solved device expressed in the
    frozen table-board frame (same world the viewer renders) plus the fulcrum
    itself at its solved pose.  Table anchoring: T_table_ff = T_table_cam @
    T_cam_ff, with T_table_cam from the frozen board solve."""
    frozen = json.loads(PROJ_PRIOR.read_text())
    T_table_cam = np.linalg.inv(rt(frozen['webcam']['rvec'], frozen['webcam']['tvec']))
    T_table_ff = T_table_cam @ T_cam_ff
    R_tf, t_tf = T_table_ff[:3, :3], T_table_ff[:3, 3]

    def device(T_dev_ff):
        R_df, t_df = T_dev_ff[:3, :3], T_dev_ff[:3, 3]
        pos_ff = -R_df.T @ t_df
        return dict(
            cam_pos_world_mm=(R_tf @ pos_ff + t_tf).tolist(),
            R=(R_df @ R_tf.T).tolist(),          # table-world -> device rotation
            fwd_world=(R_tf @ (R_df.T @ np.array([0, 0, 1.0]))).tolist())

    T_proj_ff = rt(pr, pt)
    projector = device(T_proj_ff)
    projector.update(f_px=float(KP[0, 0]), cx=float(KP[0, 2]), cy=float(KP[1, 2]))
    # projector px -> table mm: ray-cast a px grid to z=0 (ff frame), then to
    # table coords; the map is exactly homographic (both are planes)
    R_pf, t_pf = T_proj_ff[:3, :3], T_proj_ff[:3, 3]
    o_ff = -R_pf.T @ t_pf
    px, mm = [], []
    for v in np.linspace(0, 719, 10):
        for u in np.linspace(0, 1279, 14):
            d = R_pf.T @ np.array([(u - KP[0, 2]) / KP[0, 0],
                                   (v - KP[1, 2]) / KP[1, 1], 1.0])
            if abs(d[2]) < 1e-9:
                continue
            s = -o_ff[2] / d[2]
            if s <= 0:
                continue
            p_tbl = T_table_ff @ np.r_[o_ff + s * d, 1.0]
            px.append([u, v])
            mm.append(p_tbl[:2])
    H, _ = cv2.findHomography(np.float64(px), np.float64(mm))

    out = dict(timestamp=TAG, source='ff_calibrate.py (ff-marker anchored)',
               world=frozen.get('world'), measured=frozen.get('measured'),
               world_corners_mm=frozen['world_corners_mm'],
               H_px_to_mm=H.tolist(),
               projector=projector,
               webcam=dict(device(T_cam_ff), f_px=float(K[0, 0]),
                           cx=float(K[0, 2]), cy=float(K[1, 2])),
               fulcrum=dict(pos_mm=t_tf.tolist(),
                            rvec=cv2.Rodrigues(R_tf)[0].ravel().tolist(),
                            measured=_spec['measured'],
                            markers={str(i): WORLD[i].tolist()
                                     for i in sorted(WORLD)}))
    if registration is not None:
        reg = registration
        T_wk = np.float64(reg['webcam_from_kinect'])
        out['kinect'] = device(np.linalg.inv(T_wk) @ T_cam_ff)
        out['kinect']['markers'] = reg.get('kinect', {}).get('markers')
        out['kinect'].update(
            f_px=reg['K'][0][0], fy_px=reg['K'][1][1],
            cx=reg['K'][0][2], cy=reg['K'][1][2], resolution=reg['resolution'])
    return out


# ======================================================================
# Orchestration
# ======================================================================
def run_rig(kinect=True):
    polys = face_polys()
    f = json.loads(WEBCAM_K_SRC.read_text())['cam']['f_px']
    K = np.float64([[f, 0, 320], [0, f, 240], [0, 0, 1]])
    result = {'timestamp': TAG, 'units': 'mm', 'frame': _spec['frame'],
              'intrinsics_source': 'cheb_view webcam estimate; frozen projector prior',
              'alignment_nodes_modified': True, 'published': False}
    try:
        print('== phase 1: webcam solve')
        rv, tv, wmean, wmax, wfound, K = phase_webcam(K)
        result['webcam'] = dict(K=K.tolist(), rvec=rv.ravel().tolist(),
                                tvec=tv.ravel().tolist(), markers=sorted(wfound),
                                mean_px=wmean, max_px=wmax)
        result['intrinsics_source'] = 'ff-marker-refined webcam K; frozen projector prior'

        print('== phase 2: correspondences')
        obj, img, groups, surfaces, corrs = phase_correspondences(K, rv, tv, polys)

        print('== phase 3: projector solve')
        pos_ff_prior = _proj_pos_prior(rv, tv)
        KP, pr, pt, errors, train, refined_used = phase_projector(obj, img, groups, rv, tv)
        if refined_used:
            result['intrinsics_source'] += '; projector f/cy refined with soft priors and held-out selection'

        def drift_check(final_image, label=''):
            final_sol, final_found = solve_fulcrum(
                WORLD, detect(cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)), K)
            if final_sol is None or len(final_found) < 3:
                raise RuntimeError('Cannot verify ff stability after dot sweep')
            common = set(wfound) & set(final_found)
            per_marker = sorted(np.linalg.norm(wfound[i] - final_found[i], axis=1).max()
                                for i in common)
            drift = float(np.median(per_marker))
            suffix = f'_{label}' if label else ''
            result[f'marker_drift_cam_px{suffix}'] = drift
            result[f'marker_drift_worst_px{suffix}'] = float(per_marker[-1])
            if drift > DRIFT_PX:
                raise RuntimeError(f'ff/camera moved during sweep: {drift:.2f}px')

        marker_obj = np.vstack([WORLD[i] for i in wfound])
        marker_img = np.vstack(list(wfound.values()))
        frozen = json.loads(PROJ_PRIOR.read_text())
        prior_fcy = (frozen['projector']['f_px'], frozen['projector']['cy'])

        print('== phase 4a: exploratory face-dot sweep')
        dots1, survey1, final1 = phase_face_dots(
            K, rv, tv, KP, pr, pt, polys, label='dots', gid_base=100)
        drift_check(final1)
        dots, survey, final = list(dots1), list(survey1), final1
        result['accepted_face_dots_pass1'] = len(dots1)

        # If the seed was poor, many aimed dots land off their intended face.
        # Fit once, then retarget the same physical marker cells with the
        # improved projector estimate so the second sweep supplies stronger
        # raised-surface constraints.
        try:
            preliminary = phase_joint_fit(
                K, rv, tv, KP, pr, pt, marker_obj, marker_img,
                corrs + [(d[0], d[1], d[3], d[4], d[2]) for d in dots1],
                polys, pos_ff_prior, prior_fcy)
            _, _, _, KP1, pr1, pt1 = preliminary[:6]
            KP, pr, pt = KP1, pr1, pt1
            print('== phase 4b: retargeted face-dot sweep')
            dots2, survey2, final2 = phase_face_dots(
                K, rv, tv, KP1, pr1, pt1, polys,
                label='dots2', gid_base=200)
            drift_check(final2, 'pass2')
            dots.extend(dots2)
            survey.extend(survey2)
            final = final2
            result['accepted_face_dots_pass2'] = len(dots2)
        except RuntimeError as e:
            print('  retargeted sweep skipped:', e)

        result['dot_survey'] = survey
        result['accepted_face_dots'] = len(dots)
        by_face = {}
        for d in dots:
            by_face.setdefault(d[3], []).append(d)
        print('  accepted:', {f: len(v) for f, v in sorted(by_face.items())})
        if len(dots) < MIN_DOTS or len(set(d[3] for d in dots)) < MIN_DOT_FACES:
            result['validated'] = False
            raise RuntimeError('Need at least nine accepted dots on at least two faces')

        print('== phase 5: joint fit + validation')
        fit_corrs = corrs + [(d[0], d[1], d[3], d[4], d[2]) for d in dots]
        K, rv, tv, KP, pr, pt, errors, train, dot_test, valid = phase_joint_fit(
            K, rv, tv, KP, pr, pt, marker_obj, marker_img,
            fit_corrs, polys, pos_ff_prior, prior_fcy)
        heldout_eval = (~train) & valid
        result['webcam'].update(K=K.tolist(), rvec=rv.ravel().tolist(),
                                tvec=tv.ravel().tolist())
        # solved projector position in table frame — makes a degenerate
        # (far/high-f) solution obvious in the report
        T_table_ff = np.linalg.inv(
            rt(frozen['webcam']['rvec'], frozen['webcam']['tvec'])) @ rt(rv, tv)
        R_pf, t_pf = cv2.Rodrigues(pr)[0], pt.ravel()
        proj_tbl = T_table_ff @ np.r_[-R_pf.T @ t_pf, 1.0]
        result['projector_table_pos_mm'] = proj_tbl[:3].tolist()
        print(f'  projector table pos: {[round(v) for v in proj_tbl[:3]]}  '
              f'(prior {[round(v) for v in _frozen_projector_pos_table()]})')
        result['face_dot_validation'] = dict(
            count=int(dot_test.sum()),
            mean_projector_px=float(errors[dot_test].mean()),
            max_projector_px=float(errors[dot_test].max()))
        result['projector'] = dict(
            K=KP.tolist(), rvec=pr.ravel().tolist(), tvec=pt.ravel().tolist(),
            train_mean_px=float(errors[train & valid].mean()),
            heldout_mean_px=float(errors[heldout_eval].mean()),
            heldout_max_px=float(errors[heldout_eval].max()),
            rejected_correspondences=int((~valid).sum()),
            surfaces={s: surfaces.count(s) for s in set(surfaces)},
            points_mm=obj.tolist(), pixels=img.tolist(), marker_ids=groups)
        result['validated'] = bool(
            errors[heldout_eval].mean() < HELD_OUT_MEAN and errors[heldout_eval].max() < HELD_OUT_MAX
            and errors[dot_test].mean() < HELD_OUT_MEAN and errors[dot_test].max() < HELD_OUT_MAX)

        outline = phase_outline_assert(K, rv, tv, KP, pr, pt, polys)
        result['webcam_outline_assertion'] = outline
        result['validated'] = bool(result['validated']
                                   and outline['coverage_within_4px'] > OUTLINE_COVERAGE)
        print(f"  outline coverage {outline['coverage_within_4px']*100:.1f}% within 4px")

        # publish to the canonical pointer ONLY on a validated solve —
        # a failed/non-validated run leaves calibration_current.json on
        # the last good calibration (the archive is always written)
        if result['validated']:
            registration = None
            if kinect:
                print('== phase 6: kinect registration + depth factory model')
                registration = phase_kinect(result, WORLD)
            geometry = export_viewer(
                rt(rv, tv), K, KP, pr, pt, registration=registration)
            documents = {}
            if registration is not None:
                documents['td_kinect_registration.json'] = registration
                documents['camera_c1.json'] = camera_record(
                    sys.modules[__name__], 1, dict(
                        K=registration['K'], D=registration['D'],
                        camera_from_reference=registration['camera_from_reference'],
                        quality=registration['kinect']))
                documents['camera_c2.json'] = registration['depth_factory']
            documents['camera_c0.json'] = camera_record(
                sys.modules[__name__], 0, result['webcam'])
            for name, document in documents.items():
                if name.startswith('camera_c') and document['camera_id'] != 2:
                    document['scope'] = 'joint_rig_snapshot'
            result['kinect_included'] = registration is not None
            result['published'] = True
            documents['calibration_current.json'] = result
            documents['calib_geometry.json'] = geometry
            publish_documents(documents)
            print('== calibration_current.json updated', flush=True)
        else:
            raise RuntimeError('Rig validation failed; shared calibration files are untouched')
    except BaseException as error:
        result['published'] = False
        result['validated'] = False
        result['failure'] = str(error)
        raise
    finally:
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f'{TAG}_solve.json'
        path.write_text(json.dumps(result, indent=2))
        print(f'== wrote archive {path.name}', flush=True)

    print('== done. validated =', result.get('validated'))
    return result


def calibrate_camera(camera_id):
    """Only return the requested camera's record; do not change joint rig data."""
    if camera_id == 2:
        return depth_factory_intrinsics()
    if camera_id == 0:
        f = json.loads(WEBCAM_K_SRC.read_text())['cam']['f_px']
        K = np.float64([[f, 0, 320], [0, f, 240], [0, 0, 1]])
        rv, tv, mean, maximum, found, K = phase_webcam(K, capture_kinect=False)
        return dict(K=K.tolist(), D=np.zeros(5).tolist(),
                    rvec=rv.ravel().tolist(), tvec=tv.ravel().tolist(),
                    markers=sorted(found), mean_px=mean, max_px=maximum)
    if camera_id != 1:
        raise ValueError('Unknown FF camera id')
    K, D, rms, _ = kinect_factory_models()
    td("op('/project1/cam_and_kinect').par.index=1; "
       "op('/project1/final_switch').par.index=0")
    time.sleep(0.7)
    image = cv2.imread(str(capture_path('kinectC', 'camera_only')))
    if image is None or image.shape[:2] != (720, 1280):
        raise RuntimeError('Kinect color must provide a 1280x720 TD frame')
    pose, quality, _ = _paired_pose(image, WORLD, K, D)
    return dict(K=K.tolist(), D=D.ravel().tolist(),
                camera_from_reference=pose.tolist(), quality=quality,
                sdk_mapper_fit_rms_px=rms)


def main():
    from ff_cli import main as cli_main
    return cli_main(['calib', *sys.argv[1:]])


if __name__ == '__main__':
    sys.exit(main())
