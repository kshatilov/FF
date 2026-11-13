"""fulcrum_track.py - realtime fulcrum-frame pose tracking for the rig.

One process owns both cameras (exclusive grabs).  Each frame:
  * webcam + Kinect color -> ArUco detect -> non-planar solvePnP vs
    fulcrum_spec.json (ids 10-14) -> camera pose in the FULCRUM frame
    (origin = fulcrum base centre, +z up, mm).
  * projector calibration by structured light: fulcrum_project.py is
    told (via poses.json "proj_cmd") to emit one white dot at a known
    canvas pixel; each camera diffs consecutive frames, finds the blob,
    back-projects the ray onto the fulcrum faces / table plane (z=0)
    -> canvas-px -> 3D correspondence.  ~30+ correspondences ->
    solvePnP for the projector pose in the fulcrum frame.  Two passes:
    coarse full-canvas grid, then face-targeted dots through the seed.
  * colour assert: each fulcrum face has an assigned projector colour;
    the face's trapezoid is projected into the camera image and the
    dominant saturated hue is compared against the assignment.
  * annotated feeds -> ../td/preview/*.jpg (viewer panels keep working),
    all poses+asserts -> fulcrum_poses.json (polled by the viewer and
    by the face-colour projector, fulcrum_project.py).

Kinect: buffer is mirrored - detect on raw/flipV/flipH and keep the
branch whose solve has the lowest reprojection error (flip-scan on
marker loss).  Webcam: mounted upside down - 180 deg roll is a proper
rotation that solvePnP absorbs; raw detections solve directly.

Handshake: this process owns fulcrum_poses.json ("proj_cmd" tells the
renderer what to show); fulcrum_project.py owns proj_state.json
(which dot is currently displayed).  Persistent all-faces-wrong
asserts while in colour mode auto-retrigger dot calibration.

Control: http://127.0.0.1:8972 /pause /resume /status.  NOTE: 8971 is
preview_feeds' port - the viewer's feed button posts there.  Do NOT
share it: a feed pause used to silently freeze this tracker.
"""
import json
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "kinect"))
sys.path.insert(0, os.path.join(ROOT, "tools", "td"))
sys.path.insert(0, os.path.join(ROOT, "tools", "projector"))
import aruco_solve as A                                    # noqa: E402
from fulcrum_net import MEAS                              # noqa: E402

PIPE_FF = os.path.join(ROOT, "tools", "pipe", "ff")
SPEC = os.path.join(PIPE_FF, "fulcrum_spec.json")
POSES = os.path.join(PIPE_FF, "fulcrum_poses.json")
PROJ_STATE = os.path.join(PIPE_FF, "proj_state.json")
BOARD = os.path.join(PIPE_FF, "aruco_board.json")
CHEB = os.path.join(PIPE_FF, "cheb_view.json")
KREG = os.path.join(PIPE_FF, "td_kinect_registration.json")
PREV = os.path.join(ROOT, "tools", "td", "preview")

WEBCAM_INDEX, WH = 0, (640, 480)
CW, CH = 1920, 1080
FUL_IDS = set(range(10, 15))
BOARD_IDS = set(range(6))
FLIPS = {"raw": None, "flipV": 0, "flipH": 1}
CTRL_PORT = 8972   # 8971 belongs to preview_feeds - the viewer's feed
                   # button posts there; sharing it paused the tracker
PAUSED = threading.Event()
JPEG = [cv2.IMWRITE_JPEG_QUALITY, 75]
DOT_SETTLE_S = 0.15          # wait after dot switch before sampling
DOT_PENDING_S = 0.42         # give up on a dot after this (< dwell)
RECALIB_FRAMES = 150         # all-wrong asserts -> re-run dot calib
QUAD_MARGIN = 0.94           # face paint shrink toward quad centroid

# assigned projector colour per fulcrum face (BGR)
FACE_COLORS = {"+y": (0, 0, 255), "+x": (0, 255, 0),
               "-y": (255, 0, 0), "-x": (0, 255, 255),
               "top": (255, 0, 255)}


# ---- geometry ----------------------------------------------------------
def face_polys():
    """Fulcrum-frame 3D polys: 4 side-face trapezoids + top square."""
    ht, hb, H = MEAS["top"] / 2, MEAS["base"] / 2, MEAS["height"]
    faces = {}
    for name, nh in (("+y", (0, 1)), ("+x", (1, 0)),
                     ("-y", (0, -1)), ("-x", (-1, 0))):
        t = np.array([-nh[1], nh[0], 0.0])
        T = np.array([nh[0] * ht, nh[1] * ht, H])
        B = np.array([nh[0] * hb, nh[1] * hb, 0.0])
        faces[name] = np.float64(
            [T + t * ht, B + t * hb, B - t * hb, T - t * ht])
    faces["top"] = np.float64(
        [[-ht, ht, H], [ht, ht, H], [ht, -ht, H], [-ht, -ht, H]])
    return faces


def project(pts3, K, rv, tv, dist=None):
    p, _ = cv2.projectPoints(np.float64(pts3).reshape(-1, 1, 3),
                             rv, tv, K, np.zeros(5) if dist is None
                             else dist)
    return p.reshape(-1, 2)


# ---- aruco -------------------------------------------------------------
def detect(gray):
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    par = cv2.aruco.DetectorParameters()
    par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    par.minOtsuStdDev = 2.0
    par.errorCorrectionRate = 0.8
    det = cv2.aruco.ArucoDetector(dic, par)
    c, ids, _ = det.detectMarkers(gray)
    return {} if ids is None else {int(i): c[k].reshape(4, 2)
                                   for k, i in enumerate(ids.flatten())}


def solve_fulcrum(world, found, K, guess=None, dist=None):
    D = np.zeros(5) if dist is None else dist
    ful = {i: c for i, c in found.items() if i in world}
    if not ful:
        return None, ful
    obj = np.vstack([world[i] for i in ful]).reshape(-1, 1, 3)
    img = np.vstack([ful[i] for i in ful]).reshape(-1, 1, 2)
    if guess is not None:
        ok, rv, tv = cv2.solvePnP(obj, img, K, D,
                                  guess[0], guess[1],
                                  useExtrinsicGuess=True,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    else:
        ok, rv, tv = cv2.solvePnP(obj, img, K, D,
                                  flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None, ful
    rv, tv = cv2.solvePnPRefineLM(obj, img, K, D, rv, tv)
    pred, _ = cv2.projectPoints(obj, rv, tv, K, D)
    e = np.linalg.norm(pred.reshape(-1, 2) - img.reshape(-1, 2), axis=1)
    return (rv, tv, float(e.mean()), float(e.max())), ful


def k_for_flip(K, name):
    K2 = K.copy()
    if name == "flipH":
        K2[0, 2] = (CW - 1) - K2[0, 2]
    elif name == "flipV":
        K2[1, 2] = (CH - 1) - K2[1, 2]
    return K2


def kinect_intrinsics(kin, frame):
    """Kinect K + distortion.  Prefer the calibrated registration in
    td_kinect_registration.json (K+D fit to the SDK mapper, ~0.2 px rms)
    over the rough linear pinhole guess; scale K to the actual frame
    size (registration was made at 1280x720, frames are 1920x1080)."""
    fallback = A.kinect_pinhole_init(kin)
    try:
        reg = json.load(open(KREG))
        rw, rh = reg["resolution"]
        fw, fh = frame.shape[1], frame.shape[0]
        if abs(fw / rw - fh / rh) > 1e-6:
            return fallback, np.zeros(5)
        K = np.float64(reg["K"])
        K[:2] *= fw / rw
        return K, np.float64(reg["D"]).ravel()[:5]
    except Exception:
        return fallback, np.zeros(5)


def solve_table(board3, found, K, dist=None):
    """Planar table-board solve: pick the IPPE branch that puts the
    camera above the table.  Solved in the SAME image convention as the
    fulcrum solve, so composing them cancels any mirror convention."""
    D = np.zeros(5) if dist is None else dist
    bd = {i: c for i, c in found.items() if i in board3}
    if len(bd) < 2:
        return None
    obj = np.vstack([board3[i] for i in bd]).reshape(-1, 1, 3)
    img = np.vstack([bd[i] for i in bd]).reshape(-1, 1, 2)
    try:
        n, rvs, tvs, _ = cv2.solvePnPGeneric(
            obj, img, K, D, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return None
    best = None
    for k in range(n):
        rv, tv = cv2.solvePnPRefineLM(obj, img, K, D,
                                      rvs[k], tvs[k])
        pos = -cv2.Rodrigues(rv)[0].T @ tv.ravel()
        if pos[2] < 50 or np.abs(pos).max() > 3000:
            continue
        pred, _ = cv2.projectPoints(obj, rv, tv, K, D)
        e = float(np.linalg.norm(
            pred.reshape(-1, 2) - img.reshape(-1, 2), axis=1).mean())
        if best is None or e < best[0]:
            best = (e, rv, tv)
    return best


# ---- structured-light projector calibration ----------------------------
def rt(rv, tv):
    R = cv2.Rodrigues(np.float64(rv).ravel())[0]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.float64(tv).ravel()
    return T


def read_proj_state():
    try:
        return json.load(open(PROJ_STATE))
    except Exception:
        return None


def ray_surface(o, d, polys):
    """Ray (origin o, unit-free dir d, fulcrum frame) -> nearest fulcrum
    face hit, else the table plane z=0.  -> (P3, face_name|'table')."""
    best = None
    for name, q in polys.items():
        n = np.cross(q[1] - q[0], q[2] - q[0])
        n /= np.linalg.norm(n)
        dn = float(d @ n)
        if abs(dn) < 1e-9:
            continue
        t = float((q[0] - o) @ n / dn)
        if t <= 0:
            continue
        P = o + t * d
        ab, *_ = np.linalg.lstsq(
            np.stack([q[3] - q[0], q[1] - q[0]], 1), P - q[0],
            rcond=None)
        if -0.05 <= ab[0] <= 1.05 and -0.05 <= ab[1] <= 1.05:
            if best is None or t < best[0]:
                best = (t, P, name)
    if best is not None:
        return best[1], best[2]
    if abs(d[2]) > 1e-9 and -o[2] / d[2] > 0:
        return o + (-o[2] / d[2]) * d, "table"
    return None, None


def find_dot(before, now):
    """Centroid of the blob that BRIGHTENED between two grayscale
    frames (the projected dot just appeared).  Signed subtract, not
    absdiff - the previous dot vanishing must not produce a blob."""
    d = cv2.subtract(now, before)
    d = cv2.GaussianBlur(d, (5, 5), 0)
    _, m = cv2.threshold(d, 35, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    best, ba = None, 0.0
    for c in cnts:
        a = cv2.contourArea(c)
        if 15 < a < 4000 and a > ba:
            M = cv2.moments(c)
            if M["m00"] > 0:
                best = (M["m10"] / M["m00"], M["m01"] / M["m00"])
                ba = a
    return best


def grid_dots(cols, rows):
    us = np.linspace(60, 1220, cols)
    vs = np.linspace(60, 660, rows)
    return [[float(u), float(v)] for v in vs for u in us]


def hull_dots(obs, cols=9, rows=7):
    """Dense canvas grid covering the convex hull of the near-fulcrum
    observations (3D points within 200mm of the base centre, expanded
    40%) - targets the fulcrum directly, no pose needed."""
    pts = np.float32([o[0] for o in obs
                    if abs(o[1][0]) < 200 and abs(o[1][1]) < 200])
    if len(pts) < 3:
        return []
    hull = np.float64(cv2.convexHull(
        pts.reshape(-1, 1, 2)).reshape(-1, 2))
    c = hull.mean(axis=0)
    hull = c + (hull - c) * 1.4
    x0, y0 = hull.min(axis=0) - 10
    x1, y1 = hull.max(axis=0) + 10
    hull32 = np.float32(hull).reshape(-1, 1, 2)
    dots = []
    for v in np.linspace(y0, y1, rows):
        for u in np.linspace(x0, x1, cols):
            if cv2.pointPolygonTest(hull32,
                                    (float(u), float(v)), False) >= 0:
                dots.append([float(u), float(v)])
    return dots


def face_quads_from_obs(obs, polys, KW, w_rv, w_tv):
    """Per-face canvas quad via the EMPIRICAL canvas->camera homography.

    Each face is planar, so obs (canvas uv <-> webcam px) on a face
    define a homography; the quad that covers the face in webcam space
    pushed back through H^-1 is exactly the canvas region to paint.
    Bypasses projector intrinsics + camera distortion entirely."""
    quads = {}
    for name, poly in polys.items():
        sel = [(o[0], o[3]) for o in obs
               if o[2] == name and o[4] == "webcam"]
        if len(sel) < 4:
            continue
        H, _ = cv2.findHomography(
            np.float64([s[0] for s in sel]).reshape(-1, 1, 2),
            np.float64([s[1] for s in sel]).reshape(-1, 1, 2))
        if H is None:
            continue
        qcam = project(poly, KW, w_rv, w_tv).reshape(1, -1, 2)
        qc = cv2.perspectiveTransform(qcam, np.linalg.inv(H))
        quads[name] = qc.reshape(4, 2).tolist()
    return quads


def pnp_quads(polys, pos, rvec, KP):
    """Project each face polygon through the solved projector pinhole
    -> canvas quad.  pos/rvec = T_ful->proj (projector pos + rotation
    in FULCRUM coords).  TRACKS the fulcrum: feed a live-anchored
    pose and the quads follow the object."""
    R = cv2.Rodrigues(np.float64(rvec).ravel())[0]
    t = -R @ np.float64(pos).ravel()
    return {name: _proj_px(poly, KP, R, t).tolist()
            for name, poly in polys.items()}


def _proj_px(poly, K, R, t):
    P = (R @ np.float64(poly).T).T + t
    z = np.where(np.abs(P[:, 2]) < 1e-6, 1e-6, P[:, 2])
    return np.stack([K[0, 2] + K[0, 0] * P[:, 0] / z,
                     K[1, 2] + K[1, 0] * P[:, 1] / z], axis=1)


def live_face_quads(polys, pjo, KP, deltas):
    """Solved-pnp face quads + the fixed empirical correction captured
    at sweep time (per-face corner deltas in canvas px).  The pnp quad
    supplies true tracking; the delta absorbs pinhole/distortion bias
    so the paint lands where the dot survey actually saw the face."""
    if pjo is None:
        return {}
    base = pnp_quads(polys, pjo["pos"], pjo["rvec"], KP)
    out = {}
    for name, q in base.items():
        q = np.float64(q)
        if name in deltas:
            q = q + np.float64(deltas[name])
        out[name] = q.tolist()
    return out


def face_dots(polys, pos, rvec, KP):
    """2x2 bilinear points per face, projected through the seed
    projector pose -> canvas px for the targeting pass."""
    dots = []
    R = cv2.Rodrigues(np.float64(rvec).ravel())[0]
    t = -R @ np.float64(pos)
    for q in polys.values():
        for s in (0.3, 0.7):
            for tt in (0.3, 0.7):
                P = (1 - tt) * (q[0] + s * (q[3] - q[0])) + \
                    tt * (q[1] + s * (q[2] - q[1]))
                pc = R @ P + t
                if pc[2] <= 1:
                    continue
                u = KP[0, 0] * pc[0] / pc[2] + KP[0, 2]
                v = KP[1, 1] * pc[1] / pc[2] + KP[1, 2]
                if 30 <= u <= 1250 and 30 <= v <= 690:
                    dots.append([float(u), float(v)])
    return dots


def topup_dots(obs, polys, pj, KP):
    """Canvas dots aimed only at faces that still lack >=4 webcam
    observations: ring around the face's obs cluster when it has some,
    pose-quad interior points when it has none (or, as a last resort,
    midpoints between other faces' clusters)."""
    from collections import defaultdict
    pts = defaultdict(list)
    for o in obs:
        if o[2] != "table" and o[4] == "webcam":
            pts[o[2]].append(np.float64(o[0]))
    dots = []
    gap_done = False
    for name in polys:
        have = len(pts.get(name, []))
        if have >= 4:
            continue
        if have:
            m = np.mean(pts[name], axis=0)
            r = 25.0 if have > 1 else 45.0
            for a in np.linspace(0, 2 * np.pi, 7, endpoint=False):
                dots.append([float(m[0] + r * np.cos(a)),
                             float(m[1] + r * np.sin(a))])
        else:
            d = (face_dots({name: polys[name]}, pj["pos"],
                           pj["rvec"], KP) if pj else [])
            if d:
                dots += d
            elif not gap_done:
                means = [np.mean(v, axis=0)
                         for v in pts.values() if v]
                for i in range(len(means)):
                    for j in range(i + 1, len(means)):
                        mid = (means[i] + means[j]) / 2
                        dots.append([float(mid[0]), float(mid[1])])
                gap_done = True
    return dots


def solve_dots(obs, KP, seed=None):
    """obs: [(canvas_uv, P3, where, campx, cam)] -> projector pose dict
    or None.  Multi-start (seed / global SQPNP / 4 yaw guesses), LM
    refine, outlier rejection.  Face obs are weighted x2 - the fulcrum
    is the target; distant table hits are dropped (table flatness is
    an assumption that only holds near the base)."""
    obs = [o for o in obs if o[2] != "table" or
           abs(o[1][0]) < 200 and abs(o[1][1]) < 200]
    obs = obs + [o for o in obs if o[2] != "table"]
    if len(obs) < 6:
        return None
    obj = np.float64([o[1] for o in obs]).reshape(-1, 1, 3)
    img = np.float64([o[0] for o in obs]).reshape(-1, 1, 2)
    cands = []

    def add(rv, tv, label):
        rv, tv = cv2.solvePnPRefineLM(obj, img, KP, np.zeros(5), rv, tv)
        pred, _ = cv2.projectPoints(obj, rv, tv, KP, np.zeros(5))
        e = np.linalg.norm(pred.reshape(-1, 2) - img.reshape(-1, 2),
                           axis=1)
        cands.append((float(e.mean()), rv, tv, label))

    if seed is not None:
        srv = np.float64(seed["rvec"]).reshape(3, 1)
        stv = (-cv2.Rodrigues(srv)[0] @ np.float64(seed["pos"])
               ).reshape(3, 1)
        ok, rv, tv = cv2.solvePnP(obj, img, KP, np.zeros(5), srv, stv,
                                  True, cv2.SOLVEPNP_ITERATIVE)
        if ok:
            add(rv, tv, "seed")
    ok, rv, tv = cv2.solvePnP(obj, img, KP, np.zeros(5),
                              flags=cv2.SOLVEPNP_SQPNP)
    if ok:
        add(rv, tv, "sqpnp")
    if not cands or min(c[0] for c in cands) > 3.0:
        Rx = cv2.Rodrigues(np.float64([np.pi, 0, 0]))[0]
        for th in (0, np.pi / 2, np.pi, -np.pi / 2):
            R = cv2.Rodrigues(np.float64([0, 0, th]))[0] @ Rx
            t0 = -R @ np.float64([0, 0, 600.0])
            ok, rv, tv = cv2.solvePnP(obj, img, KP, np.zeros(5),
                                      cv2.Rodrigues(R)[0],
                                      t0.reshape(3, 1), True,
                                      cv2.SOLVEPNP_ITERATIVE)
            if ok:
                add(rv, tv, f"yaw{int(np.degrees(th))}")
    if not cands:
        return None
    em, rv, tv, label = min(cands, key=lambda c: c[0])
    pred, _ = cv2.projectPoints(obj, rv, tv, KP, np.zeros(5))
    e = np.linalg.norm(pred.reshape(-1, 2) - img.reshape(-1, 2), axis=1)
    keep = np.where(e < max(10.0, np.percentile(e, 75)))[0]
    if 6 <= len(keep) < len(obj):
        o2, i2 = obj[keep], img[keep]
        rv, tv = cv2.solvePnPRefineLM(o2, i2, KP, np.zeros(5), rv, tv)
        pred, _ = cv2.projectPoints(o2, rv, tv, KP, np.zeros(5))
        e = np.linalg.norm(pred.reshape(-1, 2) - i2.reshape(-1, 2),
                           axis=1)
    R = cv2.Rodrigues(rv)[0]
    return {"pos": (-R.T @ tv.ravel()), "rvec": rv.ravel(),
            "reproj": float(e.mean()), "n": len(obs), "label": label}


def sample_dot(cam, gray, sol, K, calib, pending, hist, polys):
    """Try to observe the pending projected dot in this camera's frame:
    diff vs a frame taken before the switch, centroid the blob,
    back-project the ray onto the fulcrum/table.  Mutates calib['obs']."""
    if calib is None or pending is None or cam in pending["seen"]:
        return
    if time.time() - pending["t0"] < DOT_SETTLE_S:
        return
    if sol is None:
        return
    before = None
    for t, g in hist[cam]:
        if t <= pending["t0"] + 0.05:
            before = g
    if before is None:
        return
    p = find_dot(before, gray)
    if p is None:
        return
    rv, tv = sol[0], sol[1]
    R = cv2.Rodrigues(rv)[0]
    o = -R.T @ tv.ravel()
    d = R.T @ (np.linalg.inv(K) @ np.float64([p[0], p[1], 1.0]))
    P, where = ray_surface(o, d, polys)
    if P is not None:
        u, v = calib["dots"][pending["idx"]]
        calib["obs"].setdefault(pending["idx"], []).append(
            ((u, v), P, where, p, cam))
        pending["seen"].add(cam)


# ---- colour assert -----------------------------------------------------
def classify_hue(h):
    for name, lo, hi in (("red", 165, 180), ("red", 0, 12),
                         ("yellow", 20, 38), ("green", 45, 85),
                         ("blue", 95, 130), ("magenta", 140, 165)):
        if lo <= h <= hi:
            return name
    return "?"


def assert_faces(bgr, K, rv, tv, polys, dist=None):
    """Per face: dominant saturated hue inside its projected quad vs
    the assigned colour -> (name, got, want, ok)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    res = {}
    for name, poly in polys.items():
        px = project(poly, K, rv, tv, dist)
        mask[:] = 0
        cv2.fillPoly(mask, [np.int32(px)], 255)
        px_hsv = hsv[mask > 0]
        sel = px_hsv[(px_hsv[:, 1] > 70) & (px_hsv[:, 2] > 50)]
        want = FACE_COLORS.get(name)
        want_name = {(0, 0, 255): "red", (0, 255, 0): "green",
                     (255, 0, 0): "blue", (0, 255, 255): "yellow",
                     (255, 0, 255): "magenta"}.get(want, str(want))
        if want is None or len(sel) < 30:
            res[name] = (None, want_name, None)
            continue
        got = classify_hue(np.median(sel[:, 0]))
        res[name] = (got, want_name, got == want_name)
    return res


# ---- io ----------------------------------------------------------------
def save_jpg(img, name):
    try:
        os.makedirs(PREV, exist_ok=True)
        dst = os.path.join(PREV, name + ".jpg")
        tmp = os.path.join(PREV, name + ".tmp.jpg")
        cv2.imwrite(tmp, img, JPEG)
        for _ in range(5):
            try:
                os.replace(tmp, dst)
                return
            except (PermissionError, OSError):
                time.sleep(0.02)
        cv2.imwrite(dst, img, JPEG)
    except Exception:
        pass


def write_poses(data):
    try:
        tmp = POSES + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(data, fh)
        for _ in range(5):
            try:
                os.replace(tmp, POSES)
                return
            except (PermissionError, OSError):
                time.sleep(0.02)
    except Exception:
        pass


def control_server():
    class H(BaseHTTPRequestHandler):
        def _reply(self):
            body = json.dumps({"paused": PAUSED.is_set()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/pause"):
                PAUSED.set()
                print("paused - webcam released")
            elif self.path.startswith("/resume"):
                PAUSED.clear()
                print("resumed")
            self._reply()

        def log_message(self, *a):
            pass

    ThreadingHTTPServer(("127.0.0.1", CTRL_PORT), H).serve_forever()


def annotate(vis, found, sol, K, world, asserts, label, polys=None,
             dist=None):
    D = np.zeros(5) if dist is None else dist
    if sol is not None and polys is not None:
        for name, q in polys.items():
            px = project(q, K, sol[0], sol[1], D)
            cv2.polylines(vis, [np.int32(px)], True,
                          FACE_COLORS.get(name, (255, 255, 255)), 1)
    if found:
        cv2.aruco.drawDetectedMarkers(
            vis, np.array([c.reshape(1, 4, 2) for c in found.values()]),
            np.array([[i] for i in found]))
    lines = [f"{label}  markers {sorted(found)}"]
    if sol:
        rv, tv, em, ex = sol
        cv2.drawFrameAxes(vis, K, D, rv, tv, 45)
        pos = -cv2.Rodrigues(rv)[0].T @ tv.ravel()
        lines.append(f"pnp {em:.1f}/{ex:.1f}px  cam@ "
                     f"{[round(v) for v in pos]}")
    if asserts:
        ok = sum(1 for a in asserts.values() if a[2])
        lines.append(f"assert {ok}/{len(asserts)} faces correct")
        for nm, (got, want, okk) in sorted(asserts.items()):
            lines.append(f"  {nm}: {got or '-'} want {want} "
                         f"{'OK' if okk else 'X'}")
    for j, t in enumerate(lines[:9]):
        cv2.putText(vis, t, (16, 34 + 30 * j), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


# ---- main ---------------------------------------------------------------
def open_webcam():
    # DSHOW only: MSMF wedges at the driver level after a hard kill -
    # open() itself hangs, so there is no safe fallback
    cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WH[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WH[1])
    return cap


class WebcamGrabber(threading.Thread):
    """cap.read() on MSMF can hang forever when the camera stalls -
    keep grabbing in a thread; the main loop uses the newest frame by
    timestamp and resets the capture if it goes silent."""

    def __init__(self):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture()
        self.frame, self.t = None, 0.0
        self.reset = threading.Event()

    def run(self):
        while True:
            if PAUSED.is_set():
                if self.cap.isOpened():
                    self.cap.release()
                time.sleep(0.25)
                continue
            if self.reset.is_set() or not self.cap.isOpened():
                self.reset.clear()
                if self.cap.isOpened():
                    self.cap.release()
                self.cap = open_webcam()
                if not self.cap.isOpened():
                    time.sleep(0.6)
                    continue
            ok, fr = self.cap.read()
            if ok and fr is not None:
                self.frame, self.t = fr, time.time()


def main():
    spec = json.load(open(SPEC))
    world = {int(i): np.float64(m["corners_TL_TR_BR_BL"])
             for i, m in spec["markers"].items()}
    polys = face_polys()
    # projector intrinsics + frozen device table-world poses (for
    # expressing live fulcrum-frame poses back in the table frame)
    frozen = json.load(open(os.path.join(PIPE_FF,
                                         "aruco_solve_frozen.json")))
    pj_K = {"f": frozen["projector"]["f_px"],
            "cx": frozen["projector"]["cx"],
            "cy": frozen["projector"]["cy"]}
    frozen_dev = {k: rt(frozen[k]["rvec"], frozen[k]["tvec"])
                  for k in ("webcam", "kinect") if k in frozen}
    # projected board corners in table world (z=0): lets each camera
    # re-solve its OWN table pose live - convention cancels in the
    # composition and a moved camera re-anchors itself automatically
    board3 = {int(i): np.hstack([np.float64(c), np.zeros((4, 1))])
              for i, c in frozen["world_corners_mm"].items()}
    tab_dev = {}          # camera -> T_table->cam, live-solved
    # projected canvas footprint in table mm: the fulcrum can only sit
    # where light lands, so an estimate outside this poly is a bad solve
    Hf = np.float64(frozen["H_px_to_mm"])
    FIELD = cv2.perspectiveTransform(
        np.float64([[0, 0], [1280, 0], [1280, 720], [0, 720]])
        .reshape(-1, 1, 2), Hf).reshape(-1, 2).astype(np.float32)
    FIELD_MARGIN = 60.0   # mm of slack around the lit footprint
    KP = np.float64([[pj_K["f"], 0, pj_K["cx"]],
                     [0, pj_K["f"], pj_K["cy"]], [0, 0, 1]])
    f = json.load(open(CHEB))["cam"]["f_px"]
    KW = np.float64([[f, 0, WH[0] / 2], [0, f, WH[1] / 2], [0, 0, 1]])

    threading.Thread(target=control_server, daemon=True).start()
    cam = WebcamGrabber()
    cam.start()
    kin, KK_raw, DK, kflip = None, None, np.zeros(5), "flipV"
    try:
        from kincap import Kin
        kin = Kin("color", "depth")
        fr = None
        for _ in range(15):
            fr = kin.grab("color", timeout=1)
            kin.grab("depth", timeout=1)
            if fr is not None:
                break
        if fr is not None:
            KK_raw, DK = kinect_intrinsics(kin, fr)
            print("kinect connected",
                  f"f=({KK_raw[0,0]:.0f},{KK_raw[1,1]:.0f})"
                  f" dist={'calibrated' if np.any(DK) else 'none'}")
    except Exception as e:
        print("kinect unavailable:", e)
        kin = None

    state = {n: {"rv": None, "tv": None, "last": 0.0, "miss": 99}
             for n in ("webcam", "kinect")}
    hist = {"webcam": deque(maxlen=4), "kinect": deque(maxlen=4)}
    pj, pj_t = None, 0.0
    pj_Tfp = None        # T_ful0->proj, frozen at solve time
    pj_ref = None        # T_ful0->table at that same moment
    pj_live = None       # pj re-anchored to the CURRENT fulcrum pose
    board_until, last_flash = 0.0, 0.0
    face_quads = {}
    fq_delta = {}        # per-face canvas-px correction (emp - pnp)
    emp_quads = {}       # static homography quads - fallback w/o pj
    all_obs = []
    try:   # reuse the last sweep's observations - quads work at once
        prev = json.load(open(os.path.join(PIPE_FF,
                                           "fulcrum_obs.json")))
        all_obs = [(np.float64(o[0]), np.float64(o[1]), o[2],
                    np.float64(o[3]), o[4]) for o in prev.get("obs", [])]
        fq_delta = prev.get("fq_delta", {})
        emp_quads = prev.get("emp_quads", {})
        pj0 = prev.get("pj")
        if all_obs:
            print("loaded", len(all_obs), "obs from previous sweep")
        if pj0 is not None:
            pj = {**pj_K, "pos": pj0["pos"], "rvec": pj0["rvec"]}
            pj_t = time.time()
            R_p = cv2.Rodrigues(np.float64(pj["rvec"]))[0]
            pj_Tfp = np.eye(4)
            pj_Tfp[:3, :3] = R_p
            pj_Tfp[:3, 3] = -R_p @ np.float64(pj["pos"])
            if prev.get("pj_ref") is not None:
                pj_ref = np.float64(prev["pj_ref"])
            print("loaded pj from previous sweep")
        elif all_obs:
            res0 = solve_dots(all_obs, KP)
            if res0 is not None and res0["reproj"] <= 15.0:
                pj = {**pj_K, "pos": res0["pos"],
                      "rvec": res0["rvec"]}
                pj_t = time.time()
                R_p = cv2.Rodrigues(np.float64(pj["rvec"]))[0]
                pj_Tfp = np.eye(4)
                pj_Tfp[:3, :3] = R_p
                pj_Tfp[:3, 3] = -R_p @ np.float64(pj["pos"])
                if prev.get("pj_ref") is not None:
                    pj_ref = np.float64(prev["pj_ref"])
                print("pj from persisted obs: "
                      f"{res0['reproj']:.1f}px")
    except Exception as e:
        print("obs load failed:", e)
    calib, pending, seq = None, None, 1
    bad = 0
    ful_pos = None
    ful_seen_t = 0.0
    last_Ttf = None      # last fused T_ful->table (pj_ref bookkeeping)
    calib_backoff = 8.0
    next_calib_at = time.time() + 4.0
    print("tracking ->", POSES)
    n = 0
    while True:
        n += 1
        if PAUSED.is_set():
            time.sleep(0.25)
            continue
        now = time.time()
        if cam.t and now - cam.t > 6.0:
            print("webcam stalled - resetting capture")
            try:
                cam.cap.release()   # may unblock a hung read()
            except Exception:
                pass
            cam.reset.set()
            cam.t = now  # don't spam resets while reopening
        out = {"t": now, "paused": PAUSED.is_set(),
               "faces": {k: list(v)
                                   for k, v in FACE_COLORS.items()},
               "quad_margin": QUAD_MARGIN,
               "devices": {}, "projector": None, "asserts": {}}
        tf_est = {}          # camera -> (T_table->ful, reproj)
        live_asserts = calib is None and \
            (pj is not None or face_quads)

        # ---------- webcam ----------
        frame = (cam.frame if cam.frame is not None
                 and now - cam.t < 0.8 else None)
        if frame is not None:
            st = state["webcam"]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found = detect(gray)
            bsol = solve_table(board3, found, KW)
            if bsol:
                tab_dev["webcam"] = rt(bsol[1], bsol[2])
            guess = (st["rv"], st["tv"]) if st["rv"] is not None else None
            sol, ful = solve_fulcrum(world, found, KW, guess)
            asserts = {}
            if sol:
                rv, tv, em, ex = sol
                pos = -cv2.Rodrigues(rv)[0].T @ tv.ravel()
                # reject degenerate solves (planar branch flip, bad
                # seed): a real camera sits 50-1500mm from the fulcrum
                if em > 15.0 or np.abs(pos).max() > 1500 \
                        or pos[2] < 50:
                    st["miss"] += 1
                    sol = None
                else:
                    st["rv"], st["tv"], st["last"], st["miss"] = \
                        rv, tv, now, 0
                    weak = len(ful) < 2 or em > 6.0
                    if live_asserts:
                        asserts = assert_faces(
                            frame, KW, rv, tv, polys)
                        out["asserts"]["webcam"] = {
                            k: [g, w, o] for k, (g, w, o)
                            in asserts.items()}
                    out["devices"]["webcam"] = {
                        "pos_mm": pos.tolist(),
                        "rvec": rv.ravel().tolist(),
                        "tvec": tv.ravel().tolist(), "reproj": em,
                        "markers": sorted(ful), "weak": weak,
                        "status": "weak" if weak else "tracked"}
                    anchor = tab_dev.get("webcam",
                                         frozen_dev.get("webcam"))
                    if anchor is not None:
                        # T_ful->table = inv(T_table->cam) @ T_ful->cam
                        tf_est["webcam"] = (
                            np.linalg.inv(anchor) @ rt(rv, tv),
                            em, weak)
            else:
                st["miss"] += 1
                st["rv"] = st["tv"] = None if st["miss"] > 5 else st["rv"]
            sample_dot("webcam", gray, sol, KW, calib, pending,
                       hist, polys)
            hist["webcam"].append((now, gray))
            vis = annotate(frame.copy(), found, sol, KW, world,
                           asserts, "webcam", polys)
            if calib is not None and pending is not None:
                u, v = calib["dots"][pending["idx"]]
                cv2.putText(vis, f"calib dot {pending['idx']+1}/"
                            f"{len(calib['dots'])} r{calib['round']}",
                            (16, vis.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 200, 255), 2, cv2.LINE_AA)
            save_jpg(vis, "webcam")

        # ---------- kinect ----------
        if kin is not None:
            st = state["kinect"]
            raw = None
            try:
                raw = kin.grab("color", timeout=0.5)
            except Exception:
                pass
            if raw is not None:
                kn = time.time()
                gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                KK = k_for_flip(KK_raw, kflip)
                g = gray if FLIPS[kflip] is None else cv2.flip(
                    gray, FLIPS[kflip])
                found = detect(g)
                bsol = solve_table(board3, found, KK, DK)
                if bsol:
                    tab_dev["kinect"] = rt(bsol[1], bsol[2])
                guess = ((st["rv"], st["tv"]) if st["rv"] is not None
                         else None)
                sol, ful = solve_fulcrum(world, found, KK, guess, DK)
                if sol is None and st["miss"] > 5:
                    best = None
                    for name, code in FLIPS.items():
                        g2 = (gray if code is None
                              else cv2.flip(gray, code))
                        f2 = detect(g2)
                        s2, _ = solve_fulcrum(world, f2,
                                              k_for_flip(KK_raw, name),
                                              dist=DK)
                        if s2 and (best is None or s2[2] < best[0][2]):
                            best = (s2, name, g2, f2)
                    if best:
                        sol, kflip, g, found = best
                        print("kinect flip ->", kflip)
                        ful = {i: c for i, c in found.items()
                               if i in world}
                vis = (raw if FLIPS[kflip] is None
                       else cv2.flip(raw, FLIPS[kflip]))
                asserts = {}
                if sol:
                    rv, tv, em, ex = sol
                    pos = -cv2.Rodrigues(rv)[0].T @ tv.ravel()
                    if em > 15.0 or np.abs(pos).max() > 1500 \
                            or pos[2] < 50:
                        st["miss"] += 1
                        sol = None
                    else:
                        st["rv"], st["tv"], st["last"], st["miss"] = \
                            rv, tv, kn, 0
                        weak = len(ful) < 2 or em > 6.0
                        if live_asserts:
                            asserts = assert_faces(
                                vis, KK, rv, tv, polys, DK)
                            out["asserts"]["kinect"] = {
                                k: [g_, w, o] for k, (g_, w, o)
                                in asserts.items()}
                        out["devices"]["kinect"] = {
                            "pos_mm": pos.tolist(),
                            "rvec": rv.ravel().tolist(),
                            "tvec": tv.ravel().tolist(), "reproj": em,
                            "markers": sorted(ful), "weak": weak,
                            "status": "weak" if weak else "tracked",
                            "flip": kflip}
                        anchor = tab_dev.get(
                            "kinect", frozen_dev.get("kinect"))
                        if anchor is not None:
                            tf_est["kinect"] = (
                                np.linalg.inv(anchor) @ rt(rv, tv),
                                em, weak)
                else:
                    st["miss"] += 1
                # NOTE: kinect frames arrive ~0.3-0.5s late (buffered
                # stream) - dot sampling is webcam-only, kinect's
                # observations would pair stale blobs with fresh dots
                hist["kinect"].append((kn, g))
                vis = annotate(vis.copy(), found, sol, KK, world,
                               asserts, f"kinect[{kflip}]", polys,
                               dist=DK)
                save_jpg(vis, "kinect")

        # ---------- projector calib state machine ----------
        pstate = read_proj_state()
        if calib is not None and pstate:
            if pstate.get("mode") == "dots" and \
                    pstate.get("seq") == calib["seq"]:
                idx = pstate.get("idx")
                if pending is None or pending["idx"] != idx:
                    pending = {"idx": idx,
                               "t0": float(pstate.get("t", now)),
                               "seen": set()}
            elif pstate.get("mode") == "dots_done" and \
                    pstate.get("seq") == calib["seq"]:
                obs = [o for l in calib["obs"].values() for o in l]
                obs += calib.get("carry", [])
                all_obs = obs
                try:
                    with open(os.path.join(PIPE_FF,
                                           "fulcrum_obs.json"),
                              "w") as fh:
                        json.dump({"round": calib["round"], "obs": [
                            [list(uv), P.tolist(), w, list(cp), cm]
                            for uv, P, w, cp, cm in obs]}, fh)
                except Exception:
                    pass
                res = solve_dots(obs, KP, seed=pj)
                cnt = {n: sum(1 for o in obs
                              if o[2] == n and o[4] == "webcam")
                       for n in polys}
                deficient = [n for n, c in cnt.items() if c < 4]
                print(f"round {calib['round']}: {len(obs)} obs "
                      f"(faces {cnt}) ->",
                      "FAIL" if res is None else
                      f"{res['reproj']:.2f}px via {res['label']}")
                pending = None
                if res is not None and res["reproj"] <= 12.0:
                    pj = {**pj_K, "pos": res["pos"],
                          "rvec": res["rvec"]}
                    pj_t = now
                    # freeze T_ful0->proj; the live table-chain pose
                    # re-anchors it to the CURRENT fulcrum every frame
                    R_p = cv2.Rodrigues(
                        np.float64(pj["rvec"]))[0]
                    pj_Tfp = np.eye(4)
                    pj_Tfp[:3, :3] = R_p
                    pj_Tfp[:3, 3] = -R_p @ np.float64(pj["pos"])
                    pj_ref = None
                # keep topping up deficient faces (max 4 rounds)
                if deficient and calib["round"] < 4:
                    d2 = topup_dots(obs, polys, pj, KP)
                    if d2:
                        calib = {"seq": seq, "round": calib["round"] + 1,
                                 "dots": d2, "obs": {}, "carry": obs}
                        seq += 1
                        print("topup:", len(d2), "dots for",
                              deficient)
                    else:
                        calib = None
                else:
                    rnd, calib = calib["round"], None
                    wrv, wtv = state["webcam"]["rv"], \
                        state["webcam"]["tv"]
                    if wrv is not None:
                        emp = face_quads_from_obs(obs, polys, KW,
                                                  wrv, wtv)
                        if emp:
                            emp_quads = emp
                            calib_backoff = 8.0
                            print("face_quads <- homographies:",
                                  sorted(emp))
                            # empirical correction vs the pinhole
                            # model - captured at the sweep pose, then
                            # re-applied to the live pnp quad per frame
                            if pj is not None:
                                base = pnp_quads(polys, pj["pos"],
                                                 pj["rvec"], KP)
                                fq_delta = {
                                    n: (np.float64(emp[n]) -
                                        np.float64(base[n])).tolist()
                                    for n in emp if n in base}
                    if pj is None and not emp_quads:
                        next_calib_at = now + calib_backoff
                        calib_backoff = min(calib_backoff * 2, 60.0)
                    if pj is not None and last_Ttf is not None:
                        # the fulcrum's table pose AT THE SWEEP - the
                        # reference body pose the pj solve belongs to
                        pj_ref = last_Ttf.copy()
                    try:   # persist everything for the next restart
                        with open(os.path.join(PIPE_FF,
                                               "fulcrum_obs.json"),
                                  "w") as fh:
                            json.dump({"round": rnd,
                                       "obs": [
                                           [list(uv), P.tolist(), w,
                                            list(cp), cm]
                                           for uv, P, w, cp, cm
                                           in obs],
                                       "pj": ({"pos": pj["pos"],
                                               "rvec": pj["rvec"]}
                                              if pj else None),
                                       "pj_ref": (pj_ref.tolist()
                                                  if pj_ref
                                                  is not None
                                                  else None),
                                       "fq_delta": fq_delta,
                                       "emp_quads": emp_quads}, fh)
                    except Exception:
                        pass
        if pending is not None and \
                now - pending["t0"] > DOT_PENDING_S:
            pending = None
        # auto-sweep only when there is nothing to paint with yet;
        # persistent assert failure below re-triggers it when needed
        if calib is None and not face_quads and \
                now >= next_calib_at and \
                out["devices"].get("webcam", {}).get("status") in \
                ("tracked", "weak"):
            calib = {"seq": seq, "round": 1,
                     "dots": grid_dots(7, 5), "obs": {}, "carry": []}
            seq += 1
            pending = None
            next_calib_at = float("inf")
            print("calib round 1: sweeping", len(calib["dots"]), "dots")

        # ---------- auto recalib on persistent assert failure ----------
        if calib is None and (pj is not None or face_quads) and \
                pstate and pstate.get("mode") == "colors":
            vis = [a for c in out["asserts"].values()
                   for a in c.values() if a[2] is not None]
            bad = bad + 1 if vis and not any(a[2] for a in vis) else 0
            if bad >= RECALIB_FRAMES:
                bad = 0
                pending = None
                calib = {"seq": seq, "round": 1,
                         "dots": grid_dots(7, 5), "obs": {},
                         "carry": []}
                seq += 1
                print("asserts failed persistently - recalibrating")

        # periodic board flash (~2s/45s): lets each camera re-solve its
        # own table pose so the chain stays honest when devices shift
        if calib is None and now - last_flash > 45.0:
            board_until, last_flash = now + 2.0, now
        pjo = pj_live if pj_live is not None else pj
        if pjo is not None:
            out["projector"] = {
                "pos_mm": np.asarray(pjo["pos"]).tolist(),
                "rvec": np.asarray(pjo["rvec"]).tolist(),
                "f": float(pj["f"]), "cx": float(pj["cx"]),
                "cy": float(pj["cy"]),
                "age_s": round(now - pj_t, 1)}
        # ---------- fulcrum pose in the frozen table frame ----------
        src = None
        if tf_est:
            # sanity gate: the fulcrum can only sit where light lands -
            # inside the projected field.  score = (off-field, weak,
            # not-webcam (latency), reproj); lowest wins.
            def _score(k):
                T_, e_, w_ = tf_est[k]
                p_ = T_[:3, 3]
                off = cv2.pointPolygonTest(
                    FIELD, (float(p_[0]), float(p_[1])), True) \
                    < -FIELD_MARGIN
                return (bool(off), bool(w_), k != "webcam", e_)
            src = min(tf_est, key=_score)
        if src is not None:
            T_tf, _, tf_weak = tf_est[src]   # T_ful->table
            last_Ttf = T_tf.copy()
            ful_pos = T_tf[:3, 3].copy()
            ful_seen_t = now
            off_field = bool(cv2.pointPolygonTest(
                FIELD, (float(ful_pos[0]), float(ful_pos[1])),
                True) < -FIELD_MARGIN)
            out["fulcrum_table"] = {
                "pos_mm": T_tf[:3, 3].tolist(),
                "rvec": cv2.Rodrigues(T_tf[:3, :3])[0].ravel().tolist(),
                "R": T_tf[:3, :3].tolist(), "weak": bool(tf_weak),
                "off_field": off_field,
                "src": [src], "also": sorted(tf_est)}
            for name, dev in out["devices"].items():
                # T_table->cam = T_ful->cam @ inv(T_ful->table)
                T_td = rt(dev["rvec"], dev["tvec"]) @ \
                    np.linalg.inv(T_tf)
                tp = -T_td[:3, :3].T @ T_td[:3, 3]
                dev["table_pos_mm"] = tp.tolist()
                # flag devices that sit far from their frozen spot -
                # either the device really moved or the solve is off
                if name in frozen_dev:
                    fr = frozen_dev[name]
                    fpos = -fr[:3, :3].T @ fr[:3, 3]
                    dev["moved"] = bool(
                        np.linalg.norm(tp - fpos) > 300)
            # re-anchor the solved projector pose to the LIVE fulcrum:
            # T_fn->proj = T_f0->proj @ inv(T_f0->table) @ T_fn->table
            if pj is not None and pj_Tfp is not None:
                if pj_ref is None:
                    pj_ref = T_tf.copy()
                T_np = pj_Tfp @ np.linalg.inv(pj_ref) @ T_tf
                Rn, tn = T_np[:3, :3], T_np[:3, 3]
                pj_live = {"pos": (-Rn.T @ tn).tolist(),
                           "rvec": cv2.Rodrigues(Rn)[0]
                           .ravel().tolist()}
                if out["projector"] is not None:
                    out["projector"]["pos_mm"] = pj_live["pos"]
                    out["projector"]["rvec"] = pj_live["rvec"]
                    T_tp = pj_Tfp @ np.linalg.inv(pj_ref)
                    out["projector"]["table_pos_mm"] = (
                        -T_tp[:3, :3].T @ T_tp[:3, 3]).tolist()
            # paint quads: solved-pnp through the live-anchored
            # projector pose + fixed empirical deltas; falls back to
            # the static homography quads when there is no pj
            pjo2 = pj_live if pj_live is not None else pj
            fq_new = live_face_quads(polys, pjo2, KP, fq_delta)
            if fq_new:
                face_quads = fq_new
        if not face_quads and emp_quads:
            face_quads = emp_quads
        if face_quads:
            out["face_quads"] = face_quads
        out["proj_cmd"] = (
            {"mode": "dots", "seq": calib["seq"], "dots": calib["dots"]}
            if calib is not None else
            {"mode": "colors" if pjo is not None or face_quads
             else "board" if now - ful_seen_t > 10.0
             or now < board_until
             else "dark"})
        write_poses(out)
        if n % 50 == 0:
            dev = {k: (v.get("status"), len(v.get("markers", [])))
                   for k, v in out["devices"].items()}
            print(f"[{n}] {dev} pj={'y' if pj else 'n'} "
                  f"calib={calib['round'] if calib else '-'} "
                  f"asserts={ {k: sum(1 for a in v.values() if a[2]) for k, v in out['asserts'].items()} }")


if __name__ == "__main__":
    main()
