import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools' / 'ff'))
from fulcrum_track import detect, face_polys, solve_fulcrum

OUT = ROOT / 'smoke'
TAG = time.strftime('ff_%Y%m%d_%H%M%S')


def td(code, mode='--exec'):
    p = subprocess.run(['node', str(ROOT / 'tools/td/td.mjs'), mode, code],
                       cwd=ROOT, capture_output=True, text=True, timeout=25)
    if p.returncode:
        raise RuntimeError(p.stderr or p.stdout)
    return p.stdout.strip()


def capture(name, label):
    path = OUT / f'{TAG}_{label}_{name}.png'
    td(f"op('/project1/{name}').save({str(path.as_posix())!r})")
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f'Capture failed: {path}')
    return image


def hit(uv, K, rv, tv):
    R = cv2.Rodrigues(rv)[0]
    origin = -R.T @ tv.reshape(3)
    direction = R.T @ np.linalg.solve(K, [*uv, 1.0])
    hits = []
    for name, q in face_polys().items():
        normal = np.cross(q[1] - q[0], q[2] - q[0])
        denom = direction @ normal
        if abs(denom) < 1e-9:
            continue
        distance = ((q[0] - origin) @ normal) / denom
        point = origin + distance * direction
        sides = np.array([np.cross(q[(i + 1) % 4] - q[i], point - q[i]) @ normal
                          for i in range(4)])
        if distance > 0 and (np.all(sides >= -1e-5) or np.all(sides <= 1e-5)):
            hits.append((distance, point, name))
    if hits:
        _, point, name = min(hits, key=lambda x: x[0])
        return point, name
    distance = -origin[2] / direction[2]
    if distance <= 0:
        raise RuntimeError('Ray misses table')
    return origin + distance * direction, 'table'


def face_dots(K, rv, tv, KP, pr, pt):
    observations = []
    survey = []
    black = OUT / f'{TAG}_dots_black.png'
    cv2.imwrite(str(black), np.zeros((720,1280), np.uint8))
    td(f"op('/project1/ff_calibration_pattern').par.file={black.as_posix()!r}")
    time.sleep(0.5)
    baseline = cv2.cvtColor(capture('camLogi', 'dots_black'), cv2.COLOR_BGR2GRAY)
    for face_name, q in face_polys().items():
        for v in (0.22, 0.5, 0.78):
            for u in (0.22, 0.5, 0.78):
                point = (1-v)*((1-u)*q[0]+u*q[3]) + v*((1-u)*q[1]+u*q[2])
                puv = cv2.projectPoints(point.reshape(1,3), pr, pt, KP, np.zeros(5))[0].reshape(2)
                cuv = cv2.projectPoints(point.reshape(1,3), rv, tv, K, np.zeros(5))[0].reshape(2)
                if not (12 < puv[0] < 1268 and 12 < puv[1] < 708
                        and 12 < cuv[0] < 628 and 12 < cuv[1] < 468):
                    continue
                camera_hit, camera_face = hit(cuv, K, rv, tv)
                projector_hit, projector_face = hit(puv, KP, pr, pt)
                if (camera_face != face_name or projector_face != face_name
                        or np.linalg.norm(camera_hit-point) > 2
                        or np.linalg.norm(projector_hit-point) > 2):
                    continue
                index = len(survey)
                command = np.round(puv).astype(int)
                pattern = np.zeros((720,1280), np.uint8)
                cv2.circle(pattern, tuple(command), 7, 255, -1)
                path = OUT / f'{TAG}_dot_{index:02d}.png'
                cv2.imwrite(str(path), pattern)
                td(f"op('/project1/ff_calibration_pattern').par.file={path.as_posix()!r}")
                time.sleep(0.35)
                frame = capture('camLogi', f'dot_{index:02d}')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                delta = cv2.subtract(gray, baseline)
                yy, xx = np.indices(delta.shape)
                roi = (xx-cuv[0])**2+(yy-cuv[1])**2 < 18**2
                mask = ((delta > 28) & roi).astype(np.uint8)
                count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
                candidates = [j for j in range(1,count) if 3 <= stats[j,cv2.CC_STAT_AREA] <= 180]
                entry = dict(face=face_name, projector_px=command.tolist(), expected_cam_px=cuv.tolist())
                if len(candidates) == 1:
                    j = candidates[0]
                    weights = delta*(labels == j)
                    centroid = np.array([(weights*xx).sum(),(weights*yy).sum()])/weights.sum()
                    measured, measured_face = hit(centroid, K, rv, tv)
                    entry.update(camera_px=centroid.tolist(), measured_face=measured_face,
                                 area=int(stats[j,cv2.CC_STAT_AREA]))
                    if measured_face == face_name and np.linalg.norm(measured-point) < 18:
                        observations.append((measured, command, 100+index, face_name))
                        entry['accepted'] = True
                survey.append(entry)
                print('dot', index, face_name, 'accepted' if entry.get('accepted') else 'rejected', flush=True)
    td(f"op('/project1/ff_calibration_pattern').par.file={black.as_posix()!r}")
    time.sleep(0.4)
    final = capture('camLogi', 'dots_end')
    return observations, survey, final


def main():
    spec = json.loads((ROOT / 'tools/pipe/ff/fulcrum_spec.json').read_text())
    world = {int(i): np.float64(m['corners_TL_TR_BR_BL'])
             for i, m in spec['markers'].items()}
    f = json.loads((ROOT / 'tools/pipe/ff/cheb_view.json').read_text())['cam']['f_px']
    K = np.float64([[f, 0, 320], [0, f, 240], [0, 0, 1]])
    previous = int(float(td("op('/project1/final_switch').par.index.eval()", '--eval')))
    result = {'timestamp': TAG, 'units': 'mm', 'frame': spec['frame'],
              'intrinsics_source': 'cheb_view webcam estimate; frozen projector prior',
              'alignment_nodes_modified': False}
    try:
        td("op('/project1/final_switch').par.index=0")
        time.sleep(0.7)
        clean = capture('camLogi', 'clean')
        capture('kinectC', 'clean')
        if clean.shape[:2] != (480, 640):
            raise RuntimeError('Webcam resolution differs from intrinsic estimate')
        solution, found = solve_fulcrum(world, detect(cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)), K)
        if solution is None or len(found) < 3:
            raise RuntimeError('Need at least three physical ff markers')
        rv, tv, mean, maximum = solution
        if mean > 3 or maximum > 8:
            raise RuntimeError(f'ff pose residual too high: {mean:.2f}/{maximum:.2f}px')
        result['webcam'] = dict(K=K.tolist(), rvec=rv.ravel().tolist(),
                                tvec=tv.ravel().tolist(), markers=sorted(found),
                                mean_px=mean, max_px=maximum)
        print('ff webcam:', sorted(found), 'mean/max px:', mean, maximum, flush=True)
        pattern = np.full((720, 1280), 255, np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        corners = {}
        for row in range(4):
            for col in range(6):
                i = 20 + row * 6 + col
                x, y, size = 45 + col * 205, 30 + row * 175, 130
                pattern[y:y+size, x:x+size] = cv2.aruco.generateImageMarker(dictionary, i, size)
                corners[i] = np.float64([[x,y],[x+size-1,y],[x+size-1,y+size-1],[x,y+size-1]])
        pattern_path = OUT / f'{TAG}_pattern.png'
        cv2.imwrite(str(pattern_path), pattern)
        td("r=op('/project1'); m=r.op('ff_calibration_pattern') or r.create(moviefileinTOP,'ff_calibration_pattern'); "
           f"m.par.file={pattern_path.as_posix()!r}; "
           "r.op('final_switch').inputConnectors[4].connect(m); r.op('final_switch').par.index=4")
        time.sleep(1.2)
        projected = capture('camLogi', 'pattern')
        capture('kinectC', 'pattern')
        detections = detect(cv2.cvtColor(projected, cv2.COLOR_BGR2GRAY))
        obj, img, groups, surfaces = [], [], [], []
        for i, camera_corners in detections.items():
            if i not in corners:
                continue
            for uv, projector_uv in zip(camera_corners, corners[i]):
                point, surface = hit(uv, K, rv, tv)
                obj.append(point)
                img.append(projector_uv)
                groups.append(i)
                surfaces.append(surface)
        obj, img = np.float64(obj), np.float64(img)
        print('projected markers:', sorted(set(groups)), 'points:', len(obj), flush=True)
        if len(set(groups)) < 6:
            raise RuntimeError('Insufficient projected marker coverage')
        prior = json.loads((ROOT / 'tools/pipe/ff/aruco_solve_frozen.json').read_text())['projector']
        KP = np.float64([[prior['f_px'],0,prior['cx']],[0,prior['f_px'],prior['cy']],[0,0,1]])
        train = np.array([i % 3 != 0 for i in groups])
        if train.sum() < 8 or (~train).sum() < 4:
            raise RuntimeError('Insufficient marker groups for held-out validation')
        ok, pr, pt = cv2.solvePnP(obj[train], img[train], KP, np.zeros(5), flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            raise RuntimeError('Projector solvePnP failed')
        pr, pt = cv2.solvePnPRefineLM(obj[train], img[train], KP, np.zeros(5), pr, pt)
        pred = cv2.projectPoints(obj, pr, pt, KP, np.zeros(5))[0].reshape(-1,2)
        errors = np.linalg.norm(pred - img, axis=1)
        initial = np.r_[pr.ravel(), pt.ravel(), KP[0,0], KP[1,2]]
        def residual(parameters):
            k = KP.copy()
            k[0,0] = k[1,1] = parameters[6]
            k[1,2] = parameters[7]
            prediction = cv2.projectPoints(obj[train], parameters[:3], parameters[3:6], k,
                                           np.zeros(5))[0].reshape(-1,2)
            return np.r_[(prediction-img[train]).ravel(),
                         (parameters[6]-initial[6])/500, (parameters[7]-initial[7])/300]
        refined = least_squares(residual, initial, loss='soft_l1', f_scale=3,
                               bounds=([-np.inf]*6+[500,-1500], [np.inf]*6+[4000,720]),
                               max_nfev=300)
        candidate_K = KP.copy()
        candidate_K[0,0] = candidate_K[1,1] = refined.x[6]
        candidate_K[1,2] = refined.x[7]
        candidate_pred = cv2.projectPoints(obj, refined.x[:3], refined.x[3:6], candidate_K,
                                          np.zeros(5))[0].reshape(-1,2)
        candidate_errors = np.linalg.norm(candidate_pred-img, axis=1)
        if candidate_errors[~train].mean() < errors[~train].mean():
            KP, pr, pt, errors = candidate_K, refined.x[:3], refined.x[3:6], candidate_errors
            result['intrinsics_source'] += '; projector f/cy refined with soft priors and held-out selection'
        dots, survey, final = face_dots(K, rv, tv, KP, pr, pt)
        result['dot_survey'] = survey
        result['accepted_face_dots'] = len(dots)
        final_solution, final_found = solve_fulcrum(world, detect(cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)), K)
        if final_solution is None or len(final_found) < 3:
            raise RuntimeError('Cannot verify ff stability after dot sweep')
        common = set(found) & set(final_found)
        movement = max(np.linalg.norm(found[i]-final_found[i], axis=1).max() for i in common)
        result['marker_drift_cam_px'] = float(movement)
        if movement > 2:
            raise RuntimeError(f'ff/camera moved during sweep: {movement:.2f}px')
        if len(dots) < 9 or len(set(d[3] for d in dots)) < 2:
            result['validated'] = False
            raise RuntimeError('Need at least nine accepted dots on at least two faces')
        table_count = len(obj)
        obj = np.vstack([obj, [d[0] for d in dots]])
        img = np.vstack([img, [d[1] for d in dots]])
        groups += [d[2] for d in dots]
        surfaces += [d[3] for d in dots]
        train = np.r_[train, [d[2] % 3 != 0 for d in dots]]
        weights = np.r_[np.ones(table_count), np.full(len(dots),2.0)]
        initial = np.r_[pr.ravel(), pt.ravel(), KP[0,0], KP[1,2]]
        def dot_residual(parameters):
            k = KP.copy()
            k[0,0] = k[1,1] = parameters[6]
            k[1,2] = parameters[7]
            prediction = cv2.projectPoints(obj[train], parameters[:3], parameters[3:6], k,
                                           np.zeros(5))[0].reshape(-1,2)
            return np.r_[((prediction-img[train])*weights[train,None]).ravel(),
                         (parameters[6]-initial[6])/500, (parameters[7]-initial[7])/300]
        fit = least_squares(dot_residual, initial, loss='soft_l1', f_scale=3,
                            bounds=([-np.inf]*6+[500,-1500], [np.inf]*6+[4000,720]), max_nfev=500)
        KP[0,0] = KP[1,1] = fit.x[6]
        KP[1,2] = fit.x[7]
        pr, pt = fit.x[:3], fit.x[3:6]
        pred = cv2.projectPoints(obj, pr, pt, KP, np.zeros(5))[0].reshape(-1,2)
        errors = np.linalg.norm(pred-img, axis=1)
        dot_test = (~train) & (np.arange(len(obj)) >= table_count)
        if not dot_test.any():
            raise RuntimeError('No withheld face dots')
        result['face_dot_validation'] = dict(count=int(dot_test.sum()),
                                             mean_projector_px=float(errors[dot_test].mean()),
                                             max_projector_px=float(errors[dot_test].max()))
        print('face-dot validation:', result['face_dot_validation'], flush=True)
        result['projector'] = dict(K=KP.tolist(), rvec=pr.ravel().tolist(), tvec=pt.ravel().tolist(),
                                   train_mean_px=float(errors[train].mean()),
                                   heldout_mean_px=float(errors[~train].mean()),
                                   heldout_max_px=float(errors[~train].max()),
                                   surfaces={s:surfaces.count(s) for s in set(surfaces)},
                                   points_mm=obj.tolist(), pixels=img.tolist(), marker_ids=groups)
        result['validated'] = bool(errors[~train].mean() < 5 and errors[~train].max() < 12
                                   and errors[dot_test].mean() < 5 and errors[dot_test].max() < 12)
        print('projector held-out mean/max px:', errors[~train].mean(), errors[~train].max(), flush=True)
        overlay = np.zeros((720,1280,3), np.uint8)
        for face in face_polys().values():
            uv = cv2.projectPoints(face, pr, pt, KP, np.zeros(5))[0].reshape(-1,2)
            cv2.polylines(overlay, [np.round(uv).astype(np.int32)], True, (0,255,0), 2)
        overlay_path = OUT / f'{TAG}_outline.png'
        cv2.imwrite(str(overlay_path), overlay)
        td(f"op('/project1/ff_calibration_pattern').par.file={overlay_path.as_posix()!r}")
        time.sleep(0.8)
        validation = capture('camLogi', 'validation')
        capture('kinectC', 'validation')
        expected = np.zeros(validation.shape[:2], np.uint8)
        for face in face_polys().values():
            uv = cv2.projectPoints(face, rv, tv, K, np.zeros(5))[0].reshape(-1,2)
            cv2.polylines(expected, [np.round(uv).astype(np.int32)], True, 255, 1)
        b, g, r = [c.astype(np.int16) for c in cv2.split(validation)]
        observed = ((g - r > 25) & (b - r > 15) & (g > 100)).astype(np.uint8)
        roi = cv2.dilate(expected, np.ones((25,25), np.uint8)) > 0
        observed[~roi] = 0
        distance = cv2.distanceTransform((observed == 0).astype(np.uint8), cv2.DIST_L2, 3)
        residual = distance[expected > 0]
        outline = dict(expected_pixels=int(len(residual)), observed_pixels=int(observed.sum()),
                       mean_distance_cam_px=float(residual.mean()),
                       coverage_within_4px=float((residual <= 4).mean()))
        result['webcam_outline_assertion'] = outline
        result['validated'] = bool(result['validated'] and outline['coverage_within_4px'] > 0.9)
        print('webcam outline assertion:', outline, 'validated:', result['validated'], flush=True)
    finally:
        td(f"op('/project1/final_switch').par.index={previous}")
        path = OUT / f'{TAG}_solve.json'
        path.write_text(json.dumps(result, indent=2))
        print('Result:', path, flush=True)


if __name__ == '__main__':
    main()
