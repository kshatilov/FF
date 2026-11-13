"""Measure projected-vs-physical marker misalignment and write alignment_correction.json.
Run locally (not in TD): needs a saved camLogi frame + pose snapshot.
Uses whichever tracker is active (webcam or kinect); merges with existing correction."""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('C:/dev/ProjectSynchronicity')
CORR = ROOT/'tools/pipe/ff/alignment_correction.json'
spec = json.loads((ROOT/'tools/pipe/ff/fulcrum_spec.json').read_text())
calib = json.loads((ROOT/'tools/pipe/ff/calibration_current.json').read_text())
reg = json.loads((ROOT/'tools/pipe/ff/td_kinect_registration.json').read_text())
state = json.loads((ROOT/'tools/pipe/ff/align_pose_snapshot.json').read_text())
m = spec['measured']
hb, ht, H = m['base_edge']/2, m['top_edge']/2, m['height']
SLOPE = (hb-ht)/H
FACE = {11:'+y',12:'+x',13:'-y',14:'-x',10:'top'}
PLANE = {'+y':(np.array([0.,1.,SLOPE]),hb),'+x':(np.array([1.,0.,SLOPE]),hb),
         '-y':(np.array([0.,-1.,SLOPE]),hb),'-x':(np.array([-1.,0.,SLOPE]),hb),
         'top':(np.array([0.,0.,1.]),H)}

def rt(rv,tv):
    T = np.eye(4); T[:3,:3] = cv2.Rodrigues(np.float64(rv).reshape(3))[0]
    T[:3,3] = np.float64(tv).reshape(3); return T

K = np.float64(calib['webcam']['K'])
KP = np.float64(calib['projector']['K'])
wfk = np.float64(reg['webcam_from_kinect'])
device, dev = next(iter(state['devices'].items()))
current = rt(dev['rvec'], dev['tvec'])
cam_from_local = wfk @ current if device == 'kinect' else current
proj_from_cam = rt(calib['projector']['rvec'],calib['projector']['tvec']) @ np.linalg.inv(rt(calib['webcam']['rvec'],calib['webcam']['tvec']))

img = cv2.imread(str(ROOT/'smoke/align_cam.png'))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
corners, ids, _ = det.detectMarkers(gray)
found = {} if ids is None else {int(i): corners[j].reshape(4,2) for j,i in enumerate(ids.ravel())}
print('device', device, 'detected', sorted(found))
Kinv = np.linalg.inv(K)
all_spec, all_est, markers_out = [], [], {}
for mid, quad in sorted(found.items()):
    X = np.float64(spec['markers'][str(mid)]['corners_TL_TR_BR_BL'])
    pred = cv2.projectPoints(X, cv2.Rodrigues(cam_from_local[:3,:3])[0], cam_from_local[:3,3], K, np.zeros(5))[0].reshape(-1,2)
    n,b = PLANE[FACE[mid]]
    n_w = cam_from_local[:3,:3] @ n
    b_w = b + n_w @ cam_from_local[:3,3]
    est = []
    for px in quad:
        d = Kinv @ np.array([px[0],px[1],1.0])
        t = b_w / (n_w @ d)
        est.append((np.linalg.inv(cam_from_local) @ np.r_[d*t,1.0])[:3])
    est = np.array(est)
    off = quad - pred
    print(f"id{mid}: px_off mean {off.mean(0).round(2)} max {np.abs(off).max():.2f}  mm_off mean {(est-X).mean(0).round(2)}")
    all_spec.append(X); all_est.append(est)
    ctr = est.mean(0); u = est[1]-est[0]; v = est[3]-est[0]
    markers_out[str(mid)] = dict(center=ctr.tolist(), u=(u/np.linalg.norm(u)).tolist(),
                                 v=(v/np.linalg.norm(v)).tolist(),
                                 half_u=float(np.linalg.norm(u)/2), half_v=float(np.linalg.norm(v)/2))

previous = json.loads(CORR.read_text()) if CORR.exists() else {}
if len(found) >= 4:
    all_spec = np.vstack(all_spec); all_est = np.vstack(all_est)
    cs, ce = all_spec.mean(0), all_est.mean(0)
    U,S,Vt = np.linalg.svd((all_spec-cs).T @ (all_est-ce))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1; R = Vt.T @ U.T
    T_est = np.eye(4); T_est[:3,:3] = R; T_est[:3,3] = ce - R@cs
    sane = np.abs(T_est[:3,3]).max() < 20 and np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])) < 15
    if not sane:
        print('T_est rejected (insane fit), keeping previous')
        T_est = np.float64(previous['T_est']) if 'T_est' in previous else np.eye(4)
    else:
        print('T_est trans', T_est[:3,3].round(3), 'rot deg', round(float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0]))),3))
else:
    print('too few markers for Kabsch, keeping previous T_est')
    T_est = np.float64(previous['T_est']) if 'T_est' in previous else np.eye(4)
merged = dict(previous.get('markers', {}))
merged.update(markers_out)
json.dump(dict(T_est=T_est.tolist(), markers=merged), open(CORR,'w'), indent=1)
print('markers in file', sorted(merged))
