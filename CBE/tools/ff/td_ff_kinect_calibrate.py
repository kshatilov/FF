import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / 'tools/kinect'))
sys.path.insert(0,str(ROOT / 'tools/ff'))
from kincap import Kin
from td_ff_solve import td, capture, TAG
from fulcrum_check import detect


def pose(image, world, K, distortion):
    found = {i:c for i,c in detect(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)).items() if i in world}
    if len(found) < 3:
        raise RuntimeError(f'Need three ff markers, found {sorted(found)}')
    obj,img = np.vstack([world[i] for i in found]),np.vstack(list(found.values()))
    ok,rv,tv = cv2.solvePnP(obj,img,K,distortion,flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise RuntimeError('Reference pose failed')
    rv,tv = cv2.solvePnPRefineLM(obj,img,K,distortion,rv,tv)
    predicted = cv2.projectPoints(obj,rv,tv,K,distortion)[0].reshape(-1,2)
    errors = np.linalg.norm(predicted-img,axis=1)
    if errors.mean()>3 or errors.max()>8:
        raise RuntimeError(f'Reference residual too high: {errors.mean():.2f}/{errors.max():.2f}')
    T = np.eye(4)
    T[:3,:3],T[:3,3] = cv2.Rodrigues(rv)[0],tv.ravel()
    return T,dict(markers=sorted(found),mean_px=float(errors.mean()),max_px=float(errors.max())),found


def main():
    sensor = Kin()
    time.sleep(2)
    points = np.float32([(x,y,z) for z in (0.65,0.9,1.3,1.8)
                         for y in np.linspace(-0.3,0.3,9) for x in np.linspace(-0.5,0.5,13)])
    pixels = sensor.camera_pts_to_color(points*np.float32([1,-1,1]))
    good = np.isfinite(pixels).all(axis=1)&(pixels[:,0]>0)&(pixels[:,0]<1920)&(pixels[:,1]>0)&(pixels[:,1]<1080)
    points,pixels = points[good],pixels[good]
    K0 = np.float64([[1060,0,1004],[0,1060,548],[0,0,1]])
    rms,K,D,_,_ = cv2.calibrateCamera([points],[pixels],(1920,1080),K0,np.zeros(5),
                                      flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    if rms > 0.5 or len(points)<200:
        raise RuntimeError(f'Factory mapper pinhole fit failed: rms={rms}, points={len(points)}')
    K[:2] *= 2/3
    print('Factory mapper fit:',rms,'K:',K.tolist(),'D:',D.ravel().tolist(),flush=True)
    calibration = json.loads((ROOT/'smoke/ff_20261104_143135_solve.json').read_text())
    spec = json.loads((ROOT/'tools/pipe/ff/fulcrum_spec.json').read_text())
    world = {int(i):np.float64(m['corners_TL_TR_BR_BL']) for i,m in spec['markers'].items()}
    previous = td("op('/project1/final_switch').par.index.eval()",'--eval')
    try:
        td("op('/project1/final_switch').par.index=0; op('/project1/OUT').cook(force=True); op('/project1/kinectC').cook(force=True)")
        time.sleep(.7)
        td(f"op('/project1/camLogi').cook(force=True); op('/project1/kinectC').cook(force=True); op('/project1/camLogi').save('{(ROOT/'smoke'/f'{TAG}_kinref_camLogi.png').as_posix()}'); op('/project1/kinectC').save('{(ROOT/'smoke'/f'{TAG}_kinref_kinectC.png').as_posix()}')")
        webcam = cv2.imread(str(ROOT/'smoke'/f'{TAG}_kinref_camLogi.png'))
        kinect = cv2.imread(str(ROOT/'smoke'/f'{TAG}_kinref_kinectC.png'))
        if kinect.shape[:2] != (720,1280):
            raise RuntimeError('Kinect resolution must be 1280x720')
        WC,ws,wfound = pose(webcam,world,np.float64(calibration['webcam']['K']),np.zeros(5))
        KC,ks,kfound = pose(kinect,world,K,D)
        _,_,after = pose(capture('camLogi','kinref_after'),world,np.float64(calibration['webcam']['K']),np.zeros(5))
        drift = max(np.linalg.norm(wfound[i]-after[i],axis=1).max() for i in set(wfound)&set(after))
        if drift>1.5:
            raise RuntimeError(f'ff moved during registration: {drift:.2f}px; retry with target still')
        result = dict(timestamp=TAG,source='kinectC',resolution=[1280,720],K=K.tolist(),D=D.ravel().tolist(),
                      sdk_mapper_fit_rms_px=rms,webcam=ws,kinect=ks,reference_drift_px=float(drift),
                      webcam_from_kinect=(WC@np.linalg.inv(KC)).tolist(),
                      camera_from_reference=KC.tolist())
        path = ROOT/'tools/pipe/ff/td_kinect_registration.json'
        path.write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
    finally:
        td(f"op('/project1/final_switch').par.index={previous}")


if __name__=='__main__':
    main()
