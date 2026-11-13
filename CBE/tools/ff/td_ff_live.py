import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('C:/dev/ProjectSynchronicity')
sys.path.insert(0, str(ROOT / 'tools/ff'))
import importlib
import pose_ekf
PoseEKF = importlib.reload(pose_ekf).PoseEKF

cv2.setNumThreads(2)
detector_parameters = cv2.aruco.DetectorParameters()
detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),detector_parameters)


def detect(gray):
    corners,ids,_ = detector.detectMarkers(gray)
    return {} if ids is None else {int(i):corners[j].reshape(4,2) for j,i in enumerate(ids.ravel())}


def rt(rv, tv):
    T = np.eye(4)
    T[:3,:3] = cv2.Rodrigues(np.float64(rv).reshape(3))[0]
    T[:3,3] = np.float64(tv).reshape(3)
    return T


def solve_fulcrum(world, found, K, distortion=None):
    found = {i:c for i,c in found.items() if i in world}
    if not found:
        return None, found
    obj = np.vstack([world[i] for i in found])
    img = np.vstack(list(found.values()))
    distortion = np.zeros(5) if distortion is None else distortion
    if pose_filter.accepted_time is not None and time.perf_counter()-pose_filter.accepted_time < 0.3:
        guess_rv,guess_tv = pose_filter.pose()
        ok,rv,tv = cv2.solvePnP(obj,img,K,distortion,guess_rv,guess_tv,
                               useExtrinsicGuess=True,flags=cv2.SOLVEPNP_ITERATIVE)
    else:
        ok,rv,tv = cv2.solvePnP(obj,img,K,distortion,flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None, found
    rv,tv = cv2.solvePnPRefineLM(obj,img,K,distortion,rv,tv)
    pred = cv2.projectPoints(obj,rv,tv,K,distortion)[0].reshape(-1,2)
    error = np.linalg.norm(pred-img,axis=1)
    return (rv,tv,float(error.mean()),float(error.max())), found


def face_polys():
    measured = spec['measured']
    ht,hb,H = measured['top_edge']/2,measured['base_edge']/2,measured['height']
    faces = {}
    for name,n in (('+y',(0,1)),('+x',(1,0)),('-y',(0,-1)),('-x',(-1,0))):
        t = np.array([-n[1],n[0],0.0])
        top,base = np.array([n[0]*ht,n[1]*ht,H]),np.array([n[0]*hb,n[1]*hb,0])
        faces[name] = np.float64([top+t*ht,base+t*hb,base-t*hb,top-t*ht])
    faces['top'] = np.float64([[-ht,ht,H],[ht,ht,H],[ht,-ht,H],[-ht,-ht,H]])
    return faces

CALIBRATION = ROOT / 'tools/pipe/ff/calibration_current.json'
calibration = json.loads(CALIBRATION.read_text())
spec = json.loads((ROOT / 'tools/pipe/ff/fulcrum_spec.json').read_text())
world = {int(i): np.float64(m['corners_TL_TR_BR_BL']) for i,m in spec['markers'].items()}
K = np.float64(calibration['webcam']['K'])
KP = np.float64(calibration['projector']['K'])
reference = rt(calibration['webcam']['rvec'], calibration['webcam']['tvec'])
projector_from_camera = rt(calibration['projector']['rvec'], calibration['projector']['tvec']) @ np.linalg.inv(reference)
frozen = json.loads((ROOT / 'tools/pipe/ff/aruco_solve_frozen.json').read_text())
table_from_camera = np.linalg.inv(rt(frozen['webcam']['rvec'], frozen['webcam']['tvec']))
calibration_mtime = CALIBRATION.stat().st_mtime
COLORS = {'+y': (1,0,0), '+x': (0,1,0), '-y': (0,0,1), '-x': (1,1,0), 'top': (1,0,1)}
last_tick = 0.0
last_write = 0.0
output = np.zeros((720,1280,4), np.uint8)
output[:,:,3] = 255
state = {}
CAMERA = 'webcam'
SOURCE = 'camLogi'
SHAPE = (480,640)
DISTORTION = np.zeros(5)
OUTPUT_INDEX = 5
# Dropouts during fast motion should hold/fade the last good projection
# instead of flashing black.  The EKF stays live — no stillness lock.
HOLD_ON_LOSS = True
HOLD_SECONDS = 1.5
HOLD_FADE_SECONDS = 0.75
pose_filter = PoseEKF(position_noise=3.0,rotation_noise=0.02,
                      acceleration_noise=100,angular_acceleration_noise=1.5,
                      gate=60,reacquire_after=0.25,stationary_enabled=False)
last_good_time = None
last_good_output = output.copy()
last_good_state = {}
raw_pose_history = []
track_marker_id = None
track_misses = 0
worker = None
published_state = {}
filter_samples = []
performance_samples = []
CONTROL_PARAMS = ROOT / 'tools/pipe/ff/control_params.json'
control_params = {}
control_mtime = None


def cfg(name, default):
    return control_params.get(name, default)


def load_control_params():
    global control_mtime, control_params
    try:
        mtime = CONTROL_PARAMS.stat().st_mtime
        if mtime == control_mtime:
            return
        control_mtime = mtime
        control_params.update(json.loads(CONTROL_PARAMS.read_text()))
        pos_noise = float(cfg('position_noise', pose_filter.position_noise if hasattr(pose_filter,'position_noise') else 3.0))
        rot_noise = float(cfg('rotation_noise', 0.02))
        pose_filter.measurement_noise = np.diag([pos_noise**2]*3+[rot_noise**2]*3)
        pose_filter.acceleration_noise = float(cfg('acceleration_noise', pose_filter.acceleration_noise))
        pose_filter.angular_acceleration_noise = float(cfg('angular_acceleration_noise', pose_filter.angular_acceleration_noise))
        pose_filter.gate = float(cfg('gate', pose_filter.gate))
        pose_filter.reacquire_after = float(cfg('reacquire_after', pose_filter.reacquire_after))
    except Exception:
        pass


def configure_kinect():
    global CAMERA, SOURCE, SHAPE, DISTORTION, OUTPUT_INDEX, K, projector_from_camera, table_from_camera
    registration = json.loads((ROOT/'tools/pipe/ff/td_kinect_registration.json').read_text())
    CAMERA, SOURCE, SHAPE, OUTPUT_INDEX = 'kinect', 'kinectC', (720,1280), 6
    K = np.float64(registration['K'])
    DISTORTION = np.float64(registration['D'])
    transform = np.float64(registration['webcam_from_kinect'])
    projector_from_camera = rt(calibration['projector']['rvec'], calibration['projector']['tvec']) @ np.linalg.inv(reference) @ transform
    table_from_camera = np.linalg.inv(rt(frozen['webcam']['rvec'], frozen['webcam']['tvec'])) @ transform


def reload_calibration():
    global calibration,K,KP,reference,projector_from_camera,table_from_camera,calibration_mtime
    try:
        mtime = CALIBRATION.stat().st_mtime
    except Exception:
        return
    if mtime == calibration_mtime:
        return
    calibration_mtime = mtime
    calibration.update(json.loads(CALIBRATION.read_text()))
    K = np.float64(calibration['webcam']['K'])
    KP = np.float64(calibration['projector']['K'])
    reference = rt(calibration['webcam']['rvec'], calibration['webcam']['tvec'])
    if CAMERA == 'kinect':
        configure_kinect()
    else:
        projector_from_camera = rt(calibration['projector']['rvec'], calibration['projector']['tvec']) @ np.linalg.inv(reference)
        table_from_camera = np.linalg.inv(rt(frozen['webcam']['rvec'], frozen['webcam']['tvec']))


def motion_process_scale(now, rv, tv):
    global raw_pose_history
    raw_pose_history.append((now,np.float64(tv).reshape(3).copy(),np.float64(rv).reshape(3).copy()))
    raw_pose_history[:] = [s for s in raw_pose_history if now-s[0] <= 0.7]
    if len(raw_pose_history) < 5:
        return 1.0
    t0,pos0,rot0 = raw_pose_history[0]
    dt = max(now-t0,1e-3)
    linear = np.linalg.norm(np.float64(tv).reshape(3)-pos0)/dt
    r_now = cv2.Rodrigues(np.float64(rv).reshape(3,1))[0]
    r_old = cv2.Rodrigues(rot0.reshape(3,1))[0]
    angular = np.linalg.norm(cv2.Rodrigues(r_now @ r_old.T)[0])/dt
    # Treat rotation at roughly a 300 mm lever arm for the still/moving split.
    speed = max(linear,angular*300.0)
    still = float(cfg('still_scale',0.22))
    speed_ref = max(1.0,float(cfg('speed_ref',50.0)))
    moving = max(still,float(cfg('move_scale',1.5)))
    return float(np.clip(still+speed/speed_ref,still,moving))


def tracked_roi_detect(gray):
    if track_marker_id is None or track_marker_id not in world or pose_filter.accepted_time is None:
        return {}
    rv,tv = pose_filter.pose()
    uv = cv2.projectPoints(world[track_marker_id],rv,tv,K,DISTORTION)[0].reshape(-1,2)
    center = uv.mean(axis=0)
    radius = max(120.0,float(np.linalg.norm(uv-center,axis=1).max())+80.0)
    x0 = max(0,int(center[0]-radius)); x1 = min(gray.shape[1],int(center[0]+radius))
    y0 = max(0,int(center[1]-radius)); y1 = min(gray.shape[0],int(center[1]+radius))
    if x1-x0 < 20 or y1-y0 < 20:
        return {}
    found = detect(gray[y0:y1,x0:x1])
    return {i:c+np.array([x0,y0],dtype=c.dtype) for i,c in found.items() if i == track_marker_id}


def best_marker(rv,tv,found):
    if not found:
        return None
    scores = {}
    for marker_id,corners in found.items():
        pred = cv2.projectPoints(world[marker_id],rv,tv,K,DISTORTION)[0].reshape(-1,2)
        scores[marker_id] = float(np.linalg.norm(pred-corners,axis=1).mean())
    return min(scores,key=scores.get)


def publish_projection(rv,tv,found,mean,mode='colors'):
    current = rt(rv,tv)
    projected_pose = projector_from_camera @ current
    pr = cv2.Rodrigues(projected_pose[:3,:3])[0]
    pt = projected_pose[:3,3]
    polys = face_polys()
    ordered = sorted(polys, key=lambda name: (projected_pose @ np.r_[polys[name].mean(axis=0),1])[2], reverse=True)
    for name in ordered:
        q = polys[name]
        q = q.mean(axis=0) + (q-q.mean(axis=0))*0.94
        camera_points = q @ projected_pose[:3,:3].T + pt
        if np.any(camera_points[:,2] <= 0):
            continue
        uv = cv2.projectPoints(q,pr,pt,KP,np.zeros(5))[0].reshape(-1,2)
        if not np.isfinite(uv).all() or np.abs(uv).max() > 10000:
            continue
        state['face_quads'][name] = uv.tolist()
        pixels = np.round(uv*np.array([1,-1])+np.array([0,719])).astype(np.int32)
        cv2.fillConvexPoly(output,pixels,tuple(int(c*160) for c in COLORS[name])+(255,))
    for corners in world.values():
        q = corners.mean(axis=0)+(corners-corners.mean(axis=0))*1.12
        uv = cv2.projectPoints(q,pr,pt,KP,np.zeros(5))[0].reshape(-1,2)
        if np.isfinite(uv).all() and np.abs(uv).max() < 10000:
            pixels = np.round(uv*np.array([1,-1])+np.array([0,719])).astype(np.int32)
            cv2.fillConvexPoly(output,pixels,(0,0,0,255))
    table_pose = table_from_camera @ current
    camera_pos = -current[:3,:3].T @ current[:3,3]
    projector_pos = -projected_pose[:3,:3].T @ pt
    state.update(proj_cmd={'mode':mode},
                 devices={CAMERA:dict(pos_mm=camera_pos.tolist(),rvec=np.ravel(rv).tolist(),
                                        tvec=np.ravel(tv).tolist(),markers=sorted(found),reproj=mean,
                                        status='tracked' if len(found)>1 else 'partial',weak=len(found)<2,
                                        table_pos_mm=table_from_camera[:3,3].tolist())},
                 projector=dict(pos_mm=projector_pos.tolist(),rvec=pr.ravel().tolist(),
                                table_pos_mm=(table_from_camera @ np.r_[(-projector_from_camera[:3,:3].T @ projector_from_camera[:3,3]),1])[:3].tolist()),
                 fulcrum_table=dict(pos_mm=table_pose[:3,3].tolist(),
                                    rvec=cv2.Rodrigues(table_pose[:3,:3])[0].ravel().tolist(),
                                    weak=len(found)<2,src=[CAMERA]))


def process_frame(source,now):
    global output, state, last_good_time, last_good_output, last_good_state, track_marker_id, track_misses
    reload_calibration()
    load_control_params()
    marks = [time.perf_counter()]
    output.fill(0)
    output[:,:,3] = 255
    state = dict(t=time.time(), paused=False, source='TouchDesigner '+SOURCE,
                 calibration=str(CALIBRATION), calibration_fully_validated=calibration['validated'],
                 faces={k:[int(255*c) for c in v] for k,v in COLORS.items()},
                 quad_margin=0.94, proj_cmd={'mode':'lost'}, devices={}, face_quads={})
    try:
        if source is None:
            raise RuntimeError(SOURCE+' frame unavailable')
        marks.append(time.perf_counter())
        gray = source if source.ndim==2 else np.ascontiguousarray((source[::-1,:,0]*255).astype(np.uint8))
        marks.append(time.perf_counter())
        scale_xy = np.float32([SHAPE[1]/gray.shape[1],SHAPE[0]/gray.shape[0]])
        solver_mode = 'full'
        found = {}
        if track_marker_id is not None and pose_filter.accepted_time is not None and now-pose_filter.accepted_time < 0.35:
            roi_found = tracked_roi_detect(gray)
            if track_marker_id in roi_found:
                found = {i:c*scale_xy for i,c in roi_found.items()}
                solver_mode = 'one'
        if solver_mode == 'full':
            found = {i:c*scale_xy for i,c in detect(gray).items()}
        marks.append(time.perf_counter())
        solution, found = solve_fulcrum(world, found, K, DISTORTION)
        marks.append(time.perf_counter())
        if solution is None or len(found) < (1 if solver_mode == 'one' else 2):
            if solver_mode == 'one':
                found = {i:c*scale_xy for i,c in detect(gray).items()}
                solver_mode = 'full'
                solution, found = solve_fulcrum(world, found, K, DISTORTION)
            if solution is None or len(found)<2:
                raise RuntimeError('Fulcrum lost: insufficient markers for a stable pose')
        rv,tv,mean,maximum = solution
        if mean > 3 or maximum > 8:
            raise RuntimeError(f'Pose rejected: mean={mean:.2f}, max={maximum:.2f}px')
        raw_rv,raw_tv = rv.copy(),tv.copy()
        scale = max(1.0,mean/1.5)*np.sqrt(3/max(2,len(found)))
        process_scale = motion_process_scale(now,raw_rv,raw_tv)
        (rv,tv),accepted = pose_filter.update(rv,tv,now,noise_scale=scale,process_scale=process_scale)
        state['filter'] = dict(type='multiplicative_pose_ekf',stationary=pose_filter.stationary,measurement_accepted=accepted,
                               process_scale=process_scale,solver_mode=solver_mode,tracking_marker=track_marker_id,
                               innovation_squared=pose_filter.innovation_squared,
                               rejected_frames=pose_filter.rejections,
                               velocity_mm_s=pose_filter.velocity.tolist(),
                               angular_velocity_rad_s=pose_filter.angular_velocity.tolist(),
                               raw_rvec=raw_rv.ravel().tolist(),raw_tvec=raw_tv.ravel().tolist(),
                               filtered_rvec=rv.ravel().tolist(),filtered_tvec=tv.ravel().tolist())
        filter_samples.append(dict(t=now,raw=raw_tv.ravel().tolist(),filtered=tv.ravel().tolist(),accepted=accepted))
        del filter_samples[:-120]
        if not accepted:
            track_misses += 1
            if track_misses > 2:
                track_marker_id = None
            publish_projection(rv,tv,found,mean,mode='holding')
            state['tracking_status'] = 'coasting'
            return output,state
        track_misses = 0
        if solver_mode == 'full':
            track_marker_id = best_marker(rv,tv,found)
        marks.append(time.perf_counter())
        publish_projection(rv,tv,found,mean)
        last_good_time = now
        last_good_output[:] = output
        last_good_state = state.copy()
    except Exception as error:
        held = False
        if HOLD_ON_LOSS and float(cfg('hold_enabled',1)):
            hold_seconds = max(0.1,float(cfg('hold_seconds',HOLD_SECONDS)))
            fade_seconds = max(0.05,float(cfg('hold_fade_seconds',HOLD_FADE_SECONDS)))
            anchor = pose_filter.accepted_time if pose_filter.accepted_time is not None else last_good_time
            age = float('inf') if anchor is None else max(0,now-anchor)
            visibility = max(0.0,min(1.0,(hold_seconds-age)/fade_seconds)) if np.isfinite(age) else 0.0
            coast = pose_filter.predict(now) if pose_filter.time is not None else None
            if coast is not None and visibility > 0:
                try:
                    publish_projection(coast[0],coast[1],{},float('nan'),mode='holding')
                    if visibility < 1.0:
                        output[:,:,:3] = cv2.convertScaleAbs(output[:,:,:3],alpha=visibility)
                        output[:,:,3] = 255
                    state.update(occlusion_age_s=age,
                                 tracking_status='holding' if visibility>=1.0 else 'fading')
                    held = True
                except Exception:
                    held = False
            if not held and visibility > 0 and last_good_state:
                output[:] = cv2.convertScaleAbs(last_good_output,alpha=visibility)
                output[:,:,3] = 255
                state = last_good_state.copy()
                state.update(t=time.time(),proj_cmd={'mode':'holding'},occlusion_age_s=age,
                             tracking_status='holding' if age<=fade_seconds else 'fading')
                state['fulcrum_table'] = dict(state['fulcrum_table'],weak=True)
                held = True
            if not held:
                output[:,:,:3] = 0
        else:
            output[:,:,:3] = 0
        state['error'] = str(error)
    performance_samples.append((time.perf_counter()-marks[0])*1000)
    del performance_samples[:-120]
    return output,state


def shutdown():
    global worker
    if worker is not None:
        worker.shutdown()
        worker = None


def cook(scriptOp):
    global worker,last_tick,last_write,published_state,state
    now = time.perf_counter()
    reload_calibration()
    if worker is None:
        import ff_worker
        worker = importlib.reload(ff_worker).ExternalTracker(CAMERA)
    result = worker.poll()
    if result is not None:
        image,published_state = result
        state = published_state
        scriptOp.copyNumpyArray(image)
        performance_samples.append(state['performance']['worker_ms'])
        del performance_samples[:-120]
        f = state.get('filter')
        if f is not None and state.get('proj_cmd',{}).get('mode')=='colors':
            filter_samples.append(dict(t=state['performance']['measurement_timestamp'],raw=f['raw_tvec'],
                                       filtered=f['filtered_tvec'],accepted=f['measurement_accepted']))
            del filter_samples[:-120]
    source = scriptOp.inputs[0].numpyArray(delayed=True)
    if source is not None:
        gray = np.ascontiguousarray((source[::-1,:,0]*255).astype(np.uint8))
        worker.submit(gray,now)
    if published_state and now-last_write > 0.4 and op('/project1/ff_live_tick').module.selected_tracker()==scriptOp:
        path = ROOT / 'tools/pipe/ff/fulcrum_poses.json'
        temporary = path.with_suffix('.td.tmp')
        temporary.write_text(json.dumps(published_state))
        try:
            temporary.replace(path)
        except PermissionError:
            pass
        last_write = now

