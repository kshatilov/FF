import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('C:/dev/ProjectSynchronicity')
sys.path.insert(0, str(ROOT / 'tools/ff'))

worker = None
worker_camera = None
worker_started = None
published_state = {}
last_tick = 0.0
last_result_time = None
EDGE_PAIRS = [('A', 'B'), ('B', 'C'), ('C', 'A'), ('A', 'D'), ('B', 'D'), ('C', 'D'),
              ('A', 'E'), ('C', 'E'), ('D', 'E')]

# cam_and_kinect switch index -> (tracker camera, TOP name supplying the color frame)
SOURCE_SELECT = {0: ('kinect', 'kinectC'), 1: ('webcam', None), 2: ('kinect_depth', 'kinectC')}
STARTUP_GRACE = 3.0             # worker spawn window before the watchdog fires
WATCHDOG_SECONDS = 0.6          # steady-state: no fresh result -> clear to black
BLACK = np.zeros((720, 1280, 4), np.uint8)
BLACK[:, :, 3] = 255


def current_source():
    switch = op('/project1/cam_and_kinect')
    if switch is None or not switch.inputs:
        return 'webcam', None
    index = max(0, min(int(switch.par.index.eval()), len(switch.inputs)-1))
    camera, top = SOURCE_SELECT.get(index, ('webcam', None))
    return camera, top


def projection_enabled():
    controls = op('/project1/ffv2_controls')
    if controls is None:
        return False
    par = getattr(controls.par, 'Experimentalprojection', None)
    return bool(par.eval()) if par is not None else False


def to_bgr(rgba):
    """TD numpy RGBA float bottom-up -> worker OpenCV BGR uint8 top-down."""
    return np.ascontiguousarray((rgba[::-1, :, :3][:, :, ::-1]*255).astype(np.uint8))


def clear(scriptOp):
    """Native-resolution black so no stale projection survives a transition."""
    scriptOp.copyNumpyArray(BLACK.copy())


def kill_worker():
    global worker, worker_camera, worker_started, last_result_time
    if worker is not None:
        try:
            worker.shutdown()
        except Exception:
            pass
    worker = None
    worker_camera = None
    worker_started = None
    last_result_time = None


def ensure_worker(scriptOp):
    global worker, worker_camera, worker_started, last_result_time
    camera, _ = current_source()
    if worker is not None and (worker_camera != camera or worker.process.poll() is not None):
        # source changed or worker died: drop it and its pending results
        kill_worker()
        published_state.clear()
        clear(scriptOp)
    if worker is None:
        import ffv2_worker
        import importlib
        worker = importlib.reload(ffv2_worker).ExternalTracker(camera)
        worker_camera = camera
        worker_started = time.perf_counter()
        last_result_time = None
        clear(scriptOp)
    return worker


def shutdown():
    global published_state
    kill_worker()
    published_state = {}


def watchdog_timeout(now):
    """Startup grace applies until the first result; steady-state after."""
    if worker_started is None:
        return False
    if last_result_time is None:
        return now-worker_started > STARTUP_GRACE
    return now-last_result_time > WATCHDOG_SECONDS


def cook(scriptOp):
    global last_tick, published_state, last_result_time
    now = time.perf_counter()
    last_tick = now
    try:
        w = ensure_worker(scriptOp)
    except Exception as error:
        published_state = dict(t=time.time(), status='lost', error=repr(error),
                               projection_enabled=False)
        clear(scriptOp)
        return
    try:
        result = w.poll()
    except Exception as error:
        published_state = dict(t=time.time(), status='lost', error=repr(error),
                               projection_enabled=projection_enabled())
        kill_worker()
        clear(scriptOp)
        return
    if result is not None:
        last_result_time = now
        image, published_state = result
        published_state['projection_enabled'] = projection_enabled()
        if projection_enabled():
            # worker emits top-down RGBA; TD numpy arrays are bottom-up — flip once
            scriptOp.copyNumpyArray(np.ascontiguousarray(image[::-1]))
        else:
            # experimental prototype: projector output stays OFF unless opted in
            clear(scriptOp)
    elif watchdog_timeout(now):
        published_state = dict(t=time.time(), status='lost',
                               error='watchdog: no worker result',
                               projection_enabled=projection_enabled())
        kill_worker()
        clear(scriptOp)
        return
    camera, top = current_source()
    if top is not None:
        source_op = op('/project1/'+top)
        source = source_op.numpyArray(delayed=True) if source_op is not None else None
    else:
        source = scriptOp.inputs[0].numpyArray(delayed=True) if scriptOp.inputs else None
    if source is not None:
        try:
            w.submit(to_bgr(source), now)
        except Exception as error:
            published_state = dict(t=time.time(), status='lost', error=repr(error),
                                   camera=camera, projection_enabled=projection_enabled())
            kill_worker()
            clear(scriptOp)


def cook_debug(scriptOp):
    """Camera overlay for the ffv2_debug TOP — never touches projector output."""
    _, top = current_source()
    source_op = op('/project1/'+top) if top is not None else \
        (scriptOp.inputs[0] if scriptOp.inputs else None)
    if source_op is None:
        return
    source = source_op.numpyArray(delayed=True)
    if source is None:
        return
    try:
        view = (source[::-1, :, :3][:, :, ::-1]*255).astype(np.uint8).copy()
    except Exception:
        return
    debug = (published_state or {}).get('debug')
    if debug:
        labels = debug['labels']
        uv = {l: np.array(p) for l, p in zip(labels, debug['uv'])}
        for a, b in EDGE_PAIRS:
            if a in uv and b in uv:
                cv2.line(view, tuple(np.round(uv[a]).astype(int)),
                         tuple(np.round(uv[b]).astype(int)), (255, 255, 255), 1)
        for l, p in uv.items():
            cv2.putText(view, l, tuple(np.round(p+[2, -2]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.putText(view, 'ffv2 '+(published_state or {}).get('status', '?'), (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    out = np.zeros((view.shape[0], view.shape[1], 4), np.float32)
    out[:, :, :3] = (view[:, :, ::-1].astype(np.float32))/255.0
    out[:, :, 3] = 1.0
    scriptOp.copyNumpyArray(np.ascontiguousarray(out[::-1]))
