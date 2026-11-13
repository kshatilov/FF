import json
from pathlib import Path

import cv2
import numpy as np

last_signature = None
last_controls_sig = None
CONTROL_PATH = Path('C:/dev/ProjectSynchronicity/tools/pipe/ff/control_params.json')
CONTROL_MAP = {
    'Stillscale':'still_scale', 'Movescale':'move_scale', 'Speedref':'speed_ref',
    'Holdenabled':'hold_enabled', 'Holdseconds':'hold_seconds', 'Fadeseconds':'hold_fade_seconds',
    'Posnoise':'position_noise', 'Rotnoise':'rotation_noise', 'Accel':'acceleration_noise',
    'Angaccel':'angular_acceleration_noise', 'Gate':'gate', 'Reacquire':'reacquire_after'}
CONTROL_DEFAULTS = {'Stillscale':0.22,'Movescale':1.5,'Speedref':50.0,
    'Holdenabled':1,'Holdseconds':1.5,'Fadeseconds':0.75,'Posnoise':3.0,
    'Rotnoise':0.02,'Accel':100.0,'Angaccel':1.5,'Gate':60.0,
    'Reacquire':0.25,'Mode':'webcam'}


def selected_tracker():
    root = op('/project1')
    switch = root.op('final_switch')
    index = max(0,min(int(switch.par.index.eval()),len(switch.inputs)-1))
    selected = switch.inputs[index] if switch.inputs else None
    if selected is not None and selected.name in ('animation_pipe','animation_build','animation_tetris'):
        source = root.op('anim_source')
        if source is not None and source.inputs:
            selected = source.inputs[max(0,min(int(source.par.index.eval()),len(source.inputs)-1))]
    return selected


TRACKERS = {'ff_live_cam1': 'ff_live_logic', 'ff_live': 'ff_live_logic',
            'ff_live_kinect': 'ff_live_kinect_logic'}


def reset_controls():
    for name in ('tracking_controls','output_mode'):
        comp = op('/project1/'+name)
        if comp is None:
            continue
        for reset in comp.pars():
            if not reset.name.startswith('Reset') or not reset.eval():
                continue
            target_name = reset.name[5:]
            target = next((p for p in comp.pars() if p.name.lower()==target_name.lower()),None)
            if target is not None:
                target.val = CONTROL_DEFAULTS.get(target.name,target.default)
            reset.val = 0


def sync_controls():
    global last_controls_sig
    comp = op('/project1/tracking_controls')
    if comp is None:
        return
    data = {json_name: float(getattr(comp.par, par_name).eval())
            for par_name, json_name in CONTROL_MAP.items()
            if hasattr(comp.par, par_name)}
    signature = tuple(sorted(data.items()))
    if signature == last_controls_sig:
        return
    temporary = CONTROL_PATH.with_suffix('.td.tmp')
    temporary.write_text(json.dumps(data))
    try:
        temporary.replace(CONTROL_PATH)
    except PermissionError:
        pass
    last_controls_sig = signature


def onFrameStart(frame):
    reset_controls()
    sync_controls()
    global last_signature
    selected = selected_tracker()
    if selected is None or selected.name not in TRACKERS:
        return
    selected.cook(force=True)
    root = op('/project1')
    shaders = [s for s in (root.op('anim_height'),root.op('anim_markers')) if s is not None]
    if not shaders:
        return
    logic = root.op(TRACKERS[selected.name]).module
    state = logic.published_state
    signature = (selected.path,state.get('t'))
    if signature==last_signature:
        return
    last_signature = signature
    table = root.op('anim_table')
    tetris = root.op('anim_tetris')
    pose = state.get('projector')
    if not pose or state.get('proj_cmd',{}).get('mode') not in ('colors','holding'):
        for shader in shaders:
            shader.par.vec0valuew = 0
        if table is not None:
            table.par.vec0valuew = 0
        return
    T = np.eye(4)
    T[:3,:3] = cv2.Rodrigues(np.float64(pose['rvec']))[0]
    T[:3,3] = np.float64(pose['pos_mm'])
    rotation = T[:3,:3].T
    K = logic.KP
    vectors = [T[:3,3].tolist(), rotation[:,0]/K[0,0], rotation[:,1]/K[1,1],
               rotation @ np.array([-K[0,2]/K[0,0],-K[1,2]/K[1,1],1.0])]
    for shader in shaders:
        for i,vector in enumerate(vectors):
            for axis,value in zip('xyz',vector):
                getattr(shader.par,f'vec{i}value{axis}').val = float(value)
        shader.par.vec0valuew = 1
    Tw = np.eye(4)
    tbl = state.get('fulcrum_table') or {}
    if tbl.get('rvec') is not None and tbl.get('pos_mm') is not None:
        Tw[:3,:3] = cv2.Rodrigues(np.float64(tbl['rvec']))[0]
        Tw[:3,3] = np.float64(tbl['pos_mm'])
    if table is not None:
        for i,vector in enumerate(vectors):
            for axis,value in zip('xyz',vector):
                getattr(table.par,f'vec{i}value{axis}').val = float(value)
        table.par.vec0valuew = 1
        for row,vec_i in ((0,4),(1,5),(2,6)):
            for axis,value in zip('xyzw',Tw[row]):
                getattr(table.par,f'vec{vec_i}value{axis}').val = float(value)
    if tetris is not None:
        for row,vec_i in ((0,2),(1,3),(2,4)):
            for axis,value in zip('xyzw',Tw[row]):
                getattr(tetris.par,f'vec{vec_i}value{axis}').val = float(value)
