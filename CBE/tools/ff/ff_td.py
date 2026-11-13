"""TD-side FF pattern/capture transport; pixel arrays never use shared PNG paths."""

import os
import runpy


ROOT = '/project1'
WIRE = runpy.run_path(os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'hg', 'hg_wire.py'))
CALLBACKS = '''def onCook(scriptTOP):
    scriptTOP.copyNumpyArray(scriptTOP.storage['ff_pattern_rgba'])
'''


def owned(kind, name, expected):
    path = ROOT + '/' + name
    node = op(path)
    if node is None:
        node = op(ROOT).create(kind, name)
        node.store('ff_owned', True)
    if not node.storage.get('ff_owned') or node.OPType.lower() != expected:
        raise ValueError(path + ' exists and is not the expected FF-owned operator')
    return node


def rpc(payload):
    try:
        action = payload.get('action')
        if action == 'capture':
            path = payload.get('top', '')
            if not path.startswith(ROOT + '/') or op(path) is None:
                raise ValueError('capture requires an existing /project1 TOP')
            node = op(path)
            node.cook(force=True)
            image = node.numpyArray(delayed=False)
            if image is None:
                raise ValueError('TD camera returned no frame')
            return {'ok': True, 'array': WIRE['encode_array'](
                WIRE['td_color_to_bgr'](image))}
        if action != 'pattern':
            raise ValueError('unknown FF bridge action')
        session = op(ROOT).storage.get('ff_calibration_session') or {}
        if not payload.get('owner') or payload['owner'] != session.get('owner'):
            raise ValueError('FF pattern requires the active calibration owner')
        image = WIRE['decode_array'](payload['pattern'])
        if image.shape != (720, 1280, 3):
            raise ValueError('FF projector raster must be 1280x720 BGR')
        rgba = WIRE['bgr_to_td_rgba'](image)
        callbacks = owned(textDAT, 'ff_calibration_wire_callbacks', 'textdat')
        pattern = owned(scriptTOP, 'ff_calibration_wire', 'scripttop')
        callbacks.text = CALLBACKS
        pattern.par.callbacks = callbacks.path
        pattern.storage['ff_pattern_rgba'] = rgba
        pattern.cook(force=True)
        if pattern.errors():
            raise ValueError(str(pattern.errors()))
        switch = op(ROOT + '/final_switch')
        if pattern not in switch.inputs:
            pattern.outputConnectors[0].connect(switch)
        switch.par.index.val = list(switch.inputs).index(pattern)
        switch.par.index.mode = ParMode.CONSTANT
        return {'ok': True, 'pattern_top': pattern.path}
    except Exception as error:
        return {'ok': False, 'error': str(error)}
