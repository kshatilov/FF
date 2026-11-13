"""Shared FF publication and reversible TouchDesigner output ownership."""

import ast
import json
import os
from pathlib import Path
import uuid


ROOT = Path(__file__).resolve().parents[2]
PIPE = ROOT / 'tools/pipe/ff'
SESSION_KEY = 'ff_calibration_session'


def publish_documents(documents):
    """Stage every JSON first; roll back on an interrupted publication.

    Callers order registration before the rig pointer, then viewer geometry.
    Rendering remains in the calibration session until publication finishes.
    Previous bytes are retained in pipe/ff/revisions for recovery.
    """
    revision = uuid.uuid4().hex
    archive = PIPE / 'revisions' / revision
    staged, previous, installed = {}, {}, []
    try:
        for name, document in documents.items():
            path = PIPE / name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = (json.dumps(document, indent=2, allow_nan=False) + '\n').encode()
            previous[path] = path.read_bytes() if path.exists() else None
            if previous[path] is not None:
                archive.mkdir(parents=True, exist_ok=True)
                (archive / name).write_bytes(previous[path])
            temp = path.with_name(path.name + '.' + revision + '.pending')
            temp.write_bytes(data)
            staged[path] = (temp, data)
        for path, (temp, _) in staged.items():
            os.replace(temp, path)
            installed.append(path)
    except BaseException:
        for path in reversed(installed):
            # Do not overwrite an out-of-band user edit during recovery.
            if path.read_bytes() != staged[path][1]:
                raise RuntimeError(
                    'FF publication changed concurrently; backups retained at '
                    + str(archive))
            if previous[path] is None:
                path.unlink()
            else:
                temp = path.with_name(path.name + '.' + revision + '.rollback')
                temp.write_bytes(previous[path])
                os.replace(temp, path)
        raise
    finally:
        for temp, _ in staged.values():
            if temp.exists():
                temp.unlink()
    return revision


def evaluate_json(module, expression):
    raw = module.td("__import__('json').dumps(" + expression + ')', '--eval')
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(ast.literal_eval(raw))


def restore_output(module, snapshot):
    current = evaluate_json(
        module, "op('/project1').storage.get(%r)" % SESSION_KEY)
    if not current or current.get('owner') != snapshot.get('owner'):
        raise RuntimeError('FF session ownership changed; refusing to overwrite output')
    commands = []
    for item in snapshot['parameters']:
        target = "op(%r).par[%r]" % (item['node'], item['name'])
        commands.extend([
            target + '.expr=' + repr(item['expr']),
            target + '.val=' + repr(item['value']),
            target + '.mode=getattr(ParMode,' + repr(item['mode']) + ')',
        ])
    commands.append("op('/project1').unstore(%r)" % SESSION_KEY)
    module.td('\n'.join(commands))


class OutputSession:
    def __init__(self, module):
        self.module = module
        self.snapshot = None

    def __enter__(self):
        state = evaluate_json(self.module, """{
            'hg_busy': bool(op('/project1/hg') and
                op('/project1/hg').storage.get('hg_session')),
            'ff_busy': bool(op('/project1').storage.get('ff_calibration_session')),
            'parameters': [
                {'node': node, 'name': name, 'value': op(node).par[name].eval(),
                 'expr': op(node).par[name].expr,
                 'mode': op(node).par[name].mode.name}
                for node, name in [
                    ('/project1/final_switch', 'index'),
                    ('/project1/cam_and_kinect', 'index'),
                    ('/project1/output_mode', 'Mode')]
                if op(node) is not None]
        }""")
        if state['hg_busy'] or state['ff_busy']:
            raise RuntimeError('a calibration session is active; restore it first')
        if not any(p['node'].endswith('/final_switch') for p in state['parameters']):
            raise RuntimeError('TouchDesigner final_switch is missing')
        self.snapshot = {'owner': uuid.uuid4().hex,
                         'parameters': state['parameters']}
        self.module.td("op('/project1').store(%r,%r)" % (
            SESSION_KEY, self.snapshot))
        self.module._ff_session_owner = self.snapshot['owner']
        return self

    def __exit__(self, *_):
        try:
            restore_output(self.module, self.snapshot)
        except Exception as error:
            raise RuntimeError(
                'TD restoration failed; snapshot retained. Run '
                '.\\ff calib --restore: ' + str(error)) from error


def camera_record(module, camera, result):
    return {
        'schema': 'ff.camera_calibration', 'schema_version': 1,
        'camera_id': camera, 'timestamp': module.TAG,
        'units': 'mm', 'frame': module._spec['frame'],
        'source_top': '/project1/' + ('camLogi' if camera == 0 else 'kinectC'),
        'resolution': [640, 480] if camera == 0 else [1280, 720],
        'result': result,
        'scope': 'camera_only; joint rig calibration is not replaced',
    }
