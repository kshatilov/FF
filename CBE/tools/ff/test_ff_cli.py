"""Narrow ff CLI / IO / bridge tests - all seams mocked, no TD/SDK hardware."""
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'hg'))

import ff_calibrate
import ff_calibration_io
import ff_cli


class CliParse(unittest.TestCase):
    """Parser + dispatch: no TD/SDK calls for --help or invalid args."""

    def setUp(self):
        self.rig = mock.patch.object(ff_calibrate, 'run_rig').start()
        self.cam = mock.patch.object(ff_calibrate, 'calibrate_camera').start()
        self.session = mock.patch.object(
            ff_calibration_io, 'OutputSession').start()
        self.session.return_value.__enter__ = mock.Mock(return_value='sess')
        self.session.return_value.__exit__ = mock.Mock(return_value=False)
        self.publish = mock.patch.object(
            ff_calibration_io, 'publish_documents').start()
        self.record = mock.patch.object(
            ff_calibration_io, 'camera_record').start()
        self.addCleanup(mock.patch.stopall)
        self.cam.return_value = {'K': 1}
        self.record.side_effect = lambda m, c, r: {'camera_id': c, 'result': r}

    def test_default_runs_rig_kinect_once_inside_session(self):
        self.assertEqual(ff_cli.main(['calib']), 0)
        self.rig.assert_called_once_with(kinect=True)
        self.assertTrue(self.session.called)
        self.cam.assert_not_called()
        self.publish.assert_not_called()

    def test_no_kinect_runs_legacy_rig(self):
        self.assertEqual(ff_cli.main(['calib', '--no-kinect']), 0)
        self.rig.assert_called_once_with(kinect=False)

    def test_camera_only_writes_only_camera_record(self):
        self.assertEqual(ff_cli.main(['calib', '-c_i', '0']), 0)
        self.cam.assert_called_once_with(0)
        self.rig.assert_not_called()
        self.publish.assert_called_once()
        docs = self.publish.call_args[0][0]
        self.assertEqual(list(docs), ['camera_c0.json'])
        self.assertEqual(docs['camera_c0.json']['camera_id'], 0)

    def test_camera_alias_forms(self):
        for argv in (['calib', '-c_1'], ['calib', '-c', '1'],
                     ['calib', '--camera', '1']):
            self.cam.reset_mock()
            self.assertEqual(ff_cli.main(argv), 0, argv)
            self.cam.assert_called_once_with(1)

    def test_camera2_publishes_raw_factory_record(self):
        self.assertEqual(ff_cli.main(['calib', '-c_2']), 0)
        self.cam.assert_called_once_with(2)
        self.record.assert_not_called()          # c2 uses the record verbatim
        docs = self.publish.call_args[0][0]
        self.assertEqual(docs['camera_c2.json'], {'K': 1})

    def test_mutually_exclusive_flags_no_hardware(self):
        for argv in (['calib', '-c_i', '0', '--no-kinect'],
                     ['calib', '-c_i', '0', '--restore']):
            with self.assertRaises(SystemExit):
                ff_cli.main(argv)
        self.rig.assert_not_called()
        self.cam.assert_not_called()
        self.session.assert_not_called()

    def test_invalid_camera_id_no_hardware(self):
        with self.assertRaises(SystemExit):
            ff_cli.main(['calib', '-c_i', '7'])
        self.rig.assert_not_called()

    def test_help_never_imports_calibration(self):
        sys.modules.pop('ff_calibrate', None)
        try:
            with self.assertRaises(SystemExit):
                ff_cli.main(['--help'])
            self.assertNotIn('ff_calibrate', sys.modules)
        finally:
            sys.modules['ff_calibrate'] = ff_calibrate

    def test_rig_failure_returns_nonzero_inside_session(self):
        self.rig.side_effect = RuntimeError('boom')
        self.assertEqual(ff_cli.main(['calib']), 1)
        self.session.return_value.__exit__.assert_called()


class Restore(unittest.TestCase):
    def test_restore_replays_snapshot(self):
        snapshot = {'owner': 'o1', 'parameters': [
            {'node': '/project1/final_switch', 'name': 'index',
             'value': 0, 'expr': None, 'mode': 'CONSTANT'}]}
        with mock.patch.object(ff_calibration_io, 'evaluate_json',
                               return_value=snapshot), \
                mock.patch.object(ff_calibration_io, 'restore_output',
                                  return_value=None) as ro:
            self.assertEqual(ff_cli.main(['calib', '--restore']), 0)
        ro.assert_called_once()

    def test_restore_without_session_fails(self):
        with mock.patch.object(ff_calibration_io, 'evaluate_json',
                               return_value=None):
            self.assertEqual(ff_cli.main(['calib', '--restore']), 1)


class DepthFactory(unittest.TestCase):
    """c2: SDK factory intrinsics only, no FF pose claim, handle closed."""

    FIELDS = dict(FocalLengthX=365.0, FocalLengthY=365.5,
                  PrincipalPointX=256.0, PrincipalPointY=210.0,
                  RadialDistortionSecondOrder=0.1,
                  RadialDistortionFourthOrder=-0.2,
                  RadialDistortionSixthOrder=0.05)

    def _sensor(self, fields=None, fail=None):
        sensor = mock.Mock()
        intr = types.SimpleNamespace(**(fields or self.FIELDS))
        if fail:
            sensor.mapper.GetDepthCameraIntrinsics.side_effect = fail
        else:
            sensor.mapper.GetDepthCameraIntrinsics.return_value = intr
        return sensor

    def test_exact_K_no_pose(self):
        doc = ff_calibrate.depth_factory_intrinsics(self._sensor())
        self.assertEqual(doc['camera_id'], 2)
        self.assertFalse(doc['pose_calibrated'])
        self.assertEqual(doc['method'], 'kinect_v2_sdk_factory_intrinsics')
        self.assertEqual(doc['K'][0][0], 365.0)
        self.assertEqual(doc['K'][1][1], 365.5)
        self.assertEqual(doc['K'][0][2], 256.0)
        self.assertEqual(doc['K'][1][2], 210.0)
        self.assertEqual(doc['resolution'], [512, 424])

    def test_owned_sensor_closed_on_failure(self):
        sensor = self._sensor(fail=RuntimeError('sdk dead'))
        fake = types.ModuleType('kincap')
        fake.Kin = lambda: sensor
        with mock.patch.dict(sys.modules, {'kincap': fake}):
            with self.assertRaises(RuntimeError):
                ff_calibrate.depth_factory_intrinsics()
        sensor._sensor.Close.assert_called_once()

    def test_invalid_intrinsics_refused_and_closed(self):
        bad = dict(self.FIELDS, FocalLengthX=0.0)
        sensor = self._sensor(fields=bad)
        fake = types.ModuleType('kincap')
        fake.Kin = lambda: sensor
        with mock.patch.dict(sys.modules, {'kincap': fake}):
            with self.assertRaises(RuntimeError):
                ff_calibrate.depth_factory_intrinsics()
        sensor._sensor.Close.assert_called_once()


class PublishDocuments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pipe = Path(self.tmp.name) / 'ff'
        self.pipe.mkdir()
        p = mock.patch.object(ff_calibration_io, 'PIPE', self.pipe)
        p.start()
        self.addCleanup(p.stop)

    def test_publish_writes_and_archives_previous(self):
        target = self.pipe / 'camera_c0.json'
        target.write_bytes(b'{"old":1}\n')
        rev = ff_calibration_io.publish_documents(
            {'camera_c0.json': {'new': 2}})
        self.assertEqual(json.loads(target.read_text()), {'new': 2})
        archived = self.pipe / 'revisions' / rev / 'camera_c0.json'
        self.assertEqual(archived.read_bytes(), b'{"old":1}\n')

    def test_mid_publish_failure_rolls_back_all_bytes(self):
        a = self.pipe / 'a.json'
        a.write_bytes(b'{"old":1}\n')
        docs = {'a.json': {'n': 1}, 'b.json': {'n': 2}}
        real_replace = os.replace
        calls = []

        def flaky(src, dst):
            calls.append(dst)
            if len(calls) == 2:
                raise OSError('injected replace failure')
            return real_replace(src, dst)

        with mock.patch.object(ff_calibration_io.os, 'replace', flaky):
            with self.assertRaises(OSError):
                ff_calibration_io.publish_documents(docs)
        self.assertEqual(a.read_bytes(), b'{"old":1}\n')
        self.assertFalse((self.pipe / 'b.json').exists())
        revs = list((self.pipe / 'revisions').iterdir())
        self.assertTrue(revs)
        self.assertEqual((revs[0] / 'a.json').read_bytes(), b'{"old":1}\n')
        # staged temp files cleaned
        self.assertFalse(list(self.pipe.glob('*.pending')))
        self.assertFalse(list(self.pipe.glob('*.rollback')))

    def test_first_publish_no_archive_needed(self):
        rev = ff_calibration_io.publish_documents({'x.json': {'v': 1}})
        self.assertEqual(json.loads((self.pipe / 'x.json').read_text()),
                         {'v': 1})


class OutputSessionTests(unittest.TestCase):
    """Snapshot/restore against a fake module.td - no TD process."""

    def _module(self, state):
        module = mock.Mock()
        state = dict(state)
        stored = {}

        def td(code, mode='--exec'):
            if 'storage.get' in code and 'hg_busy' not in code:
                return json.dumps(stored.get('snapshot'))
            if 'dumps' in code:
                return json.dumps(state)
            if '.store(' in code:
                import ast as _ast
                stored['snapshot'] = _ast.literal_eval(
                    code.split(',', 1)[1].rsplit(')', 1)[0])
            module.executed.append(code)
            return ''
        module.executed = []
        module.td.side_effect = td
        return module

    STATE = {
        'hg_busy': False, 'ff_busy': False,
        'parameters': [
            {'node': '/project1/final_switch', 'name': 'index',
             'value': 7, 'expr': "op('/project1/output_mode').par['Mode']"
                                ".menuIndex", 'mode': 'EXPRESSION'},
            {'node': '/project1/cam_and_kinect', 'name': 'index',
             'value': 0, 'expr': None, 'mode': 'CONSTANT'},
            {'node': '/project1/output_mode', 'name': 'Mode',
             'value': 'ffv2', 'expr': None, 'mode': 'CONSTANT'},
        ]}

    def test_enter_stores_owner_exit_restores_every_par(self):
        module = self._module(self.STATE)
        sess = ff_calibration_io.OutputSession(module)
        with sess:
            self.assertTrue(module._ff_session_owner)
        restore = [c for c in module.executed if 'expr=' in c][-1]
        self.assertIn('/project1/final_switch', restore)
        self.assertIn('/project1/cam_and_kinect', restore)
        self.assertIn('/project1/output_mode', restore)
        self.assertIn('unstore', module.executed[-1])

    def test_busy_session_refused(self):
        module = self._module(dict(self.STATE, ff_busy=True))
        with self.assertRaises(RuntimeError):
            ff_calibration_io.OutputSession(module).__enter__()

    def test_hg_session_refused(self):
        module = self._module(dict(self.STATE, hg_busy=True))
        with self.assertRaises(RuntimeError):
            ff_calibration_io.OutputSession(module).__enter__()

    def test_missing_final_switch_refused(self):
        state = dict(self.STATE, parameters=[])
        module = self._module(state)
        with self.assertRaises(RuntimeError):
            ff_calibration_io.OutputSession(module).__enter__()

    def test_restore_refuses_foreign_owner(self):
        module = self._module(self.STATE)
        sess = ff_calibration_io.OutputSession(module)
        sess.__enter__()
        foreign = {'owner': 'other', 'parameters': []}
        with mock.patch.object(ff_calibration_io, 'evaluate_json',
                               return_value=foreign):
            with self.assertRaises(RuntimeError):
                ff_calibration_io.restore_output(module, sess.snapshot)

    def test_restore_error_mentions_restore_flag(self):
        module = self._module(self.STATE)
        sess = ff_calibration_io.OutputSession(module)
        sess.__enter__()
        with mock.patch.object(ff_calibration_io, 'restore_output',
                               side_effect=RuntimeError('dead bridge')):
            with self.assertRaises(RuntimeError) as ctx:
                sess.__exit__()
        self.assertIn('--restore', str(ctx.exception))

    def test_evaluate_json_literal_fallback(self):
        module = mock.Mock()
        # wscontrol eval returns a Python repr of the JSON string
        module.td.return_value = repr('{"camera": 0, "final": 7}')
        self.assertEqual(
            ff_calibration_io.evaluate_json(module, 'x'), {'camera': 0,
                                                           'final': 7})


class RunRigSeam(unittest.TestCase):
    """Mocked expensive phases: publish only fresh docs, never on failure."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch
        self.mocks = []
        def M(name, value):
            m = p.object(ff_calibrate, name, value)
            m.start()
            self.mocks.append(m)
            return m
        self.addCleanup(lambda: [m.stop() for m in self.mocks])
        np_ = np
        self.K = np_.eye(3)
        rv = np_.zeros((3, 1)); tv = np_.zeros((3, 1))
        K = np_.eye(3)
        M('_spec', {'frame': {}, 'measured': {}})
        M('OUT', Path(self.tmp.name))
        M('TAG', 'testtag')
        M('WEBCAM_K_SRC', self._write({'cam': {'f_px': 516.0}}))
        M('face_polys', lambda: [])
        self.wfound = {10: np_.zeros((4, 2)), 11: np_.ones((4, 2)),
                       12: np_.ones((4, 2)) * 2}
        M('phase_webcam', lambda *a, **k:
          (rv, tv, 0.5, 1.0, self.wfound, K))
        M('phase_correspondences', lambda *a:
          (np_.zeros((4, 3)), np_.zeros((4, 2)), [0, 0, 0, 0], ['+y'] * 4, []))
        M('_proj_pos_prior', lambda *a: np_.zeros(3))
        errs = np_.zeros(8)
        M('phase_projector', lambda *a:
          (K, rv, tv, errs, np_.ones(8, bool), False))
        dots = [(np_.zeros(3), np_.zeros(2), i, '+y', 1.0) for i in range(6)] + \
               [(np_.zeros(3), np_.zeros(2), i, '+x', 1.0) for i in range(6)]
        M('phase_face_dots', lambda *a, **k:
          (list(dots), [{'n': 12}], np_.zeros((4, 4, 3), np_.uint8)))
        M('detect', lambda *a: {})
        M('solve_fulcrum', lambda *a:
          (object(), dict(self.wfound)))
        errors = np_.array([0.1] * 8)
        fit = (K, rv, tv, K, rv, tv, errors,
               np_.array([1, 1, 1, 1, 0, 0, 0, 0], bool),
               np_.array([1, 1, 0, 0, 1, 1, 0, 0], bool),
               np_.ones(8, bool))
        M('phase_joint_fit', lambda *a, **k: fit)
        M('phase_outline_assert', lambda *a:
          {'coverage_within_4px': 0.95})
        frozen = {'webcam': {'rvec': [0, 0, 0], 'tvec': [0, 0, 0]},
                  'projector': {'f_px': 700, 'cy': 360,
                                'rvec': [0, 0, 0], 'tvec': [0, 0, 100]}}
        M('PROJ_PRIOR', self._write(frozen))
        self.geometry = mock.Mock(return_value={'geo': True})
        M('export_viewer', self.geometry)
        self.publish = mock.Mock()
        M('publish_documents', self.publish)

    def _write(self, doc):
        f = Path(self.tmp.name) / (str(len(os.listdir(self.tmp.name))) + '.json')
        f.write_text(json.dumps(doc))
        return f

    REG = {'K': [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'D': [0] * 5,
           'camera_from_reference': [[1, 0, 0, 0], [0, 1, 0, 0],
                                     [0, 0, 1, 0], [0, 0, 0, 1]],
           'kinect': {'markers': [10, 11, 12]}, 'resolution': [1280, 720],
           'depth_factory': {'camera_id': 2, 'pose_calibrated': False}}

    def test_late_kinect_failure_preserves_all_configs(self):
        m = mock.patch.object(ff_calibrate, 'phase_kinect',
                              side_effect=RuntimeError('SDK down'))
        m.start()
        self.mocks.append(m)
        with self.assertRaises(RuntimeError):
            ff_calibrate.run_rig()
        self.publish.assert_not_called()
        self.geometry.assert_not_called()
        archive = Path(self.tmp.name) / 'testtag_solve.json'
        result = json.loads(archive.read_text())
        self.assertFalse(result['published'])
        self.assertIn('SDK down', result['failure'])

    def test_viewer_gets_fresh_registration_before_publish(self):
        reg = dict(self.REG)
        seen = {}
        def publish(docs):
            seen['docs'] = docs
            seen['geometry_arg'] = self.geometry.call_args[1]['registration']
        self.publish.side_effect = publish
        m = mock.patch.object(ff_calibrate, 'phase_kinect',
                              return_value=reg)
        m.start()
        self.mocks.append(m)
        result = ff_calibrate.run_rig()
        self.assertTrue(result['published'])
        self.assertIs(seen['geometry_arg'], reg)          # fresh object to viewer
        docs = seen['docs']
        self.assertEqual(set(docs),
            {'td_kinect_registration.json', 'camera_c0.json',
             'camera_c1.json', 'camera_c2.json',
             'calibration_current.json', 'calib_geometry.json'})
        self.assertEqual(docs['camera_c0.json']['scope'], 'joint_rig_snapshot')
        self.assertEqual(docs['camera_c1.json']['scope'], 'joint_rig_snapshot')
        self.assertIs(docs['camera_c2.json'], reg['depth_factory'])
        self.assertIs(docs['calib_geometry.json'], self.geometry.return_value)

    def test_no_kinect_skips_registration(self):
        pk = mock.patch.object(ff_calibrate, 'phase_kinect').start()
        self.addCleanup(mock.patch.stopall)
        result = ff_calibrate.run_rig(kinect=False)
        pk.assert_not_called()
        self.assertFalse(result['kinect_included'])
        docs = self.publish.call_args[0][0]
        self.assertNotIn('td_kinect_registration.json', docs)
        self.assertNotIn('camera_c1.json', docs)


# ---------------------------------------------------------------
# ff_td bridge seam: run the TD-side helper under a fake td module.
# ---------------------------------------------------------------

class _FakePar:
    def __init__(self):
        self.expr = None
        self.val = None
        self.mode = None
        self.callbacks = None
        self.index = None


class _FakeOp:
    def __init__(self, path, optype):
        self.path = path
        self.OPType = optype
        self.storage = {}
        self.par = types.SimpleNamespace()
        self.par.callbacks = None
        self.par.index = types.SimpleNamespace(val=None, mode=None)
        self.inputs = []
        self.outputConnectors = [types.SimpleNamespace(
            connect=lambda other: other.inputs.append(self))]
        self.errors_v = ''
        self.text = ''
        self._frame = None

    def store(self, k, v):
        self.storage[k] = v

    def cook(self, force=False):
        pass

    def errors(self):
        return self.errors_v

    def numpyArray(self, delayed=False):
        return self._frame

    def copyNumpyArray(self, arr):
        self.copied = arr


def _load_ff_td(ops):
    ns = {'__file__': str(HERE / 'ff_td.py')}
    def op(path):
        return ops.get(path)
    root = ops.setdefault('/project1', _FakeOp('/project1', 'basecomp'))

    def create(kind, name):
        node = _FakeOp('/project1/' + name, kind.__name__.lower())
        ops[node.path] = node
        return node
    root.create = create
    ns.update({'op': op,
               'textDAT': type('textDAT', (), {}),
               'scriptTOP': type('scriptTOP', (), {}),
               'ParMode': types.SimpleNamespace(CONSTANT='CONSTANT')})
    code = compile((HERE / 'ff_td.py').read_text(), 'ff_td.py', 'exec')
    exec(code, ns)
    return ns, ops


class FfTdBridge(unittest.TestCase):
    def test_unknown_action_and_owner_refusal(self):
        ns, ops = _load_ff_td({})
        rpc = ns['rpc']
        self.assertEqual(rpc({'action': 'nope'})['ok'], False)
        res = rpc({'action': 'pattern', 'owner': 'x',
                   'pattern': {'shape': [720, 1280, 3]}})
        self.assertFalse(res['ok'])
        self.assertIn('owner', res['error'])

    def test_capture_requires_project1_top(self):
        ns, ops = _load_ff_td({})
        res = ns['rpc']({'action': 'capture', 'top': '/elsewhere/x'})
        self.assertFalse(res['ok'])
        cam = _FakeOp('/project1/camLogi', 'videodeviceintop')
        cam._frame = np.zeros((2, 2, 4), np.float32)
        ops['/project1/camLogi'] = cam
        res = ns['rpc']({'action': 'capture', 'top': '/project1/camLogi'})
        self.assertTrue(res['ok'])
        self.assertIn('array', res)

    def test_pattern_creates_owned_top_and_wires_switch(self):
        session_owner = 'own1'
        root = _FakeOp('/project1', 'basecomp')
        root.storage['ff_calibration_session'] = {'owner': session_owner}
        ops = {'/project1': root}
        root.create = lambda kind, name: ops.setdefault(
            '/project1/' + name,
            _FakeOp('/project1/' + name, kind.__name__.lower()))
        switch = _FakeOp('/project1/final_switch', 'switchtop')
        switch.inputs = []
        ops['/project1/final_switch'] = switch
        ns, ops = _load_ff_td(ops)
        import hg_wire
        pattern = hg_wire.encode_array(np.zeros((720, 1280, 3), np.uint8))
        res = ns['rpc']({'action': 'pattern', 'owner': session_owner,
                        'pattern': pattern})
        self.assertTrue(res['ok'], res.get('error'))
        top = ops['/project1/ff_calibration_wire']
        self.assertTrue(top.storage['ff_owned'])
        self.assertIn(top, switch.inputs)
        self.assertEqual(switch.par.index.mode, 'CONSTANT')

    def test_wrong_shape_pattern_refused(self):
        ns, ops = _load_ff_td({})
        ops['/project1'].storage['ff_calibration_session'] = {'owner': 'o'}
        import hg_wire
        pattern = hg_wire.encode_array(np.zeros((4, 4, 3), np.uint8))
        res = ns['rpc']({'action': 'pattern', 'owner': 'o',
                        'pattern': pattern})
        self.assertFalse(res['ok'])
        self.assertIn('1280x720', res['error'])


class TdClientHelper(unittest.TestCase):
    def test_default_and_custom_helper(self):
        import hg
        self.assertEqual(hg.TdClient().helper, hg.TD_HELPER)
        custom = str(HERE / 'ff_td.py')
        self.assertEqual(hg.TdClient(helper=custom).helper, custom)
        with self.assertRaises(hg.HgError):
            hg.TdClient(helper=str(HERE / 'missing.py'))


if __name__ == '__main__':
    unittest.main()
