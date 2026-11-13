"""Tests for tools/cc + tools/td/projective_profiles.py - synthetic only."""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, os.path.join(TOOLS_DIR, "hg"))
sys.path.insert(0, os.path.join(TOOLS_DIR, "td"))
sys.path.insert(0, HERE)

import hg                                # noqa: E402
import hg_geometry                       # noqa: E402
import projective_profiles as profiles   # noqa: E402
import cc                                # noqa: E402
from test_hg import (_fake_td_rpc, FakeOp, _FakePar, _FakeParMode,  # noqa: E402
                    _gate_eval, FakeTdClient)


def _P_doc(camera=0):
    src = {"id": str(camera), "type": "camera",
           "resolution": [640, 480] if camera == 0 else [1280, 720],
           "top_path": "/project1/camLogi" if camera == 0
           else "/project1/kinectC",
           "capture_backend": "touchdesigner"}
    if camera == 1:
        src["mirrorimage"] = False
        src["native_color_mapping"] = {
            "model": "full_frame_resize",
            "native_resolution": [1920, 1080],
            "td_resolution": [1280, 720]}
    return {"schema": "ff.hg", "schema_version": 1,
            "kind": "homography", "source": src,
            "destination": {"id": "P0", "type": "projector",
                            "resolution": [1280, 720]},
            "source_space": "raw_distorted_pixels",
            "lens_policy":
                "planar_approximation_no_distortion_correction",
            "H_source_to_destination": np.eye(3).tolist()}


def _model(root, sid="Table_01", camera=0, **over):
    """Accepted model bound to the current profile in root."""
    prof = profiles.load_profile(root, sid, camera)
    m = {"schema": "ff.hg.table_color", "schema_version": 1,
         "method": "global_attenuation_gray_balance",
         "source_id": str(camera),
         "source_top": "/project1/camLogi" if camera == 0
         else "/project1/kinectC",
         "destination_id": "P0", "surface_id": sid,
         "source_size": list(
             prof["P"]["source"]["resolution"]),
         "destination_size": [1280, 720],
         "calibration_sha256": prof["geometry_sha256"],
         "gain_rgb": [0.85, 0.9, 1.0], "headroom": 0.62,
         "accepted": True}
    m.update(over)
    return m


class SurfaceIds(unittest.TestCase):
    def test_valid(self):
        for sid in ("Table_01", "a", "A9_-z", "0wall"):
            self.assertEqual(sid, profiles.validate_surface_id(sid))

    def test_rejected(self):
        for bad in ("../x", "a/b", "a\\b", "a.b", ".h", "Table 01",
                    "CON", "prn", "COM3", "lpt9", "ünïcode", "",
                    "a" * 65, "-lead", "_lead", None, 12):
            with self.assertRaises(profiles.ProfileError, msg=repr(bad)):
                profiles.validate_surface_id(bad)

    def test_case_collision_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "tools", "pipe", "p0",
                                     "table_01"))
            with self.assertRaises(profiles.ProfileError):
                profiles.profile_dir(tmp, "Table_01", 0)


class ProfileStore(unittest.TestCase):
    def test_publish_geometry_archives_and_invalidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            P1 = _P_doc()
            prof1, path = profiles.publish_geometry(tmp, P1,
                                                    "Table_01", 0)
            self.assertTrue(os.path.isfile(path))
            self.assertIsNone(prof1["TC"])
            self.assertIsNone(prof1["photometric_revision"])
            model = _model(tmp)
            prof2, _ = profiles.publish_color(tmp, model,
                                              "Table_01", 0)
            self.assertEqual(prof1["geometry_revision"],
                             prof2["geometry_revision"])
            self.assertNotEqual(prof1["revision"], prof2["revision"])
            self.assertIsNotNone(prof2["photometric_revision"])
            self.assertEqual(model["gain_rgb"],
                             prof2["TC"]["gain_rgb"])
            # EVERY revision is archived under its own uuid, starting
            # with the first publish
            rev_dir = os.path.join(os.path.dirname(path), "revisions")
            for prof in (prof1, prof2):
                arch = os.path.join(rev_dir, prof["revision"],
                                    "profile.json")
                with open(arch) as fh:
                    self.assertEqual(prof, json.load(fh))
            # new geometry publish archives the old profile verbatim and
            # invalidates TC - stale gains are never reused on a new H
            P2 = dict(P1)
            P2["H_source_to_destination"] = (
                np.eye(3) * 1.01).tolist()
            prof3, _ = profiles.publish_geometry(tmp, P2,
                                                 "Table_01", 0)
            self.assertIsNone(prof3["TC"])
            self.assertIsNone(prof3["photometric_revision"])
            self.assertNotEqual(prof2["geometry_revision"],
                                prof3["geometry_revision"])
            arch = os.path.join(rev_dir, prof3["revision"],
                                "profile.json")
            with open(arch) as fh:
                self.assertEqual(prof3, json.load(fh))
            # a non-uuid old revision is never used as an archive path
            bad_dir = os.path.join(tmp, "bad", "revisions")
            with self.assertRaises(profiles.ProfileError):
                profiles._archive_current(bad_dir,
                                          {"revision": "../escape"})

    def test_publish_color_binding_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiles.publish_geometry(tmp, _P_doc(), "Table_01", 0)
            for over in ({"schema": "x"},
                         {"method": "other"},
                         {"source_id": "1"},
                         {"source_top": "/project1/kinectC"},
                         {"source_size": [320, 240]},
                         {"destination_size": [640, 480]},
                         {"surface_id": "Other"},
                         {"calibration_sha256": "0" * 64},
                         {"accepted": False},
                         {"gain_rgb": [1.2, 0.9, 1.0]},
                         {"gain_rgb": [float("nan"), 0.9, 1.0]},
                         {"headroom": 0.0},
                         {"headroom": 1.5}):
                m = _model(tmp, **over)
                with self.assertRaises(profiles.ProfileError,
                                       msg=repr(over)):
                    profiles.publish_color(tmp, m, "Table_01", 0)
            # no profile at all
            with self.assertRaises(profiles.ProfileError) as cm:
                profiles.publish_color(tmp, {}, "Nowhere", 0)
            self.assertIn("run hg -c", str(cm.exception))

    def test_migrate_legacy_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiles.publish_geometry(tmp, _P_doc(), "Table_01", 0)
            prof = profiles.load_profile(tmp, "Table_01", 0)
            legacy = _model(tmp)
            legacy.pop("destination_id")
            legacy.pop("surface_id")
            out, _ = profiles.publish_color(
                tmp, legacy, "Table_01", 0, allow_migrate=True,
                provenance="C:/abs/legacy.json")
            tc = out["TC"]
            self.assertEqual("Table_01", tc["surface_id"])
            self.assertEqual("P0", tc["destination_id"])
            self.assertEqual("C:/abs/legacy.json",
                             tc["migration"]["provenance"])
            # conflicting metadata refused, not silently re-associated
            legacy["surface_id"] = "Elsewhere"
            with self.assertRaises(profiles.ProfileError):
                profiles.publish_color(tmp, legacy, "Table_01", 0,
                                       allow_migrate=True)


class CcCli(unittest.TestCase):
    def _mode(self, argv):
        return cc.validate(cc.build_parser().parse_args(argv))

    def test_valid_combos(self):
        self.assertEqual("measure", self._mode(
            ["-c", "0", "-p", "0", "-s_id", "Table_01"]))
        self.assertEqual("measure", self._mode(["-c", "1", "-p", "0"]))
        self.assertEqual("load", self._mode(
            ["--load", "m.json", "-c", "0", "-p", "0"]))
        self.assertEqual("list", self._mode(["--list"]))

    def test_rejected_combos(self):
        with self.assertRaises(cc.CcError):
            self._mode(["-c", "2", "-p", "0"])      # depth refused
        for argv in ([], ["-c", "0"], ["-c", "3", "-p", "0"],
                     ["-c", "0", "-p", "1"],
                     ["-c", "0", "-p", "0", "-s_id", "../x"]):
            with self.assertRaises(cc.CcError, msg=argv):
                self._mode(argv)


class CcLoadFlow(unittest.TestCase):
    """cc --load binds a reviewed model and registers it; failures keep
    honest artifact state instead of claiming success."""

    def test_load_registers_bundle(self):
        client = FakeTdClient(frame=np.zeros((2, 2, 3), np.uint8))
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(cc, "ROOT", tmp):
                profiles.publish_geometry(tmp, _P_doc(), "Table_01", 0)
                model = _model(tmp)
                mpath = os.path.join(tmp, "model.json")
                with open(mpath, "w") as fh:
                    json.dump(model, fh)
                cc.run_load_path(mpath, 0, "Table_01", client=client)
            prof = profiles.load_profile(tmp, "Table_01", 0)
            self.assertEqual(model["gain_rgb"], prof["TC"]["gain_rgb"])
            reg = [c for c in client.rpc_calls
                   if c.get("action") == "register"][0]
            self.assertEqual(prof["geometry_sha256"],
                             reg["profile"]["geometry_sha256"])
            self.assertEqual(prof, reg["profile"])

    def test_register_failure_keeps_file(self):
        client = FakeTdClient(frame=np.zeros((2, 2, 3), np.uint8))
        real = client.rpc

        def flaky(p):
            if p.get("action") == "register":
                raise hg.HgError("TD: dead")
            return real(p)
        client.rpc = flaky
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(cc, "ROOT", tmp):
                profiles.publish_geometry(tmp, _P_doc(), "Table_01", 0)
                model = _model(tmp)
                mpath = os.path.join(tmp, "model.json")
                with open(mpath, "w") as fh:
                    json.dump(model, fh)
                with self.assertRaises(cc.CcError) as cm:
                    cc.run_load_path(mpath, 0, "Table_01",
                                     client=client)
                self.assertIn("TD load failed", str(cm.exception))
                # valid profile stays on disk
                prof = profiles.load_profile(tmp, "Table_01", 0)
                self.assertEqual(model["gain_rgb"],
                                 prof["TC"]["gain_rgb"])


class CcMeasureFlow(unittest.TestCase):
    """run_measure wires staging + publish + register around the
    authored calibrate(); the math itself is not exercised here."""

    def test_measure_publishes_accepted_model(self):
        client = FakeTdClient(frame=np.zeros((2, 2, 3), np.uint8))
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(cc, "ROOT", tmp):
                profiles.publish_geometry(tmp, _P_doc(), "Table_01", 0)
                mdir = os.path.join(tmp, "meas")
                fake = mock.Mock()
                captured = {}

                def _cal(directory, calibration=None, camera_id=0,
                         surface_id="Table_01", client=None):
                    # staging is a readable copy OUTSIDE the measurement
                    # directory (Measurement owns mkdir of that dir)
                    self.assertTrue(os.path.isfile(calibration))
                    self.assertFalse(os.path.abspath(calibration)
                                     .startswith(os.path.abspath(
                                         directory) + os.sep))
                    captured["calibration"] = calibration
                    return _model(tmp)
                fake.calibrate.side_effect = _cal
                args = cc.build_parser().parse_args(
                    ["-c", "0", "-p", "0", "-o", mdir])
                with mock.patch.dict(sys.modules,
                                     {"measurement": fake}):
                    cc.run_measure(args, client=client)
                self.assertFalse(os.path.exists(
                    captured["calibration"]))     # tempdir cleaned up
                prof = profiles.load_profile(tmp, "Table_01", 0)
                self.assertIsNotNone(prof["TC"])
                self.assertTrue(
                    [c for c in client.rpc_calls
                     if c.get("action") == "register"])

    def test_failed_measure_never_replaces_profile(self):
        client = FakeTdClient(frame=np.zeros((2, 2, 3), np.uint8))
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(cc, "ROOT", tmp):
                profiles.publish_geometry(tmp, _P_doc(), "Table_01", 0)
                before = profiles.load_profile(tmp, "Table_01", 0)
                fake = mock.Mock()
                fake.calibrate.side_effect = ValueError("rejected")
                args = cc.build_parser().parse_args(
                    ["-c", "0", "-p", "0", "-o",
                     os.path.join(tmp, "meas")])
                with mock.patch.dict(sys.modules,
                                     {"measurement": fake}):
                    with self.assertRaises(cc.CcError) as cm:
                        cc.run_measure(args, client=client)
                self.assertIn("profile unchanged", str(cm.exception))
                self.assertEqual(before, profiles.load_profile(
                    tmp, "Table_01", 0))
                self.assertFalse(
                    [c for c in client.rpc_calls
                     if c.get("action") == "register"])

    def test_measure_real_constructor_seam(self):
        """run_measure through the REAL Measurement constructor: it owns
        the fresh output dir (mkdir exist_ok=False) while the staging
        calibration stays outside and readable. Only show() is stubbed
        with an authored numeric camera response - no acceptance math
        is reimplemented here."""
        import measurement
        client = FakeTdClient(
            frame=np.zeros((480, 640, 3), np.uint8),
            inspect_info={
                "camLogi": {"exists": True, "width": 640,
                            "height": 480},
                "protocol_version": hg.PROTOCOL_VERSION})
        g_true = np.array([1.0, 0.9, 0.8])

        def fake_show(self, rgb, label, settle=1.2, frame_count=8):
            cam = np.asarray(rgb, float) * g_true
            self.records.append({
                "index": len(self.records), "label": label,
                "projector_rgb": list(rgb),
                "camera_rgb": cam.tolist(),
                "clipped_fraction_rgb": [0.0, 0.0, 0.0]})
            self.publish()
            return cam

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(cc, "ROOT", tmp):
                profiles.publish_geometry(tmp, _P_doc(), "Table_01", 0)
                mdir = os.path.join(tmp, "meas_new")
                args = cc.build_parser().parse_args(
                    ["-c", "0", "-p", "0", "-o", mdir])
                self.assertFalse(os.path.exists(mdir))
                with mock.patch.object(measurement.Measurement,
                                       "show", fake_show), \
                        mock.patch.object(hg, "ROOT", tmp):
                    cc.run_measure(args, client=client)
            # Measurement created the directory itself
            self.assertTrue(os.path.isdir(mdir))
            self.assertTrue(os.path.isfile(
                os.path.join(mdir, "model.json")))
            self.assertTrue(os.path.isfile(
                os.path.join(mdir, "measurements.json")))
            prof = profiles.load_profile(tmp, "Table_01", 0)
            self.assertIsNotNone(prof["TC"])
            self.assertTrue(
                [c for c in client.rpc_calls
                 if c.get("action") == "register"])
            # restore ran through the real __exit__ -> restored flag
            with open(os.path.join(mdir, "measurements.json")) as fh:
                self.assertTrue(json.load(fh).get("restored"))

    def test_missing_profile_hint(self):
        client = FakeTdClient(frame=np.zeros((2, 2, 3), np.uint8))
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(cc, "ROOT", tmp):
                args = cc.build_parser().parse_args(["-c", "0", "-p", "0"])
                with self.assertRaises(cc.CcError) as cm:
                    cc.run_measure(args, client=client)
                self.assertIn("run .\\hg -c 0 -p 0", str(cm.exception))


class TdColorFlow(unittest.TestCase):
    """TD-side: registry entries apply through activate/resync."""

    MODEL = {"schema": "ff.hg.table_color", "schema_version": 1,
             "method": "global_attenuation_gray_balance",
             "source_id": "0", "source_top": "/project1/camLogi",
             "destination_id": "P0", "surface_id": "Table_01",
             "source_size": [640, 480],
             "destination_size": [1280, 720],
             "gain_rgb": [0.8, 0.9, 1.0], "headroom": 0.5,
             "accepted": True}

    def setUp(self):
        self.rpc, self.reg = _fake_td_rpc()
        self.assertTrue(self.rpc({"action": "setup"})["ok"])

    def _profile(self, tc=None, camera=0, surface="Table_01"):
        P = _P_doc(camera)
        import uuid
        sha = hashlib.sha256(json.dumps(
            P, sort_keys=True).encode()).hexdigest()
        if tc is not None:
            tc = dict(tc)
            tc.setdefault("calibration_sha256", sha)
        return {"schema": "ff.projective.profile",
                "schema_version": 1, "camera_id": camera,
                "projector_id": 0, "surface_id": surface,
                "revision": uuid.uuid4().hex,
                "geometry_revision": uuid.uuid4().hex,
                "photometric_revision": uuid.uuid4().hex if tc else None,
                "geometry_sha256": sha,
                "P": P, "TC": tc}

    def _register(self, tc=None, **kw):
        prof = kw.pop("profile", None) or self._profile(tc=tc, **kw)
        return self.rpc({"action": "register", "profile": prof,
                         "warp_params": hg_geometry.warp_parameters(
                             prof["P"])})

    def test_bad_tc_refused_before_mutation(self):
        for over in ({"schema": "x"}, {"method": "other"},
                     {"source_id": "2"},
                     {"source_top": "/project1/kinectC"},
                     {"gain_rgb": [0.5]},
                     {"gain_rgb": [0.5, float("nan"), 0.5]},
                     {"gain_rgb": [1.1, 0.5, 0.5]},
                     {"headroom": 0.0}, {"headroom": 2.0},
                     {"accepted": False},
                     {"accepted": False,
                      "validation": {"accepted": True}}):
            m = dict(self.MODEL); m.update(over)
            res = self._register(tc=m)
            self.assertFalse(res["ok"], over)
        self.assertEqual({}, self.reg["/project1/hg"].storage.get(
            "hg_profiles", {}))

    def test_gain_headroom_dynamic_not_hardcoded(self):
        # composite uniforms come from the model fields, not constants
        for hr, gains in ((0.5, [0.8, 0.9, 1.0]),
                          (0.75, [1.0, 0.5, 0.25])):
            m = dict(self.MODEL)
            m.update(headroom=hr, gain_rgb=gains)
            self.assertTrue(self._register(tc=m)["ok"])
            res = self.rpc({"action": "activate", "camera_id": 0,
                            "surface_id": "Table_01", "select": False})
            self.assertTrue(res["ok"], res)
            color = self.reg["/project1/hg_table_color"]
            for a, g in zip("xyz", gains):
                self.assertAlmostEqual(
                    g * hr, color.par["vec0value%s" % a].eval())

    def test_activate_c1_profile_and_c1_tc(self):
        tc1 = dict(self.MODEL)
        tc1.update(source_id="1", source_top="/project1/kinectC",
                   source_size=[1280, 720])
        self.assertTrue(self._register(tc=tc1, camera=1)["ok"])
        res = self.rpc({"action": "activate", "camera_id": 1,
                        "surface_id": "Table_01", "select": False})
        self.assertTrue(res["ok"], res)
        self.assertEqual([1, "Table_01"], res["ready"])
        cs = self.reg["/project1/cam_and_kinect"]
        self.assertEqual(1, cs.par.index.eval())
        self.assertEqual(1, _gate_eval(
            self.reg["/project1/hg_output"], self.reg))
        # c1 gains never leak onto c0: switching camera drops to black
        cs.par.index.val = 0
        self.rpc({"action": "resync"})
        self.assertIsNone(
            self.reg["/project1/hg"].storage.get("hg_ready_pair"))

    def test_controls_preserved_across_reapply(self):
        self.assertTrue(self._register(tc=dict(self.MODEL))["ok"])
        self.assertTrue(self.rpc({"action": "activate",
                                  "camera_id": 0,
                                  "surface_id": "Table_01",
                                  "select": False})["ok"])
        ctl = self.reg["/project1/hg_table_color_controls"]
        ctl.par.Enabled.val = False
        ctl.par.Strength.val = 0.4
        self.rpc({"action": "resync"})   # same pair: no reapply needed
        # force reapply through a camera flip
        self.reg["/project1/cam_and_kinect"].par.index.val = 1
        self.rpc({"action": "resync"})
        self.reg["/project1/cam_and_kinect"].par.index.val = 0
        self.rpc({"action": "resync"})
        self.assertFalse(ctl.par.Enabled.eval())
        self.assertEqual(0.4, ctl.par.Strength.eval())

    def test_partial_controls_repaired(self):
        ctl = FakeOp("/project1/hg_table_color_controls", "baseCOMP")
        ctl.storage["hg_owned"] = True
        ctl.par.Enabled = False
        self.assertTrue(self._register(tc=dict(self.MODEL))["ok"])
        self.assertTrue(self.rpc({"action": "activate",
                                  "camera_id": 0,
                                  "surface_id": "Table_01",
                                  "select": False})["ok"])
        self.assertFalse(ctl.par.Enabled.eval())
        self.assertEqual(1.0, ctl.par.Strength.eval())


class MeasurementWiring(unittest.TestCase):
    """The authored measurement module is exercised only at its seams:
    begin payload, shared-switch capture path, client plumbing."""

    def test_show_uses_shared_switch_and_pair_payload(self):
        import measurement
        with tempfile.TemporaryDirectory() as tmp:
            P = _P_doc()
            cal = os.path.join(tmp, "cal.json")
            with open(cal, "w") as fh:
                json.dump(P, fh)
            mdir = os.path.join(tmp, "m")
            client = FakeTdClient(
                frame=np.zeros((480, 640, 3), np.uint8),
                inspect_info={
                    "camLogi": {"exists": True, "width": 640,
                                "height": 480},
                    "protocol_version": hg.PROTOCOL_VERSION})
            with mock.patch.object(hg, "ROOT", tmp):
                session = measurement.Measurement(
                    mdir, calibration=cal, camera_id=0,
                    surface_id="Table_01", client=client)
            session.show([64, 64, 64], "t", settle=0, frame_count=1)
            begin = [c for c in client.rpc_calls
                     if c.get("action") == "begin"][0]
            self.assertEqual(0, begin["camera_id"])
            self.assertEqual("Table_01", begin["surface_id"])
            # capture went through the shared selector, not the raw TOP
            self.assertEqual("/project1/cam_and_kinect",
                             session.client.last_top
                             if hasattr(session.client, "last_top")
                             else client.captured_tops[-1])


class PipeMigration(unittest.TestCase):
    """out/calibration -> tools/pipe store migration: byte-exact copies,
    idempotent, refuses conflicts and malformed contract paths."""

    def _stage_legacy(self, tmp):
        """Publish real profiles into a scratch repo, then relocate the
        pipe tree to the legacy path (bytes untouched)."""
        import shutil
        staging = os.path.join(tmp, "stage")
        p1, _ = profiles.publish_geometry(staging, _P_doc(0),
                                          "Table_01", 0)
        model = _model(staging)
        profiles.publish_color(staging, model, "Table_01", 0)
        profiles.publish_geometry(staging, _P_doc(1), "Wall", 1)
        src = profiles.calibration_root(staging)
        legacy = profiles.legacy_calibration_root(tmp)
        os.makedirs(os.path.dirname(legacy), exist_ok=True)
        shutil.copytree(src, legacy)
        return p1, legacy

    def test_migrate_byte_exact_and_idempotent(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            self._stage_legacy(tmp)
            legacy = profiles.legacy_calibration_root(tmp)
            rep = profiles.migrate_legacy_profiles(tmp)
            self.assertFalse(rep["refused"])
            self.assertFalse(rep["ignored"])
            copied = [r.replace(os.sep, "/") for r in rep["copied"]]
            # current + revision archives for BOTH surfaces
            self.assertIn("Table_01/c0/profile.json", copied)
            self.assertIn("Wall/c1/profile.json", copied)
            revs = [r for r in copied if "/revisions/" in r]
            # Table_01/c0: geometry + TC revisions; Wall/c1: geometry
            self.assertEqual(3, len(revs))
            self.assertEqual(
                2, len([r for r in revs
                        if r.startswith("Table_01/c0/")]))
            # every file byte-exact, legacy untouched
            for dirpath, _d, files in os.walk(legacy):
                for name in files:
                    src = os.path.join(dirpath, name)
                    rel = os.path.relpath(src, legacy)
                    with open(src, "rb") as a, open(os.path.join(
                            profiles.calibration_root(tmp), rel),
                            "rb") as b:
                        self.assertEqual(a.read(), b.read(), rel)
            self.assertTrue(os.path.isdir(legacy))
            # second run: everything skipped, nothing recopied
            rep2 = profiles.migrate_legacy_profiles(tmp)
            self.assertFalse(rep2["copied"])
            self.assertFalse(rep2["refused"])
            self.assertEqual(sorted(rep["copied"]),
                             sorted(rep2["skipped"]))

    def test_conflict_refused_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._stage_legacy(tmp)
            dst = os.path.join(
                profiles.calibration_root(tmp),
                "Table_01", "c0", "profile.json")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            conflict = b'{"different": true}' + b"\n"
            with open(dst, "wb") as fh:
                fh.write(conflict)
            rep = profiles.migrate_legacy_profiles(tmp)
            refused = [r.replace(os.sep, "/") for r in rep["refused"]]
            self.assertIn("Table_01/c0/profile.json", refused)
            with open(dst, "rb") as fh:
                self.assertEqual(conflict, fh.read())
            # the other surface still migrated
            self.assertIn("Wall/c1/profile.json",
                          [r.replace(os.sep, "/")
                           for r in rep["copied"]])

    def test_missing_legacy_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            rep = profiles.migrate_legacy_profiles(tmp)
            self.assertEqual(
                {"copied": [], "skipped": [], "refused": [],
                 "ignored": []}, rep)

    def test_malformed_paths_refused_strays_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy = profiles.legacy_calibration_root(tmp)
            bad_sid = os.path.join(legacy, "bad.sid", "c0",
                                   "profile.json")
            bad_rev = os.path.join(legacy, "Table_01", "c0",
                                   "revisions", "not-a-uuid",
                                   "profile.json")
            stray = os.path.join(legacy, "Table_01", "c0",
                                 "notes.txt")
            for path in (bad_sid, bad_rev, stray):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as fh:
                    fh.write("{}\n" if path.endswith("json")
                             else "x\n")
            rep = profiles.migrate_legacy_profiles(tmp)
            refused = [r.replace(os.sep, "/") for r in rep["refused"]]
            self.assertIn("bad.sid/c0/profile.json", refused)
            self.assertIn(
                "Table_01/c0/revisions/not-a-uuid/profile.json",
                refused)
            self.assertIn(
                "Table_01/c0/notes.txt",
                [r.replace(os.sep, "/") for r in rep["ignored"]])
            # nothing written under the new root
            self.assertFalse(os.path.exists(
                profiles.calibration_root(tmp)))


if __name__ == "__main__":
    unittest.main()
