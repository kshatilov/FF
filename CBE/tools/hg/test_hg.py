"""Tests for tools/hg.py - synthetic only, no hardware/display/TD."""
import ctypes
import enum
import json
import os
import pathlib
import runpy
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(TOOLS_DIR, "td"))
HG_TD = os.path.join(TOOLS_DIR, "td", "projective_td.py")

import hg
import hg_geometry
import hg_wire


def quad(w, h, scale=0.2):
    m = scale
    return np.array([[w * m, h * m], [w * (1 - m), h * m * 1.1],
                     [w * (1 - m), h * (1 - m)], [w * m, h * (1 - m)]])


class GeometryContract(unittest.TestCase):
    def test_perspective_roundtrip_and_uv(self):
        src = np.array([[0, 0], [100, 5], [98, 60], [3, 62], [50, 30],
                        [20, 45]], np.float64)
        H_true = np.array([[1.4, 0.05, 30], [0.02, 1.1, 15],
                           [2e-4, -1e-4, 1.0]])
        dst = hg_geometry.transform_points(H_true, src)
        H, H_inv, diag = hg_geometry.fit_homography(src, dst)
        np.testing.assert_allclose(
            hg_geometry.transform_points(H, src), dst, atol=1e-5)
        np.testing.assert_allclose(
            hg_geometry.transform_points(H_inv, dst), src, atol=1e-5)
        self.assertFalse(diag["independent_validation"])
        # UV roundtrip across different resolutions (bottom-left UV).
        uv = hg_geometry.uv_homography(H, (1280, 720), (1920, 1080))
        uv_inv = hg_geometry.uv_homography(H_inv, (1920, 1080), (1280, 720))
        np.testing.assert_allclose(uv @ uv_inv, np.eye(3), atol=1e-9)
        p_uv = np.array([0.25, 0.75, 1.0])
        px = hg_geometry.uv_to_pixel((1280, 720)) @ p_uv
        self.assertAlmostEqual(px[0], 0.25 * 1280 - 0.5)
        self.assertAlmostEqual(px[1], (1 - 0.75) * 720 - 0.5)
        dst_uv = uv @ p_uv
        dst_px = hg_geometry.uv_to_pixel((1920, 1080)) @ dst_uv
        dst_px = dst_px[:2] / dst_px[2]
        expect = hg_geometry.transform_points(H, [[px[0], px[1]]])[0]
        np.testing.assert_allclose(dst_px, expect, atol=1e-6)

    def test_outlier_rejection(self):
        rng = np.random.default_rng(0)
        src = rng.uniform([20, 20], [300, 200], size=(12, 2))
        H_true = np.array([[1.1, 0.03, 5], [0.01, 0.9, 7],
                           [1e-4, 0, 1.0]])
        dst = hg_geometry.transform_points(H_true, src)
        dst[0] += [80, 60]                      # gross outlier
        H, _, diag = hg_geometry.fit_homography(src, dst)
        self.assertGreaterEqual(diag["inlier_count"], 4)
        np.testing.assert_allclose(
            hg_geometry.transform_points(H, src[1:]),
            hg_geometry.transform_points(H_true, src[1:]), atol=1.0)

    def test_rejects_bad_points(self):
        good4 = quad(640, 480)
        with self.assertRaises(ValueError):
            hg_geometry.fit_homography(good4[:3], good4[:3])       # <4
        col = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], float)
        with self.assertRaises(ValueError):
            hg_geometry.fit_homography(col, good4)                 # collinear
        dup = np.array([[0, 0], [0, 0], [1, 0], [0, 1]], float)
        with self.assertRaises(ValueError):
            hg_geometry.fit_homography(dup, good4)                 # duplicates
        with self.assertRaises(ValueError):
            hg_geometry.points_array([[np.nan, 0]] * 4)            # NaN
        for bad in ([[0, 0]], [[10.5, 20]], [[-1, 10]], [[640]]):
            with self.assertRaises(ValueError):
                hg_geometry.image_size(bad)                        # dims
        with self.assertRaises(ValueError):
            hg_geometry.transform_points(np.eye(3),
                                         [[1, np.inf]])            # nonfinite


class CliValidation(unittest.TestCase):
    def args(self, **kw):
        ns = hg.build_parser().parse_args(kw.pop("argv"))
        return ns

    def _mode(self, argv):
        return hg.validate(hg.build_parser().parse_args(argv))

    def test_valid_combos(self):
        self.assertEqual("points", self._mode(
            ["-c", "0", "-p", "0", "--points", "x.json"]))
        self.assertEqual("live", self._mode(["-c", "1", "-p", "0"]))
        self.assertEqual("live", self._mode(
            ["-c", "2", "-p", "0", "--via", "v.json"]))
        self.assertEqual("map", self._mode(["-c", "1", "-c", "2", "--map"]))
        self.assertEqual("map", self._mode(["-c", "2", "-c", "1", "--map"]))
        self.assertEqual("list", self._mode(["--list"]))
        self.assertEqual("setup_td", self._mode(["--setup-td"]))
        self.assertEqual("restore_td", self._mode(["--restore-td"]))
        # -c 0 -p 0 accepted with no --monitor (TD branch selects itself)
        self.assertEqual("live", self._mode(["-c", "0", "-p", "0"]))
        self.assertEqual("warp", self._mode(["-c", "0", "-p", "0",
                                             "--warp"]))
        self.assertEqual("warp", self._mode(
            ["-c", "1", "-p", "0", "--warp", "--calibration", "x.json"]))
        self.assertEqual("table_color", self._mode(
            ["-c", "0", "-p", "0", "--table-color", "m.json"]))
        self.assertEqual("live", self._mode(
            ["-c", "0", "-p", "0", "-s_id", "Table_01"]))
        self.assertEqual("live", self._mode(
            ["-c", "0", "-p", "0", "--surface-id", "Wall_2"]))
        self.assertEqual("live", self._mode(
            ["-c", "0", "-p", "0", "--surface", "Wall-2"]))
        self.assertEqual("init_profile", self._mode(
            ["-c", "0", "-p", "0", "--init-profile", "x.json"]))

    def test_rejected_combos(self):
        bad = [
            ["-p", "0"],                                  # no camera
            ["-c", "0"],                                  # no projector
            ["-c", "0", "-p", "1"],                       # unknown projector
            ["-c", "0", "-c", "1", "-p", "0"],            # two cams, no --map
            ["-c", "0", "-c", "1", "--map"],              # map only 1<->2
            ["-c", "1", "-c", "2", "-c", "0", "--map"],   # three cams
            ["-c", "1", "-c", "2", "--map", "-p", "0"],   # map + projector
            ["-c", "2", "-p", "0"],                       # depth needs --via
            ["-c", "0", "-p", "0", "--via", "v.json"],    # via only for c2
            ["-c", "1", "-c", "2", "--map", "--points", "x"],
            ["-c", "0", "-p", "0", "--monitor", "1"],     # removed option
            ["-c", "0", "-p", "0", "--webcam-index", "0"],  # removed option
            ["--setup-td", "-c", "0"],
            ["-c", "0", "--warp"],                        # no -p
            ["-c", "2", "-p", "0", "--warp"],             # cams only 0/1
            ["-c", "0", "-c", "1", "-p", "0", "--warp"],  # one camera
            ["-c", "0", "-p", "0", "--warp", "--points", "x"],
            ["-c", "0", "-p", "0", "--warp", "--via", "v"],
            ["-c", "1", "-c", "2", "--map", "--warp"],
            ["-c", "0", "-p", "0", "--calibration", "x"],  # only --warp
            ["-c", "1", "-p", "0", "--table-color", "m"],  # c0 only
            ["-c", "0", "--table-color", "m"],             # no -p
            ["-c", "0", "-p", "0", "--table-color", "m",
             "--warp"],
            ["-c", "0", "-p", "0", "--table-color", "m",
             "--points", "x"],
        ]
        for argv in bad:
            with self.assertRaises(hg.HgError, msg=argv):
                self._mode(argv)

    def test_main_nonzero_on_bad_combo(self):
        self.assertEqual(2, hg.main(["-c", "2", "-p", "0"]))

    def test_surface_id_rejected(self):
        for bad in ("../x", "a/b", "a\\b", "Table 01", ".hidden",
                    "CON", "com1", "ünïcode", "", "a" * 65):
            with self.assertRaises(hg.HgError, msg=bad):
                self._mode(["-c", "0", "-p", "0", "-s_id", bad])

    def test_unknown_and_negative_ids(self):
        for argv in (["-c", "99", "-p", "0"],
                     ["-c", "-1", "-p", "0", "--points", "x"],
                     ["-c", "0", "-p", "0", "--webcam-index", "-3"],
                     ["-c", "1", "-c", "7", "--map"]):
            with self.assertRaises(hg.HgError, msg=argv):
                self._mode(argv)


class PointsMode(unittest.TestCase):
    def _fixture(self, tmp, cam_pts, proj_pts, cam_size=(640, 480),
                 proj_size=(1920, 1080)):
        path = os.path.join(tmp, "pairs.json")
        with open(path, "w") as fh:
            json.dump({"camera_size": cam_size,
                       "projector_size": proj_size,
                       "camera_points": np.asarray(cam_pts).tolist(),
                       "projector_points": np.asarray(proj_pts).tolist()},
                      fh)
        return path

    def test_offline_writes_strict_json_no_hardware(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = quad(640, 480)
            H_true = np.array([[2.1, 0.1, 40], [0.05, 1.8, 20],
                               [3e-4, 1e-4, 1.0]])
            dst = hg_geometry.transform_points(H_true, src)
            inp = self._fixture(tmp, src, dst)
            out = os.path.join(tmp, "o", "c0_p0.json")
            # offline mode must never reach for TD or any device
            with mock.patch.object(hg, "TdClient",
                                   side_effect=AssertionError("TD!")), \
                    mock.patch.object(hg, "ROOT", tmp):
                rc = hg.main(["-c", "0", "-p", "0", "--points", inp,
                              "-o", out])
            self.assertEqual(0, rc)
            with open(out, encoding="utf-8") as fh:
                doc = json.load(fh)
            self.assertEqual(("ff.hg", 1, "homography"),
                             (doc["schema"], doc["schema_version"],
                              doc["kind"]))
            self.assertEqual("raw_distorted_pixels", doc["source_space"])
            self.assertEqual(
                "planar_approximation_no_distortion_correction",
                doc["lens_policy"])
            np.testing.assert_allclose(
                doc["H_source_to_destination"], H_true /
                np.linalg.norm(H_true), atol=1e-6)
            self.assertFalse(doc["diagnostics"]["independent_validation"])
            for key in ("H_destination_to_source",
                        "H_source_uv_to_destination_uv",
                        "H_destination_uv_to_source_uv", "plane"):
                self.assertIn(key, doc)
            # strict JSON: round-trip has no NaN/Inf tokens
            with open(out, encoding="utf-8") as fh:
                raw = fh.read()
            self.assertNotIn("NaN", raw)
            self.assertNotIn("Infinity", raw)

    def test_out_of_bounds_points_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = quad(640, 480)
            src[0] = [700, 10]
            inp = self._fixture(tmp, src, quad(1920, 1080))
            with mock.patch.object(hg, "ROOT", tmp):
                self.assertEqual(1, hg.main(
                    ["-c", "0", "-p", "0", "--points", inp,
                     "-o", os.path.join(tmp, "x.json")]))


class PatternRecipe(unittest.TestCase):
    """Hardware-free checks of the live-path ArUco recipe (DICT_4X4_250,
    ids 100-111, 4x3 grid)."""

    def test_build_and_detect_all_ids(self):
        img, known = hg.build_pattern(1280, 720)
        result = hg.detect_pattern(img, known)
        self.assertIsNotNone(result, "all 12 known markers should detect")
        cam_pts, proj_pts = result
        self.assertEqual((48, 2), cam_pts.shape)
        self.assertEqual((48, 2), proj_pts.shape)
        # corner ordering recovers the stored drawn pixels (sub-pixel
        # refinement may move detections slightly below 1 px)
        self.assertLess(np.abs(cam_pts - proj_pts).max(), 1.0)
        # sorted-id order: first 4 entries belong to id 100, tl/tr/br/bl
        tl = proj_pts[0]
        self.assertEqual(known[100][0], tl.tolist())

    def test_perspective_warp_fit(self):
        import cv2
        img, known = hg.build_pattern(1280, 720)
        H_true = np.array([[0.45, 0.01, 20.0], [0.005, 0.5, 15.0],
                           [1e-5, 2e-5, 1.0]])
        frame = cv2.warpPerspective(img, H_true.astype(np.float64),
                                    (640, 480),
                                    borderValue=(255, 255, 255))
        result = hg.detect_pattern(frame, known)
        self.assertIsNotNone(result, "warped pattern should still detect")
        cam_pts, proj_pts = result
        H, _, _ = hg_geometry.fit_homography(cam_pts, proj_pts)
        rng = np.random.default_rng(7)
        heldout = rng.uniform([120, 120], [1160, 600], size=(20, 2))
        roundtrip = hg_geometry.transform_points(
            H, hg_geometry.transform_points(H_true, heldout))
        np.testing.assert_allclose(roundtrip, heldout, atol=2.0)

    def test_black_backed_recipe_structure(self):
        """Black canvas, white tile + quiet zone, marker corners are the
        outer code bounds (not the tile), ids 100-111."""
        import cv2
        img, known = hg.build_pattern(1280, 720)
        self.assertEqual((720, 1280, 3), img.shape)
        self.assertEqual(sorted(known), list(range(100, 112)))
        module = min(1280 // (hg.ARUCO_COLS * 9), 720 // (hg.ARUCO_ROWS * 9))
        tile, pitch, side = 8 * module, 9 * module, 6 * module
        gw = (hg.ARUCO_COLS - 1) * pitch + tile
        gh = (hg.ARUCO_ROWS - 1) * pitch + tile
        ox, oy = (1280 - gw) // 2, (720 - gh) // 2
        # canvas outside the grid is black
        self.assertEqual(img[0, 0].tolist(), [0, 0, 0])
        self.assertEqual(img[oy - 1, ox - 1].tolist(), [0, 0, 0])
        # tile body is white; the one-module band around the code stays
        # white (quiet zone); the one-module band between tiles is black
        self.assertEqual(img[oy, ox].tolist(), [255, 255, 255])
        self.assertEqual(img[oy + module - 1, ox + module - 1].tolist(),
                         [255, 255, 255])
        gap_x = ox + tile  # first black column between tiles 0 and 1
        self.assertEqual(img[oy + tile // 2, gap_x].tolist(), [0, 0, 0])
        # stored corners are the marker outer bounds: tile origin + one
        # module inset, spanning `side` pixels
        for i, mid in enumerate(hg.ARUCO_IDS):
            r, c = divmod(i, hg.ARUCO_COLS)
            x, y = ox + c * pitch + module, oy + r * pitch + module
            np.testing.assert_array_equal(
                np.asarray(known[mid]),
                np.array([[x, y], [x + side - 1, y],
                          [x + side - 1, y + side - 1], [x, y + side - 1]]))

    def test_marker_side_grows_with_raster(self):
        """Frozen recipe numbers: 156px at 1280x720, 102 at 640x480,
        240 at 1920x1080 - and all detect."""
        for (w, h, side) in [(1280, 720, 156), (640, 480, 102),
                             (1920, 1080, 240)]:
            img, known = hg.build_pattern(w, h)
            tl, tr = known[100][0], known[100][1]
            self.assertEqual(tr[0] - tl[0] + 1, side,
                             f"{w}x{h} marker side")
            result = hg.detect_pattern(img, known)
            self.assertIsNotNone(result, f"{w}x{h} should detect")
            self.assertEqual((48, 2), result[0].shape)


class SanitizeMap(unittest.TestCase):
    def test_color_to_depth(self):
        depth = np.full((424, 512), 900, np.uint16)
        xy = np.random.uniform(0, 400, (1080, 1920, 2)).astype(np.float32)
        xy[0, 0] = [np.inf, 5]                    # inf
        xy[0, 1] = [-np.inf, 5]                   # -inf
        xy[0, 2] = [np.nan, np.nan]               # nan
        xy[0, 3] = [600, 10]                      # out of bounds x
        xy[0, 4] = [511.6, 10]                    # rounds outside
        xy[0, 5] = [50.4, 30.2]                   # fractional ok
        iy, ix = 30, 50
        depth[iy, ix] = 900
        xy[0, 6] = [10, 10]
        depth[10, 10] = 0                         # zero depth -> invalid
        out, valid = hg_geometry.sanitize_sdk_map(xy, depth, 1, (512, 424))
        self.assertTrue(valid[0, 5])
        for k in (0, 1, 2, 3, 4, 6):
            self.assertFalse(valid[0, k])
            np.testing.assert_array_equal(out[0, k], [-1, -1])
        np.testing.assert_allclose(out[0, 5], [50.4, 30.2])

    def test_depth_to_color(self):
        depth = np.full((424, 512), 700, np.uint16)
        xy = np.random.uniform(0, 1900, (424, 512, 2)).astype(np.float32)
        xy[5, 5] = [np.nan, 0]
        xy[6, 6] = [2000, 0]
        depth[9, 9] = 0
        out, valid = hg_geometry.sanitize_sdk_map(xy, depth, 2, (1920, 1080))
        self.assertFalse(valid[5, 5])
        self.assertFalse(valid[6, 6])
        self.assertFalse(valid[9, 9])             # zero depth
        np.testing.assert_array_equal(out[6, 6], [-1, -1])

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            hg_geometry.sanitize_sdk_map(np.zeros((4, 4), np.float32),
                                         np.zeros((4, 4), np.uint16),
                                         1, (512, 424))
        with self.assertRaises(ValueError):
            hg_geometry.sanitize_sdk_map(
                np.zeros((424, 512, 2), np.float32),
                np.zeros((424, 512), np.int32), 2, (1920, 1080))
        with self.assertRaises(ValueError):
            hg_geometry.sanitize_sdk_map(
                np.zeros((424, 512, 2), np.float32),
                np.zeros((424, 512), np.uint16), 0, (1920, 1080))


class MockMapper(unittest.TestCase):
    """SDK CoordinateMapper contract: correct counts, shapes, dtypes."""

    class FakeMapper:
        def __init__(self):
            self.calls = []

        def MapColorFrameToDepthSpace(self, n_in, depth_ptr, n_out, out):
            self.calls.append(("c2d", n_in.value, n_out.value))
            a = np.frombuffer(out, np.float32)
            a[:] = np.linspace(0, 511, a.size)

        def MapDepthFrameToColorSpace(self, n_in, depth_ptr, n_out, out):
            self.calls.append(("d2c", n_in.value, n_out.value))
            a = np.frombuffer(out, np.float32)
            a[:] = np.linspace(0, 1919, a.size)

    def test_both_directions(self):
        mapper = self.FakeMapper()
        kin = types_simple(mapper)
        depth = np.full((424, 512), 800, np.uint16)
        fake = _fake_pykinect2()
        dims = types.SimpleNamespace(DW=512, DH=424, CW=1920, CH=1080)
        with mock.patch.dict(sys.modules, {"pykinect2": fake}), \
                mock.patch.object(hg, "_kin", return_value=dims):
            c2d = hg._sdk_map(kin, depth, 1, (512, 424))
            d2c = hg._sdk_map(kin, depth, 2, (1920, 1080))
        self.assertEqual(("c2d", 512 * 424, 1920 * 1080),
                         mapper.calls[0])
        self.assertEqual(("d2c", 512 * 424, 512 * 424),
                         mapper.calls[1])
        self.assertEqual(((1080, 1920, 2), np.float32),
                         (c2d.shape, c2d.dtype))
        self.assertEqual(((424, 512, 2), np.float32),
                         (d2c.shape, d2c.dtype))


def _fake_pykinect2():
    """SDK-free pykinect2 stand-in: real ctypes float32 x/y structures."""
    class _DepthSpacePoint(ctypes.Structure):
        _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]

    class _ColorSpacePoint(ctypes.Structure):
        _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]

    fake = types.ModuleType("pykinect2")
    fake.PyKinectV2 = types.SimpleNamespace(
        _DepthSpacePoint=_DepthSpacePoint, _ColorSpacePoint=_ColorSpacePoint)
    return fake


def types_simple(mapper):
    class K:
        pass
    k = K()
    k.mapper = mapper
    return k


class DenseOutput(unittest.TestCase):
    def test_npz_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c1_c2.json")
            xy = np.random.uniform(0, 511, (1080, 1920, 2))
            xy[0, 0] = -1
            valid = np.ones((1080, 1920), bool)
            valid[0, 0] = False
            depth = np.full((424, 512), 900, np.uint16)
            doc, npz = hg.save_dense_map(
                path, hg._device_entry(hg.CAMERAS[1]),
                hg._device_entry(hg.CAMERAS[2]), xy, valid, depth, {})
            with np.load(npz) as z:
                xy_l = z["xy"].copy()
                valid_l = z["valid"].copy()
                depth_l = z["depth_mm"].copy()
            self.assertEqual(doc["map_file"], os.path.basename(npz))
            self.assertEqual("forward_source_to_destination",
                             doc["map_direction"])
            self.assertTrue(doc["snapshot_only"] and doc["depth_required"])
            self.assertEqual(((1080, 1920, 2), np.float32),
                             (xy_l.shape, xy_l.dtype))
            self.assertEqual(np.uint8, valid_l.dtype)
            self.assertEqual((424, 512), depth_l.shape)
            np.testing.assert_array_equal(xy_l[0, 0], [-1, -1])
            self.assertEqual(0, valid_l[0, 0])


class FakeTdClient:
    """Stands in for TdClient: records rpc, serves canned frames."""

    def __init__(self, frame=None, rgba=None, inspect_info=None):
        self.frame = frame
        self.rgba = rgba
        self.rpc_calls = []
        self.captured_tops = []
        self.ended = False
        self._info = inspect_info or {
            "camLogi": {"exists": True, "width": 640, "height": 480},
            "kinectC": {"exists": True, "width": 1280, "height": 720},
            "OART": {"exists": True, "width": 1280, "height": 720},
            "protocol_version": hg.PROTOCOL_VERSION}

    def rpc(self, payload):
        self.rpc_calls.append(payload)
        if payload.get("action") == "end":
            self.ended = True
        return {"ok": True}

    def inspect(self):
        return self._info

    def capture_color(self, top):
        self.captured_tops.append(top)
        return self.frame.copy()

    def capture_depth_rgba(self):
        return self.rgba.copy()


class DepthToProjector(unittest.TestCase):
    def _via_doc(self, H, src_res=(1280, 720), backend="touchdesigner"):
        ncm = {"model": "full_frame_resize",
               "native_resolution": [1920, 1080],
               "td_resolution": list(src_res)}
        src = {"id": "1", "name": "k", "type": "camera",
               "resolution": list(src_res)}
        if backend == "touchdesigner":
            src.update({"capture_backend": "touchdesigner",
                        "top_path": "/project1/kinectC",
                        "mirrorimage": False,
                        "native_color_mapping": ncm})
        else:
            src["capture_backend"] = "offline_correspondences"
            if list(src_res) != [1920, 1080] and backend != "offline_strip":
                src["native_color_mapping"] = ncm
        return {"schema": "ff.hg", "schema_version": 1,
                "kind": "homography",
                "created_utc": "2026-01-01T00:00:00+00:00",
                "source": src,
                "destination": {"id": "P0", "name": "p",
                                "type": "projector",
                                "resolution": [1280, 720]},
                "source_space": "raw_distorted_pixels",
                "lens_policy":
                    "planar_approximation_no_distortion_correction",
                "coordinate_convention": {"origin": "top-left"},
                "H_source_to_destination": H.tolist()}

    def test_malformed_via_rejected_before_hardware(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "bad.json")
            with open(bad, "w") as fh:
                json.dump({"kind": "homography"}, fh)
            args = hg.build_parser().parse_args(
                ["-c", "2", "-p", "0", "--via", bad])
            with mock.patch.object(hg, "TdClient",
                                   side_effect=AssertionError("TD!")), \
                 mock.patch.object(hg, "_kin",
                                   side_effect=AssertionError("hardware!")):
                with self.assertRaises(hg.HgError):
                    hg.run_depth_to_projector(args)
            self.assertEqual(1, hg.main(
                ["-c", "2", "-p", "0", "--via",
                 os.path.join(tmp, "missing.json")]))

    def test_via_variants_rejected(self):
        good = self._via_doc(np.eye(3))
        variants = [
            {**good, "schema_version": 2},                 # wrong version
            {**good, "kind": "dense_map"},
            {**good, "source": {**good["source"],
                                "id": "0"}},               # wrong source
            {**good, "source": {**good["source"],
                                "type": "projector"}},
            {**good, "source": {**good["source"],
                                "resolution": [1920]}},    # bad dims
            {**good, "source": {**good["source"],
                                "resolution": [1366, 768]}},  # unknown
            {**good, "source": {**good["source"],
                                "mirrorimage": True}},        # mirrored
            {**good, "source": {**good["source"],
                                "top_path": "/project1/other"}},
            {**good, "source": {**good["source"],
                                "native_color_mapping":
                                {"model": "crop", "native_resolution":
                                 [1920, 1080],
                                 "td_resolution": [1280, 720]}}},
            # offline 1280x720 without explicit resize metadata
            self._via_doc(np.eye(3), backend="offline_strip"),
            {**good, "destination": {**good["destination"],
                                     "resolution":
                                     [1920.5, 1080]}},     # fractional dims
            {**good, "source_space": "native_sdk_pixels"},
            {**good, "lens_policy": "something_else"},
            {**good, "H_source_to_destination":
                [[1, 0, 0], [0, 1, 0]]},                   # wrong H shape
            {**good, "H_source_to_destination":
                np.zeros((3, 3)).tolist()},                # singular
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for i, doc in enumerate(variants):
                path = os.path.join(tmp, f"v{i}.json")
                with open(path, "w") as fh:
                    json.dump(doc, fh)
                with self.assertRaises(hg.HgError, msg=f"variant {i}"):
                    hg._load_via(path)
            # non-object JSON root
            path = os.path.join(tmp, "list.json")
            with open(path, "w") as fh:
                json.dump([1, 2, 3], fh)
            with self.assertRaises(hg.HgError):
                hg._load_via(path)
            # native 1920x1080 source accepted
            path = os.path.join(tmp, "native.json")
            with open(path, "w") as fh:
                json.dump(self._via_doc(np.eye(3), src_res=(1920, 1080)),
                          fh)
            _, _, _, H_native, _, _ = hg._load_via(path)
            np.testing.assert_allclose(H_native, np.eye(3))

    def test_compose_synthetic(self):
        H_true = np.array([[0.9, 0.02, 10], [0.01, 0.95, 5],
                           [1e-5, 0, 1.0]])
        rgba = np.zeros((424, 512, 4), np.float32)
        rgba[:, :, 2] = 0.8                            # metres -> 800mm
        client = FakeTdClient(rgba=rgba)
        with tempfile.TemporaryDirectory() as tmp:
            via = os.path.join(tmp, "c1_p0.json")
            with open(via, "w") as fh:
                json.dump(self._via_doc(H_true), fh)
            # deterministic native color grid: depth px * 3.5
            gx, gy = np.meshgrid(np.arange(512) * 3.5,
                                 np.arange(424) * 3.5)
            d2c = np.stack([gx, gy], axis=2).astype(np.float32)
            d2c[0, 0] = [-1, -1]
            kin = types.SimpleNamespace()              # mapper only
            fake = types.SimpleNamespace(Kin=lambda *a: kin)
            with mock.patch.object(hg, "TdClient", return_value=client), \
                 mock.patch.object(hg, "_kin", return_value=fake), \
                 mock.patch.object(hg, "_sdk_map", return_value=d2c), \
                 mock.patch.object(hg, "_close_kin"):
                out = os.path.join(tmp, "c2_p0.json")
                rc = hg.main(["-c", "2", "-p", "0", "--via", via,
                              "-o", out])
            self.assertEqual(0, rc)
            with open(out, encoding="utf-8") as fh:
                doc = json.load(fh)
            self.assertEqual("dense_map", doc["kind"])
            prov = doc["provenance"]
            self.assertTrue(prov["fullframe_resize_assumption"])
            np.testing.assert_allclose(
                prov["homography_copy"]["H_source_to_destination"],
                H_true)
            H_native = hg_geometry.resized_source_homography(
                H_true, [1920, 1080], [1280, 720])
            np.testing.assert_allclose(
                prov["homography_copy"]
                ["H_native_1920x1080_to_destination"], H_native)
            with np.load(os.path.join(tmp, doc["map_file"])) as z:
                xy_l = z["xy"].copy()
                valid_l = z["valid"].copy()
            # depth px *3.5 -> native color px -> H_native -> proj px
            y, x = 100, 100
            expect = hg_geometry.transform_points(
                H_native, [[x * 3.5, y * 3.5]])[0]
            np.testing.assert_allclose(xy_l[y, x], expect, atol=1e-3)
            self.assertEqual(1, valid_l[y, x])
            self.assertEqual(0, valid_l[0, 0])
            np.testing.assert_array_equal(xy_l[0, 0], [-1, -1])


class FailurePaths(unittest.TestCase):
    def test_malformed_points_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "bad.json")
            with open(bad, "w") as fh:
                fh.write("{ not json !!")
            out = os.path.join(tmp, "out.json")
            self.assertEqual(1, hg.main(
                ["-c", "0", "-p", "0", "--points", bad, "-o", out]))
            self.assertFalse(os.path.exists(out))

    def test_publication_failure_preserves_previous(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp = os.path.join(tmp, "pairs.json")
            with open(inp, "w") as fh:
                json.dump({"camera_size": [640, 480],
                           "projector_size": [1920, 1080],
                           "camera_points": quad(640, 480).tolist(),
                           "projector_points":
                               quad(1920, 1080).tolist()}, fh)
            out = os.path.join(tmp, "c0_p0.json")
            with mock.patch.object(hg, "ROOT", tmp):
                self.assertEqual(0, hg.main(
                    ["-c", "0", "-p", "0", "--points", inp, "-o", out]))
            with open(out, encoding="utf-8") as fh:
                original = fh.read()
            with mock.patch.object(hg.os, "replace",
                                   side_effect=OSError("denied")):
                self.assertEqual(1, hg.main(
                    ["-c", "0", "-p", "0", "--points", inp, "-o", out]))
            with open(out, encoding="utf-8") as fh:
                self.assertEqual(original, fh.read())
            self.assertFalse(
                [f for f in os.listdir(tmp) if f.endswith(".tmp")])

    def test_failed_hardware_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "c1_c2.json")
            with mock.patch.object(hg, "_kin",
                                   side_effect=ImportError("no sdk")):
                self.assertEqual(1, hg.main(
                    ["-c", "1", "-c", "2", "--map", "-o", out]))
            self.assertFalse(os.path.exists(out))


# -------------------------------------------------- fake TouchDesigner

class _FakeParMode(enum.Enum):
    CONSTANT = 0
    EXPRESSION = 1
    EXPORT = 2
    BIND = 3


class _FakePar:
    def __init__(self, val=None, expr="", mode=_FakeParMode.CONSTANT,
                 menuNames=None, menuLabels=None):
        self.val = val
        self.expr = expr
        self.mode = mode
        self.menuNames = menuNames
        self.menuLabels = menuLabels
        self._menuIndex = 0

    @property
    def menuIndex(self):
        names = self.menuNames
        if names and self.val in names:
            return names.index(self.val)
        return self._menuIndex

    @menuIndex.setter
    def menuIndex(self, v):
        self._menuIndex = v

    def eval(self):
        return self.val


class _ParGroup:
    def __init__(self):
        object.__setattr__(self, "_d", {})

    def __getattr__(self, k):
        try:
            return self._d[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        if not isinstance(v, _FakePar):
            v = _FakePar(v)
        self._d[k] = v

    def __getitem__(self, k):
        return self._d[k]


class _Connector:
    def __init__(self):
        self.in_op = None

    def connect(self, o):
        self.in_op = o

    def disconnect(self):
        self.in_op = None


class FakeOp:
    registry = {}

    def __init__(self, path, opt_type="TOP"):
        self.path = path
        self.name = path.rsplit("/", 1)[-1]
        self.OPType = opt_type
        self.storage = {}
        self.par = _ParGroup()
        self.width, self.height = 1280, 720
        self.nodeX, self.nodeY = 0, 0
        self.text = ""
        self.arr = None
        self._ins = [_Connector()]
        FakeOp.registry[path] = self

    def set_inputs(self, n):
        self._ins = [_Connector() for _ in range(n)]

    def wire(self, i, o):
        self._ins[i].in_op = o

    @property
    def inputs(self):
        return [c.in_op for c in self._ins]

    @property
    def inputConnectors(self):
        return self._ins

    def create(self, cls, name):
        o = FakeOp(self.path + "/" + name,
                   getattr(cls, "_tname", "TOP"))
        o.set_inputs(4)                    # TOPs expose multiple inlets
        if o.OPType == "switchTOP":
            o.par.index = _FakePar(val=0)
        if o.OPType == "executeDAT":
            o.par.active = _FakePar(val=False)
            o.par.framestart = _FakePar(val=False)
        if o.OPType == "constantTOP":
            for a in ("colorr", "colorg", "colorb", "alpha"):
                setattr(o.par, a,
                        _FakePar(val=1.0))      # TD default white
        return o

    def cook(self, force=False):
        pass

    def errors(self):
        return getattr(self, "err", "")

    def copyNumpyArray(self, arr):
        self.copied = arr

    def numpyArray(self, delayed=True):
        return self.arr

    def appendCustomPage(self, name):
        page = types.SimpleNamespace()

        def _mk(par_name, val):
            self.par._d[par_name] = _FakePar(val)
            return self.par._d[par_name]

        # indexable-but-not-list/tuple like a real TD ParTuple
        page.appendToggle = (
            lambda n, label=None: _ParTuple(_mk(n, False)))
        page.appendFloat = lambda n, label=None: _mk(n, 0.0)
        page.appendStr = lambda n, label=None: _mk(n, "")
        return page

    def clear(self):
        self.rows = []

    def appendRow(self, row):
        if not hasattr(self, "rows"):
            self.rows = []
        self.rows.append(list(row))


class _ParTuple:
    """Indexable container like a real TD ParTuple (not a list/tuple)."""

    def __init__(self, par):
        self._par = par

    def __getitem__(self, i):
        if i != 0:
            raise IndexError(i)
        return self._par

    def __len__(self):
        return 1


_SCRIPT_TOP = types.SimpleNamespace(_tname="scriptTOP")
_TEXT_DAT = types.SimpleNamespace(_tname="textDAT")
_KINECT_TOP = types.SimpleNamespace(_tname="kinectTOP")
_GLSL_TOP = types.SimpleNamespace(_tname="glslTOP")
_BASE_COMP = types.SimpleNamespace(_tname="baseCOMP")
_CONST_TOP = types.SimpleNamespace(_tname="constantTOP")
_SWITCH_TOP = types.SimpleNamespace(_tname="switchTOP")
_TABLE_DAT = types.SimpleNamespace(_tname="tableDAT")
_EXECUTE_DAT = types.SimpleNamespace(_tname="executeDAT")


def _gate_eval(out_op, reg):
    """Evaluate the hg_output gate expression semantics on the fake
    (owned Calibrating/Ready toggles + cam selector + surface id are
    real par dependencies, like the live GATE_EXPR)."""
    hg_op = reg["/project1/hg"]
    ctl = reg["/project1/hg_pair_controls"]
    if ctl.par.Calibrating.eval():
        return 2
    cs = reg["/project1/cam_and_kinect"]
    sid = ctl.par.Surfaceid.eval()
    eff = max(0, min(2, int(cs.par.index.eval())))
    if (ctl.par.Ready.eval() and not cs.par.blend.eval()
            and hg_op.storage.get("hg_ready_pair") == [eff, sid]):
        return 1
    return 0


def _fake_td_rpc():
    """Build the inspected /project1 world and return rpc + registry."""
    FakeOp.registry = {}
    root = FakeOp("/project1", "baseCOMP")
    cam = FakeOp("/project1/camLogi", "videodeviceinTOP")
    cam.width, cam.height = 640, 480
    kc = FakeOp("/project1/kinectC", "kinectTOP")
    kc.width, kc.height = 1280, 720
    kc.par.image = "color"
    kc.par.mirrorimage = False
    kd = FakeOp("/project1/kinectD", "kinectTOP")
    kd.par.image = "infrared"
    kd.par.mirrorimage = False
    camk = FakeOp("/project1/cam_and_kinect", "switchTOP")
    camk.set_inputs(3)
    camk.wire(0, cam)
    camk.wire(1, kc)
    camk.wire(2, kd)
    camk.par.index = _FakePar(val=0)
    camk.par.blend = _FakePar(val=False)
    oart = FakeOp("/project1/OART", "nullTOP")
    oart.set_inputs(1)
    oart.wire(0, camk)
    hgo = FakeOp("/project1/hg", "nullTOP")
    hgo.set_inputs(1)
    hgo.wire(0, oart)
    sw = FakeOp("/project1/final_switch", "switchTOP")
    sw.set_inputs(9)
    sw.wire(7, oart)
    sw.par.index = _FakePar(
        val=0, expr="op('/project1/output_mode').par.Mode.menuIndex",
        mode=_FakeParMode.EXPRESSION)
    om = FakeOp("/project1/output_mode")
    # real TD: menuNames are the value strings par.val selects by,
    # menuLabels the display text; index 7 is 'ffv2'/'FFv2 track'
    names = ["camlogi", "kinectc", "kinectd", "chebu", "harbour",
             "skeleton", "scan", "ffv2", "hg_dup"]
    labels = ["camLogi", "Kinect C", "Kinect D", "Chebu", "Harbour",
              "Skeleton", "Scan", "FFv2 track", "hg dup"]
    om.par.Mode = _FakePar(val="camlogi", menuNames=list(names),
                           menuLabels=list(labels))
    module = runpy.run_path(
        HG_TD,
        init_globals={"op": lambda p: FakeOp.registry.get(p),
                      "scriptTOP": _SCRIPT_TOP, "textDAT": _TEXT_DAT,
                      "kinectTOP": _KINECT_TOP,
                      "glslTOP": _GLSL_TOP,
                      "baseCOMP": _BASE_COMP,
                      "constantTOP": _CONST_TOP,
                      "switchTOP": _SWITCH_TOP,
                      "tableDAT": _TABLE_DAT,
                      "executeDAT": _EXECUTE_DAT,
                      "ParMode": _FakeParMode})
    return module["rpc"], FakeOp.registry


class TdSession(unittest.TestCase):
    def setUp(self):
        self.rpc, self.reg = _fake_td_rpc()
        self.tmp = tempfile.mkdtemp()
        self.pattern_img = np.zeros((4, 4, 3), np.uint8)
        self.pattern = hg_wire.encode_array(self.pattern_img)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_setup_wiring_and_helpers(self):
        res = self.rpc({"action": "setup"})
        self.assertTrue(res["ok"], res)
        reg = self.reg
        for name in ("hg_pattern", "hg_pattern_callbacks", "hg_depth_xyz",
                     "hg_output", "hg_output_black", "hg_pair_controls",
                     "hg_profiles", "hg_watch"):
            self.assertIn("/project1/" + name, reg)
            self.assertTrue(reg["/project1/" + name].storage["hg_owned"],
                            name)
        pat = reg["/project1/hg_pattern"]
        self.assertEqual("scriptTOP", pat.OPType)
        self.assertEqual("hg_pattern_callbacks",
                         pat.par.callbacks.eval())
        self.assertIn("onCook", reg["/project1/hg_pattern_callbacks"].text)
        self.assertEqual("depthpointcloud",
                         reg["/project1/hg_depth_xyz"].par.image.eval())
        # actual op classes, not textDAT fallbacks
        self.assertEqual("constantTOP",
                         reg["/project1/hg_output_black"].OPType)
        self.assertEqual("switchTOP", reg["/project1/hg_output"].OPType)
        self.assertEqual("tableDAT", reg["/project1/hg_profiles"].OPType)
        self.assertEqual("executeDAT", reg["/project1/hg_watch"].OPType)
        # opaque black source at the output raster, gate unblended
        black = reg["/project1/hg_output_black"]
        self.assertEqual(0.0, black.par.colorr.eval())
        self.assertEqual(0.0, black.par.colorb.eval())
        self.assertEqual(1.0, black.par.alpha.eval())
        self.assertEqual([1280, 720], [black.par.resolutionw.eval(),
                                     black.par.resolutionh.eval()])
        out = reg["/project1/hg_output"]
        self.assertIs(out.inputs[0], reg["/project1/hg_output_black"])
        self.assertIs(out.inputs[2], pat)
        self.assertFalse(out.par.blend.eval())
        # constant fallback val first, expr set, EXPRESSION mode last
        self.assertEqual(0, out.par.index.eval())
        self.assertIn("Calibrating", out.par.index.expr)
        self.assertIn("Ready", out.par.index.expr)
        self.assertIn("cam_and_kinect", out.par.index.expr)
        self.assertIn("blend", out.par.index.expr)  # fail closed on blend
        self.assertIs(out.par.index.mode, _FakeParMode.EXPRESSION)
        # executeDAT runs its own .text, not a callbacks DAT
        wd = reg["/project1/hg_watch"]
        self.assertIn("onFrameStart", wd.text)
        self.assertTrue(wd.par.active.eval())
        self.assertTrue(wd.par.framestart.eval())
        self.assertIsNone(getattr(wd.par, "callbacks", None)
                          and wd.par.callbacks.eval())
        ctl = reg["/project1/hg_pair_controls"]
        self.assertEqual("Table_01", ctl.par.Surfaceid.eval())
        self.assertFalse(ctl.par.Calibrating.eval())
        self.assertFalse(ctl.par.Ready.eval())
        sw = reg["/project1/final_switch"]
        self.assertIs(sw.inputs[7], reg["/project1/hg"])
        # selection untouched
        self.assertIs(sw.par.index.mode, _FakeParMode.EXPRESSION)
        self.assertEqual(0, sw.par.index.eval())
        # cam selector untouched: user switch, expected inputs, no blend
        cs = reg["/project1/cam_and_kinect"]
        self.assertIs(cs.inputs[0], reg["/project1/camLogi"])
        self.assertIs(cs.inputs[1], reg["/project1/kinectC"])
        self.assertIs(cs.inputs[2], reg["/project1/kinectD"])
        self.assertEqual(0, cs.par.index.eval())
        # kinectD infrared untouched
        self.assertEqual("infrared",
                         reg["/project1/kinectD"].par.image.eval())

    def test_setup_refuses_foreign_names(self):
        FakeOp("/project1/hg_pattern")     # un-owned collision
        res = self.rpc({"action": "setup"})
        self.assertFalse(res["ok"])
        self.assertIn("not hg-owned", res["error"])

    def test_setup_reuses_existing_depth_helper(self):
        # pre-existing hg_depth_xyz with correct settings (not hg-owned)
        # is reused, not reconfigured/refused
        d = FakeOp("/project1/hg_depth_xyz", "kinectTOP")
        d.par.image = "depthpointcloud"
        d.par.mirrorimage = False
        res = self.rpc({"action": "setup"})
        self.assertTrue(res["ok"], res)
        # foreign hg_depth_xyz with wrong settings is refused
        rpc2, reg2 = _fake_td_rpc()
        d2 = FakeOp("/project1/hg_depth_xyz", "kinectTOP")
        d2.par.image = "infrared"
        res2 = rpc2({"action": "setup"})
        self.assertFalse(res2["ok"])

    def test_begin_end_roundtrip(self):
        self.assertTrue(self.rpc({"action": "setup"})["ok"])
        reg = self.reg
        oart, sw, out = (reg["/project1/OART"],
                         reg["/project1/final_switch"],
                         reg["/project1/hg_output"])
        cs = reg["/project1/cam_and_kinect"]
        res = self.rpc({"action": "begin", "owner": "u1",
                        "camera_id": 1, "surface_id": "Wall_9",
                        "pattern": self.pattern})
        self.assertTrue(res["ok"], res)
        self.assertIs(oart.inputs[0], out)
        self.assertIs(out.inputs[2], reg["/project1/hg_pattern"])
        self.assertEqual(2, _gate_eval(out, reg))      # pattern gate
        self.assertIs(sw.par.index.mode, _FakeParMode.CONSTANT)
        self.assertEqual(7, sw.par.index.eval())
        # session selected camera + surface
        self.assertEqual(1, cs.par.index.eval())
        self.assertEqual("Wall_9",
                         reg["/project1/hg_pair_controls"]
                         .par.Surfaceid.eval())
        # concurrent session refused
        res = self.rpc({"action": "begin", "owner": "u2",
                        "pattern": self.pattern})
        self.assertFalse(res["ok"])
        # wrong owner cannot end
        res = self.rpc({"action": "end", "owner": "u2"})
        self.assertFalse(res["ok"])
        res = self.rpc({"action": "end", "owner": "u1"})
        self.assertTrue(res["ok"], res)
        self.assertIs(oart.inputs[0], reg["/project1/cam_and_kinect"])
        self.assertIs(sw.par.index.mode, _FakeParMode.EXPRESSION)
        self.assertEqual(
            "op('/project1/output_mode').par.Mode.menuIndex",
            sw.par.index.expr)
        self.assertEqual(0, sw.par.index.eval())
        self.assertEqual(0, cs.par.index.eval())      # selector restored
        self.assertEqual("Table_01",
                         reg["/project1/hg_pair_controls"]
                         .par.Surfaceid.eval())
        self.assertNotIn("hg_session", reg["/project1/hg"].storage)

    def test_begin_failure_rolls_back(self):
        self.assertTrue(self.rpc({"action": "setup"})["ok"])
        reg = self.reg
        # fail on the first mutation (OART input connect), let restore work
        conn = reg["/project1/OART"].inputConnectors[0]
        real_connect = conn.connect
        calls = []

        def flaky(o):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("simulated mid-begin failure")
            return real_connect(o)

        conn.connect = flaky
        res = self.rpc({"action": "begin", "owner": "u1",
                        "pattern": self.pattern})
        self.assertFalse(res["ok"])
        self.assertIn("rolled back", res["error"])
        self.assertIs(reg["/project1/OART"].inputs[0],
                      reg["/project1/cam_and_kinect"])
        self.assertIs(reg["/project1/final_switch"].par.index.mode,
                      _FakeParMode.EXPRESSION)
        self.assertNotIn("hg_session", reg["/project1/hg"].storage)

    def test_begin_missing_or_bad_pattern_fails_pre_mutation(self):
        self.assertTrue(self.rpc({"action": "setup"})["ok"])
        sw = self.reg["/project1/final_switch"]
        for bad in ({"action": "begin", "owner": "u1"},
                    {"action": "begin", "owner": "u1",
                     "pattern": {"encoding": "zlib-base64",
                                 "dtype": "uint8", "shape": [4, 4, 3],
                                 "data": "!!!notbase64!!!"}},
                    {"action": "begin", "owner": "u1",
                     "pattern": "not a dict"}):
            res = self.rpc(bad)
            self.assertFalse(res["ok"], bad)
        # nothing mutated: switch untouched, no session stored
        self.assertIs(sw.par.index.mode, _FakeParMode.EXPRESSION)
        self.assertEqual(0, sw.par.index.eval())
        self.assertNotIn("hg_session", self.reg["/project1/hg"].storage)

    def test_begin_decodes_and_caches_pattern(self):
        self.assertTrue(self.rpc({"action": "setup"})["ok"])
        img = np.zeros((8, 12, 3), np.uint8)
        img[2:6, 3:9] = [10, 20, 240]
        res = self.rpc({"action": "begin", "owner": "u1",
                        "pattern": hg_wire.encode_array(img)})
        self.assertTrue(res["ok"], res)
        pat = self.reg["/project1/hg_pattern"]
        np.testing.assert_array_equal(
            pat.storage["pattern_rgba"], hg_wire.bgr_to_td_rgba(img))
        res = self.rpc({"action": "end", "owner": "u1"})
        self.assertTrue(res["ok"], res)
        self.assertNotIn("pattern_rgba", pat.storage)  # cache cleared

    def test_prepare_pattern_no_wiring(self):
        self.assertTrue(self.rpc({"action": "setup"})["ok"])
        sw, oart = (self.reg["/project1/final_switch"],
                    self.reg["/project1/OART"])
        res = self.rpc({"action": "prepare_pattern",
                        "pattern": self.pattern})
        self.assertTrue(res["ok"], res)
        self.assertEqual([4, 4], [res["height"], res["width"]])
        pat = self.reg["/project1/hg_pattern"]
        self.assertIn("pattern_rgba", pat.storage)
        # nothing rewired, nothing selected, no session
        self.assertIs(oart.inputs[0],
                      self.reg["/project1/cam_and_kinect"])
        self.assertIs(sw.par.index.mode, _FakeParMode.EXPRESSION)
        self.assertNotIn("hg_session",
                         self.reg["/project1/hg"].storage)
        # refused while a session is active
        self.rpc({"action": "begin", "owner": "u1",
                  "pattern": self.pattern})
        res = self.rpc({"action": "prepare_pattern",
                        "pattern": self.pattern})
        self.assertFalse(res["ok"])
        self.rpc({"action": "end", "owner": "u1"})

    def test_pattern_callback_uses_cached_array(self):
        self.assertTrue(self.rpc({"action": "setup"})["ok"])
        src = self.reg["/project1/hg_pattern_callbacks"].text
        ns = {}
        exec(src, ns)
        fake = types.SimpleNamespace(
            storage={}, copied=None,
            copyNumpyArray=lambda a: setattr(fake, "copied", a))
        ns["onCook"](fake)
        self.assertEqual((720, 1280, 4), fake.copied.shape)  # black
        rgba = np.ones((4, 8, 4), np.float32)
        fake.storage["pattern_rgba"] = rgba
        ns["onCook"](fake)
        self.assertIs(rgba, fake.copied)

    def test_end_disconnects_pattern_when_original_inputs_empty(self):
        self.assertTrue(self.rpc({"action": "setup"})["ok"])
        reg = self.reg
        pat = reg["/project1/hg_pattern"]
        pat.storage["pattern_rgba"] = np.zeros((4, 4, 4), np.float32)
        reg["/project1/OART"].inputConnectors[0].connect(pat)
        reg["/project1/hg"].storage["hg_session"] = {
            "owner": "u1", "oart_inputs": [],
            "switch_index": {"val": 0, "expr": "e", "mode": "EXPRESSION"}}
        res = self.rpc({"action": "end", "owner": "u1"})
        self.assertTrue(res["ok"], res)
        self.assertIsNone(reg["/project1/OART"].inputs[0])
        self.assertNotIn("pattern_rgba", pat.storage)

    def _warp_doc(self, H=None):
        return {"schema": "ff.hg", "schema_version": 1,
                "kind": "homography",
                "source": {"id": "0", "name": "camLogi", "type": "camera",
                           "resolution": [640, 480],
                           "top_path": "/project1/camLogi",
                           "capture_backend": "touchdesigner"},
                "destination": {"id": "P0", "name": "p0",
                                "type": "projector",
                                "resolution": [1280, 720]},
                "source_space": "raw_distorted_pixels",
                "lens_policy":
                    "planar_approximation_no_distortion_correction",
                "H_source_to_destination":
                    np.asarray(H if H is not None else np.eye(3)).tolist(),
                "created_utc": "2026-01-01T00:00:00+00:00"}

    TC = {"schema": "ff.hg.table_color", "schema_version": 1,
          "method": "global_attenuation_gray_balance",
          "source_id": "0", "source_top": "/project1/camLogi",
          "destination_id": "P0", "surface_id": "Table_01",
          "source_size": [640, 480],
          "destination_size": [1280, 720],
          "gain_rgb": [0.8, 0.9, 1.0], "headroom": 0.6,
          "accepted": True}

    def _profile(self, tc=None, camera=0, surface="Table_01",
                 revision=None):
        import hashlib, uuid
        P = self._warp_doc()
        sha = hashlib.sha256(json.dumps(
            P, sort_keys=True).encode()).hexdigest()
        if tc is not None:
            tc = dict(tc)
            tc.setdefault("calibration_sha256", sha)
        prof = {"schema": "ff.projective.profile",
                "schema_version": 1, "camera_id": camera,
                "projector_id": 0, "surface_id": surface,
                "revision": revision or uuid.uuid4().hex,
                "geometry_revision": uuid.uuid4().hex,
                "photometric_revision":
                    uuid.uuid4().hex if tc else None,
                "geometry_sha256": sha,
                "P": P, "TC": tc}
        return prof

    def _register(self, tc=None, **kw):
        prof = kw.pop("profile", None) or self._profile(tc=tc)
        payload = {"action": "register", "profile": prof,
                   "warp_params": hg_geometry.warp_parameters(
                       prof["P"])}
        payload.update(kw)
        return self.rpc(payload)

    def _activate(self, camera=0, surface="Table_01", select=True):
        return self.rpc({"action": "activate",
                         "camera_id": camera, "surface_id": surface,
                         "select": select})

    def test_register_activate_routes_and_selects(self):
        self.assertTrue(self.rpc({"action": "setup"})["ok"])
        tc = dict(self.TC)
        res = self._register(tc=tc, provenance="C:/abs/profile.json")
        self.assertTrue(res["ok"], res)
        self.assertEqual("ready", res["status"])
        res = self._activate()
        self.assertTrue(res["ok"], res)
        self.assertEqual([0, "Table_01"], res["ready"])
        reg = self.reg
        w = reg["/project1/hg_warp"]
        color = reg["/project1/hg_table_color"]
        out = reg["/project1/hg_output"]
        self.assertTrue(w.storage["hg_owned"])
        self.assertIn("uH0", reg["/project1/hg_warp_shader"].text)
        self.assertIs(w.inputs[0], reg["/project1/cam_and_kinect"])
        self.assertIs(color.inputs[0], w)
        self.assertIs(out.inputs[1], color)
        self.assertIs(reg["/project1/OART"].inputs[0], out)
        self.assertEqual(1, _gate_eval(out, reg))      # live gate open
        for a, v in zip("xyz", [0.8 * 0.6, 0.9 * 0.6, 1.0 * 0.6]):
            self.assertAlmostEqual(
                v, color.par["vec0value%s" % a].eval())
        self.assertIn("Enabled", color.par.vec1valuex.expr)
        self.assertIn("Strength", color.par.vec1valuex.expr)
        sw = reg["/project1/final_switch"]
        # select=true binds the switch to the user's output_mode menu:
        # label 7 renamed only, other labels/indices preserved
        self.assertIs(sw.par.index.mode, _FakeParMode.EXPRESSION)
        self.assertEqual("op('/project1/output_mode').par.Mode.menuIndex",
                         sw.par.index.expr)
        mode = reg["/project1/output_mode"].par.Mode
        self.assertEqual("ffv2", mode.eval())      # original name string
        self.assertEqual(7, mode.menuIndex)
        names = mode.menuNames                     # names never change:
        self.assertEqual("ffv2", names[7])         # other clients select it
        self.assertEqual("hg_dup", names[8])   # duplicate inlet untouched
        labels = mode.menuLabels                   # only label 7 renamed
        self.assertEqual("Projective / hg", labels[7])
        self.assertEqual("FFv2 track" != labels[7], True)
        self.assertEqual("hg dup", labels[8])
        for i in (0, 1, 6):
            self.assertNotEqual("Projective / hg", labels[i])
        # Ready par is the gate's real dependency
        self.assertTrue(reg["/project1/hg_pair_controls"].par.Ready.eval())
        self.assertFalse(
            reg["/project1/hg_pair_controls"].par.Calibrating.eval())
        # controls untouched node + registry row
        self.assertIsNone(
            reg["/project1/cam_and_kinect"].storage.get("hg_owned"))
        rows = reg["/project1/hg_profiles"].rows
        self.assertEqual("ready", rows[1][5])
        act = reg["/project1/hg"].storage["hg_active_profile"]
        self.assertEqual("Table_01", act["surface_id"])
        # one-time activation layout: owned nodes anchored along the
        # OART row, user nodes never moved
        oart = reg["/project1/OART"]
        self.assertEqual(1, reg["/project1/hg"].storage[
            "hg_layout_version"])
        self.assertLess(w.nodeX, color.nodeX)
        self.assertLess(color.nodeX, out.nodeX)
        self.assertEqual(oart.nodeY, w.nodeY)
        self.assertLess(reg["/project1/hg_pair_controls"].nodeY,
                        oart.nodeY)

    def test_activate_select_false_keeps_selection(self):
        self.assertTrue(self.rpc({"action": "setup"})["ok"])
        self.assertTrue(self._register(tc=dict(self.TC))["ok"])
        res = self._activate(select=False)
        self.assertTrue(res["ok"], res)
        sw = self.reg["/project1/final_switch"]
        self.assertIs(sw.par.index.mode, _FakeParMode.EXPRESSION)
        self.assertEqual(0, sw.par.index.eval())

    def test_geometry_only_profile_gates_black(self):
        self.assertTrue(self.rpc({"action": "setup"})["ok"])
        res = self._register()                          # TC=None
        self.assertTrue(res["ok"], res)
        self.assertEqual("geometry_only", res["status"])
        res = self._activate()
        self.assertTrue(res["ok"], res)
        self.assertIsNone(res["ready"])                  # gate stays 0
        self.assertEqual(0, _gate_eval(
            self.reg["/project1/hg_output"], self.reg))
        ctl = self.reg["/project1/hg_pair_controls"]
        self.assertIn("geometry only", ctl.par.Status.eval())

    def test_activate_missing_profile_refused(self):
        self.assertTrue(self.rpc({"action": "setup"})["ok"])
        res = self._activate()
        self.assertFalse(res["ok"])
        self.assertIn("no profile", res["error"])
        self.assertIn("hg -c", res["error"])
        self.assertEqual(0, _gate_eval(
            self.reg["/project1/hg_output"], self.reg))

    def test_register_refusals(self):
        self.assertTrue(self.rpc({"action": "setup"})["ok"])
        reg = self.reg
        # malformed bundles
        for bad in ({}, {"profile": None},
                    {"profile": {"schema": "x"}}):
            res = self.rpc({"action": "register", **bad})
            self.assertFalse(res["ok"], bad)
        prof = self._profile()
        wp = hg_geometry.warp_parameters(prof["P"])
        bad = dict(prof); bad["geometry_sha256"] = "0" * 64
        res = self.rpc({"action": "register", "profile": bad,
                        "warp_params": wp})
        self.assertFalse(res["ok"])          # hash mismatch refused
        res = self.rpc({"action": "register", "profile": prof})
        self.assertFalse(res["ok"])          # missing warp_params
        # GPU rows that don't derive from P
        wpb = dict(wp)
        wpb["inverse_uv_rows"] = [[v * 1.01 for v in r] for r in
                                  wp["inverse_uv_rows"]]
        res = self.rpc({"action": "register", "profile": prof,
                        "warp_params": wpb})
        self.assertFalse(res["ok"])
        # TC bound to a different surface/hash/projector refused
        tc_ok = dict(self.TC)
        for tc_bad in (dict(tc_ok, surface_id="Elsewhere"),
                       dict(tc_ok, destination_id="P1"),
                       dict(tc_ok, calibration_sha256="0" * 64),
                       dict(tc_ok, source_top="/project1/kinectC")):
            res = self._register(
                profile=self._profile(tc=tc_bad))
            self.assertFalse(res["ok"], tc_bad)
        # non-canonical revision ids refused
        bad = self._profile(); bad["revision"] = "../escape"
        res = self.rpc({"action": "register", "profile": bad,
                        "warp_params": wp})
        self.assertFalse(res["ok"])
        bad = self._profile(); bad["camera_id"] = True
        res = self.rpc({"action": "register", "profile": bad,
                        "warp_params": wp})
        self.assertFalse(res["ok"])
        self.assertEqual({}, reg["/project1/hg"].storage.get(
            "hg_profiles", {}))
        # active session
        reg["/project1/hg"].storage["hg_session"] = {"owner": "x"}
        self.assertFalse(self._register()["ok"])
        reg["/project1/hg"].storage.pop("hg_session")

    def test_watcher_safe_states(self):
        self.assertTrue(self.rpc({"action": "setup"})["ok"])
        reg = self.reg
        cs = reg["/project1/cam_and_kinect"]
        ctl = reg["/project1/hg_pair_controls"]
        # c1 selected without profile -> black + hint
        cs.par.index.val = 1
        self.assertTrue(self.rpc({"action": "resync"})["ok"])
        self.assertIsNone(reg["/project1/hg"].storage.get(
            "hg_ready_pair"))
        self.assertIn("no profile", ctl.par.Status.eval())
        # depth selection -> unsupported, black
        cs.par.index.val = 2
        self.rpc({"action": "resync"})
        self.assertIn("depth", ctl.par.Status.eval())
        # unknown surface -> black
        cs.par.index.val = 0
        ctl.par.Surfaceid = "Nowhere"
        self.rpc({"action": "resync"})
        self.assertEqual(0, _gate_eval(
            reg["/project1/hg_output"], reg))
        # session suppresses resync mutations
        reg["/project1/hg"].storage["hg_session"] = {"owner": "x"}
        reg["/project1/hg"].storage["hg_ready_pair"] = [0, "Table_01"]
        self.rpc({"action": "resync"})
        self.assertEqual([0, "Table_01"],
                         reg["/project1/hg"].storage["hg_ready_pair"])
        reg["/project1/hg"].storage.pop("hg_session")

    def test_blend_on_selector_fails_closed(self):
        self.assertTrue(self.rpc({"action": "setup"})["ok"])
        self.assertTrue(self._register(tc=dict(self.TC))["ok"])
        self.assertTrue(self._activate()["ok"])
        reg = self.reg
        out = reg["/project1/hg_output"]
        self.assertEqual(1, _gate_eval(out, reg))
        # user turns on the camera-switch blend -> mixed feed, black
        cs = reg["/project1/cam_and_kinect"]
        cs.par.blend.val = True
        self.rpc({"action": "resync"})
        self.assertIsNone(reg["/project1/hg"].storage.get(
            "hg_ready_pair"))
        self.assertIn("blend", reg["/project1/hg_pair_controls"]
                      .par.Status.eval())
        self.assertEqual(0, _gate_eval(out, reg))
        # blend back off resyncs the same profile
        cs.par.blend.val = False
        self.rpc({"action": "resync"})
        self.assertEqual(1, _gate_eval(out, reg))

    def test_begin_end_restores_active_profile(self):
        self.assertTrue(self.rpc({"action": "setup"})["ok"])
        self.assertTrue(self._register(tc=dict(self.TC))["ok"])
        self.assertTrue(self._activate()["ok"])
        reg = self.reg
        out = reg["/project1/hg_output"]
        self.rpc({"action": "begin", "owner": "u1",
                  "pattern": self.pattern})
        res = self.rpc({"action": "end", "owner": "u1"})
        self.assertTrue(res["ok"], res)
        # active profile path restored (OART <- hg_output, gate open)
        self.assertIs(reg["/project1/OART"].inputs[0], out)
        self.assertEqual(1, _gate_eval(out, reg))

    def test_gate_transitions_same_pair_session(self):
        """live(1) -> pattern(2) -> live(1) on the SAME camera/surface
        must re-cook the gate through par dependencies alone."""
        self.assertTrue(self.rpc({"action": "setup"})["ok"])
        self.assertTrue(self._register(tc=dict(self.TC))["ok"])
        self.assertTrue(self._activate()["ok"])
        reg = self.reg
        out = reg["/project1/hg_output"]
        ctl = reg["/project1/hg_pair_controls"]
        self.assertEqual(1, _gate_eval(out, reg))
        self.rpc({"action": "begin", "owner": "u1",
                  "camera_id": 0, "surface_id": "Table_01",
                  "pattern": self.pattern})
        self.assertTrue(ctl.par.Calibrating.eval())
        self.assertFalse(ctl.par.Ready.eval())
        self.assertEqual(2, _gate_eval(out, reg))
        self.assertTrue(self.rpc({"action": "end", "owner": "u1"})["ok"])
        self.assertFalse(ctl.par.Calibrating.eval())
        self.assertTrue(ctl.par.Ready.eval())
        self.assertEqual(1, _gate_eval(out, reg))

    def test_failed_begin_gates_black(self):
        self.assertTrue(self.rpc({"action": "setup"})["ok"])
        self.assertTrue(self._register(tc=dict(self.TC))["ok"])
        self.assertTrue(self._activate()["ok"])
        reg = self.reg
        out = reg["/project1/hg_output"]
        self.assertEqual(1, _gate_eval(out, reg))
        # fail the first mutation, then let the rollback succeed
        conn = reg["/project1/OART"].inputConnectors[0]
        real_connect = conn.connect
        calls = []

        def flaky(o):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("simulated mid-begin failure")
            return real_connect(o)
        conn.connect = flaky
        res = self.rpc({"action": "begin", "owner": "u1",
                        "pattern": self.pattern})
        self.assertFalse(res["ok"])
        ctl = reg["/project1/hg_pair_controls"]
        self.assertFalse(ctl.par.Calibrating.eval())
        self.assertFalse(ctl.par.Ready.eval())
        self.assertEqual(0, _gate_eval(out, reg))

    def test_new_tc_revision_reapplies_uniforms(self):
        self.assertTrue(self.rpc({"action": "setup"})["ok"])
        self.assertTrue(self._register(tc=dict(self.TC))["ok"])
        self.assertTrue(self._activate()["ok"])
        reg = self.reg
        color = reg["/project1/hg_table_color"]
        old = color.par.vec0valuex.eval()
        # same camera/surface, different coefficients/revision: the
        # gate closes and the NEW uniforms are installed - no user
        # selector change needed
        tc2 = dict(self.TC); tc2["gain_rgb"] = [0.5, 0.9, 1.0]
        self.assertTrue(self._register(tc=tc2)["ok"])
        self.assertFalse(
            reg["/project1/hg_pair_controls"].par.Ready.eval())
        self.assertEqual(0, _gate_eval(
            reg["/project1/hg_output"], reg))
        self.rpc({"action": "resync"})              # watcher frame
        self.assertNotEqual(old, color.par.vec0valuex.eval())
        self.assertAlmostEqual(0.5 * 0.6,
                               color.par.vec0valuex.eval())
        self.assertEqual(1, _gate_eval(
            reg["/project1/hg_output"], reg))
        # a new H for the same key invalidates TC -> geometry only,
        # gate black
        prof = self._profile()
        prof["P"] = dict(prof["P"])
        import hashlib
        prof["P"]["H_source_to_destination"] = (
            np.eye(3) * 1.01).tolist()
        prof["geometry_sha256"] = hashlib.sha256(json.dumps(
            prof["P"], sort_keys=True).encode()).hexdigest()
        res = self.rpc({"action": "register", "profile": prof,
                        "warp_params": hg_geometry.warp_parameters(
                            prof["P"])})
        self.assertTrue(res["ok"], res)
        self.assertEqual("geometry_only", res["status"])
        self.rpc({"action": "resync"})
        self.assertEqual(0, _gate_eval(
            reg["/project1/hg_output"], reg))

    def test_apply_failure_leaves_gate_black(self):
        self.assertTrue(self.rpc({"action": "setup"})["ok"])
        self.assertTrue(self._register(tc=dict(self.TC))["ok"])
        reg = self.reg
        # owned color node that fails its shader cook at apply time
        color = FakeOp("/project1/hg_table_color", "glslTOP")
        color.storage["hg_owned"] = True
        color.err = "compile failed"
        res = self._activate()
        self.assertTrue(res["ok"], res)
        self.assertIsNone(res["ready"])
        self.assertEqual(0, _gate_eval(
            reg["/project1/hg_output"], reg))
        self.assertIn("apply failed",
                      reg["/project1/hg_pair_controls"].par.Status.eval())

    def test_watcher_text_is_executable_and_ticks(self):
        self.assertTrue(self.rpc({"action": "setup"})["ok"])
        wd = self.reg["/project1/hg_watch"]
        self.assertEqual("executeDAT", wd.OPType)
        g = {"op": lambda p: FakeOp.registry.get(p),
             "scriptTOP": _SCRIPT_TOP, "textDAT": _TEXT_DAT,
             "kinectTOP": _KINECT_TOP, "glslTOP": _GLSL_TOP,
             "baseCOMP": _BASE_COMP, "constantTOP": _CONST_TOP,
             "switchTOP": _SWITCH_TOP, "tableDAT": _TABLE_DAT,
             "executeDAT": _EXECUTE_DAT, "ParMode": _FakeParMode}
        exec(wd.text, g)
        # depth selection through the real cached watch function
        self.reg["/project1/cam_and_kinect"].par.index.val = 2
        g["onFrameStart"](0)
        self.assertIn("depth", self.reg["/project1/hg_pair_controls"]
                      .par.Status.eval())
        self.assertIn("_watch_fn",
                      self.reg["/project1/hg"].storage)
        # an unexpected tick error clears ready state, never silent:
        # make _resync itself blow up by removing pair controls
        self.reg["/project1/hg"].storage["hg_ready_pair"] = [0, "Table_01"]
        hg_op = self.reg["/project1/hg"]
        del FakeOp.registry["/project1/hg_pair_controls"]
        g["onFrameStart"](0)
        self.assertIsNone(hg_op.storage.get("hg_ready_pair"))
        self.assertIsNone(hg_op.storage.get("hg_active_profile"))

    def test_update_pattern_within_session(self):
        self.assertTrue(self.rpc({"action": "setup"})["ok"])
        self.rpc({"action": "begin", "owner": "u1",
                  "pattern": self.pattern})
        reg = self.reg
        pat, oart, sw = (reg["/project1/hg_pattern"],
                         reg["/project1/OART"],
                         reg["/project1/final_switch"])
        img2 = np.full((6, 10, 3), 200, np.uint8)
        res = self.rpc({"action": "update_pattern", "owner": "u1",
                        "pattern": hg_wire.encode_array(img2)})
        self.assertTrue(res["ok"], res)
        self.assertEqual([6, 10], [res["height"], res["width"]])
        np.testing.assert_array_equal(
            pat.storage["pattern_rgba"], hg_wire.bgr_to_td_rgba(img2))
        # routing/selection/session unchanged
        self.assertIs(oart.inputs[0],
                      reg["/project1/hg_output"])
        self.assertIs(sw.par.index.mode, _FakeParMode.CONSTANT)
        self.assertEqual(7, sw.par.index.eval())
        sess = reg["/project1/hg"].storage["hg_session"]
        self.assertEqual("u1", sess["owner"])
        self.assertEqual(["/project1/cam_and_kinect"],
                         sess["oart_inputs"])
        res = self.rpc({"action": "end", "owner": "u1"})
        self.assertTrue(res["ok"], res)

    def test_update_pattern_refusals(self):
        self.assertTrue(self.rpc({"action": "setup"})["ok"])
        reg = self.reg
        pat, oart, sw = (reg["/project1/hg_pattern"],
                         reg["/project1/OART"],
                         reg["/project1/final_switch"])
        # no active session
        res = self.rpc({"action": "update_pattern", "owner": "u1",
                        "pattern": self.pattern})
        self.assertFalse(res["ok"])
        self.rpc({"action": "begin", "owner": "u1",
                  "pattern": self.pattern})
        cached = pat.storage["pattern_rgba"]
        # wrong owner
        res = self.rpc({"action": "update_pattern", "owner": "u2",
                        "pattern": self.pattern})
        self.assertFalse(res["ok"])
        self.assertIs(cached, pat.storage["pattern_rgba"])
        # malformed array
        res = self.rpc({"action": "update_pattern", "owner": "u1",
                        "pattern": "nope"})
        self.assertFalse(res["ok"])
        self.assertIs(cached, pat.storage["pattern_rgba"])
        # externally rewired OART
        oart.inputConnectors[0].disconnect()
        res = self.rpc({"action": "update_pattern", "owner": "u1",
                        "pattern": self.pattern})
        self.assertFalse(res["ok"])
        self.assertIn("routing", res["error"])
        self.assertIs(cached, pat.storage["pattern_rgba"])
        oart.inputConnectors[0].connect(reg["/project1/hg_output"])
        # externally changed camera selection
        reg["/project1/cam_and_kinect"].par.index.val = 1
        res = self.rpc({"action": "update_pattern", "owner": "u1",
                        "pattern": self.pattern})
        self.assertFalse(res["ok"])
        self.assertIn("camera selection", res["error"])
        reg["/project1/cam_and_kinect"].par.index.val = 0
        # externally changed surface selection
        reg["/project1/hg_pair_controls"].par.Surfaceid = "Other"
        res = self.rpc({"action": "update_pattern", "owner": "u1",
                        "pattern": self.pattern})
        self.assertFalse(res["ok"])
        self.assertIn("surface", res["error"])
        reg["/project1/hg_pair_controls"].par.Surfaceid = "Table_01"
        # externally changed output selection
        sw.par.index.val = 3
        res = self.rpc({"action": "update_pattern", "owner": "u1",
                        "pattern": self.pattern})
        self.assertFalse(res["ok"])
        self.assertIn("selection", res["error"])
        self.assertIs(cached, pat.storage["pattern_rgba"])
        sw.par.index.val = 7
        res = self.rpc({"action": "update_pattern", "owner": "u1",
                        "pattern": self.pattern})
        self.assertTrue(res["ok"], res)
        self.rpc({"action": "end", "owner": "u1"})

    def test_manual_restore_after_crash(self):
        self.assertTrue(self.rpc({"action": "setup"})["ok"])
        self.rpc({"action": "begin", "owner": "u1",
                  "pattern": self.pattern})
        # simulate dead client: never end; manual restore must fix
        res = self.rpc({"action": "restore"})
        self.assertTrue(res["ok"], res)
        self.assertIs(self.reg["/project1/OART"].inputs[0],
                      self.reg["/project1/cam_and_kinect"])
        res = self.rpc({"action": "restore"})
        self.assertFalse(res["ok"])          # nothing left to restore

    def test_capture_wire_bgr_and_raw(self):
        top = self.reg["/project1/camLogi"]
        top.arr = (np.arange(12, dtype=np.float32)
                   .reshape(2, 2, 3) / 12.0)
        res = self.rpc({"action": "capture", "top": "/project1/camLogi",
                        "format": "bgr"})
        self.assertTrue(res["ok"], res)
        np.testing.assert_array_equal(
            hg_wire.decode_array(res["array"]),
            hg_wire.td_color_to_bgr(top.arr))
        self.assertEqual("bgr", res["format"])
        # file-capture keys are rejected, not ignored
        res = self.rpc({"action": "capture", "top": "/project1/camLogi",
                        "png": "x.png"})
        self.assertFalse(res["ok"])
        res = self.rpc({"action": "capture", "top": "/project1/camLogi",
                        "format": "raw"})
        self.assertTrue(res["ok"], res)
        np.testing.assert_array_equal(
            hg_wire.decode_array(res["array"]),
            top.arr.astype(np.float32))

    def test_capture_depth_validates_helper(self):
        self.assertTrue(self.rpc({"action": "setup"})["ok"])
        d = self.reg["/project1/hg_depth_xyz"]
        d.par.image = "infrared"                     # misconfigured
        res = self.rpc({"action": "capture",
                        "top": "/project1/hg_depth_xyz",
                        "format": "raw"})
        self.assertFalse(res["ok"])
        d.par.image = "depthpointcloud"
        d.arr = None                                 # no data -> error
        res = self.rpc({"action": "capture",
                        "top": "/project1/hg_depth_xyz",
                        "format": "raw"})
        self.assertFalse(res["ok"])
        d.arr = np.zeros((424, 512, 4), np.float32)
        res = self.rpc({"action": "capture",
                        "top": "/project1/hg_depth_xyz",
                        "format": "raw"})
        self.assertTrue(res["ok"], res)
        self.assertEqual((424, 512, 4),
                         hg_wire.decode_array(res["array"]).shape)
        # TOP errors surface instead of stale data
        self.reg["/project1/camLogi"].err = "cook exploded"
        res = self.rpc({"action": "capture", "top": "/project1/camLogi",
                        "format": "bgr"})
        self.assertFalse(res["ok"])
        self.assertIn("cook exploded", res["error"])

    def test_setup_preflight_refusals(self):
        def fresh():
            return _fake_td_rpc()

        # wrong node type
        rpc, reg = fresh()
        reg["/project1/camLogi"].OPType = "moviefileinTOP"
        res = rpc({"action": "setup"})
        self.assertFalse(res["ok"])
        self.assertNotIn("/project1/hg_pattern", reg)

        # hg input chain changed
        rpc, reg = fresh()
        reg["/project1/hg"].wire(0, reg["/project1/kinectC"])
        res = rpc({"action": "setup"})
        self.assertFalse(res["ok"])
        self.assertNotIn("/project1/hg_pattern", reg)

        # cycle: OART downstream of hg
        rpc, reg = fresh()
        reg["/project1/OART"].wire(0, reg["/project1/hg"])
        res = rpc({"action": "setup"})
        self.assertFalse(res["ok"])
        self.assertIn("cycle" if "cycle" in res["error"] else "downstream",
                      res["error"])

        # switch input 7 unavailable
        rpc, reg = fresh()
        reg["/project1/final_switch"].set_inputs(3)
        res = rpc({"action": "setup"})
        self.assertFalse(res["ok"])

        # active session blocks setup
        rpc, reg = fresh()
        reg["/project1/hg"].storage["hg_session"] = {"owner": "u9"}
        res = rpc({"action": "setup"})
        self.assertFalse(res["ok"])

    def test_restore_failure_retains_snapshot(self):
        self.assertTrue(self.rpc({"action": "setup"})["ok"])
        self.rpc({"action": "begin", "owner": "u1",
                  "pattern": self.pattern})
        reg = self.reg
        # delete the original OART input: reconnect must fail actionably
        del FakeOp.registry["/project1/cam_and_kinect"]
        res = self.rpc({"action": "end", "owner": "u1"})
        self.assertFalse(res["ok"])
        self.assertIn("restore-td", res["error"])
        # snapshot retained; switch restored anyway
        self.assertIn("hg_session", reg["/project1/hg"].storage)
        self.assertIs(reg["/project1/final_switch"].par.index.mode,
                      _FakeParMode.EXPRESSION)
        # pattern disconnected even though original path was missing
        self.assertIsNone(reg["/project1/OART"].inputs[0])
        # wrong owner still cannot end it
        res = self.rpc({"action": "end", "owner": "u2"})
        self.assertFalse(res["ok"])
        # node returns -> manual restore succeeds
        camk = FakeOp("/project1/cam_and_kinect", "switchTOP")
        camk.set_inputs(3)
        camk.par.index = _FakePar(val=0)
        res = self.rpc({"action": "restore"})
        self.assertTrue(res["ok"], res)
        self.assertIs(reg["/project1/OART"].inputs[0], camk)
        self.assertNotIn("hg_session", reg["/project1/hg"].storage)

    def test_begin_rollback_failure_retains_snapshot(self):
        self.assertTrue(self.rpc({"action": "setup"})["ok"])
        conn = self.reg["/project1/OART"].inputConnectors[0]
        def always_fail(o):
            raise RuntimeError("nope")
        conn.connect = always_fail
        res = self.rpc({"action": "begin", "owner": "u1",
                        "pattern": self.pattern})
        self.assertFalse(res["ok"])
        self.assertNotIn("rolled back:", res["error"])
        self.assertIn("restore-td", res["error"])
        self.assertIn("hg_session",
                      self.reg["/project1/hg"].storage)


class WarpParameters(unittest.TestCase):
    def _doc(self, H, src_res=(640, 480), dst_res=(1280, 720),
             kind="homography"):
        return {"schema": "ff.hg", "schema_version": 1, "kind": kind,
                "source": {"id": "0", "name": "camLogi", "type": "camera",
                           "resolution": list(src_res),
                           "top_path": "/project1/camLogi",
                           "capture_backend": "touchdesigner"},
                "destination": {"id": "P0", "name": "p0",
                                "type": "projector",
                                "resolution": list(dst_res)},
                "source_space": "raw_distorted_pixels",
                "lens_policy":
                    "planar_approximation_no_distortion_correction",
                "H_source_to_destination": np.asarray(H).tolist(),
                "created_utc": "2026-01-01T00:00:00+00:00"}

    def test_inverse_uv_rows_pull_map(self):
        H = np.array([[3.0, 0.0, 20.0], [0.0, 2.0, 30.0], [0, 0, 1]])
        params = hg_geometry.warp_parameters(self._doc(H))
        self.assertEqual("0", params["source_id"])
        self.assertEqual("/project1/camLogi", params["source_top"])
        rows = np.asarray(params["inverse_uv_rows"])
        self.assertLessEqual(np.abs(rows).max(), 1.0)  # normalized
        H_inv = np.linalg.inv(H)
        for pd in ([100, 100], [1200, 700], [640, 360]):
            # dest px centre (top-left) -> bottom-left UV -> rows -> src UV
            uv_d = np.array([(pd[0] + .5) / 1280,
                             1.0 - (pd[1] + .5) / 720])
            q = rows @ np.array([uv_d[0], uv_d[1], 1.0])
            uv_s = q[:2] / q[2]
            src_px = np.array([uv_s[0] * 640 - .5,
                               (1.0 - uv_s[1]) * 480 - .5])
            np.testing.assert_allclose(
                src_px, hg_geometry.transform_points(H_inv, [pd])[0],
                atol=1e-6)

    def test_rejects_bad_artifacts(self):
        with self.assertRaises(ValueError):
            hg_geometry.warp_parameters(
                self._doc(np.eye(3), kind="dense_map"))
        with self.assertRaises(ValueError):
            hg_geometry.warp_parameters(self._doc(np.zeros((3, 3))))
        bad = self._doc(np.eye(3))
        bad["source"].pop("top_path")
        with self.assertRaises(ValueError):
            hg_geometry.warp_parameters(bad)
        bad2 = self._doc(np.eye(3))
        bad2["source"]["id"] = "2"
        with self.assertRaises(ValueError):
            hg_geometry.warp_parameters(bad2)


class TdLivePath(unittest.TestCase):
    def _args(self, tmp, extra=None):
        argv = ["-c", "0", "-p", "0", "-o",
                os.path.join(tmp, "c0_p0.json")]
        if extra:
            argv += extra
        return hg.build_parser().parse_args(argv)

    def _no_files(self):
        """Live path must never touch image/session files."""
        return (mock.patch.object(np, "save",
                                side_effect=AssertionError("no np.save")),
                mock.patch("cv2.imwrite",
                           side_effect=AssertionError("no imwrite")),
                mock.patch("cv2.imread",
                           side_effect=AssertionError("no imread")),
                mock.patch("tempfile.mkdtemp",
                           side_effect=AssertionError("no workspace")))

    def _frame(self):
        import cv2
        pattern, _ = hg.build_pattern(1280, 720)
        return cv2.warpPerspective(
            pattern, np.array([[0.45, 0.01, 20.0], [0.005, 0.5, 15.0],
                               [1e-5, 2e-5, 1.0]]), (640, 480),
            borderValue=(255, 255, 255))

    def test_live_success_and_cleanup(self):
        client = FakeTdClient(frame=self._frame())
        with tempfile.TemporaryDirectory() as tmp:
            guards = self._no_files()
            for g_ in guards:
                g_.start()
            try:
                with mock.patch.object(hg, "ROOT", tmp):
                    hg.run_live(self._args(tmp), client=client,
                                settle=0, timeout=5)
            finally:
                for g_ in guards:
                    g_.stop()
            self.assertTrue(client.ended)
            begin = [c for c in client.rpc_calls
                     if c.get("action") == "begin"][0]
            np.testing.assert_array_equal(
                hg_wire.decode_array(begin["pattern"]),
                hg.build_pattern(1280, 720)[0])
            with open(os.path.join(tmp, "c0_p0.json")) as fh:
                doc = json.load(fh)
            self.assertEqual("touchdesigner",
                             doc["source"]["capture_backend"])
            self.assertEqual("/project1/camLogi",
                             doc["source"]["top_path"])
            self.assertEqual([1280, 720],
                             doc["destination"]["resolution"])
            self.assertEqual(7,
                             doc["destination"]["td_branch"]
                             ["switch_input"])
            self.assertNotIn("display", doc["destination"])

    def test_live_failure_still_restores(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        client = FakeTdClient(frame=frame)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(hg, "ROOT", tmp):
                with self.assertRaises(hg.HgError) as cm:
                    hg.run_live(self._args(tmp), client=client,
                                settle=0, timeout=0.5)
            self.assertTrue(client.ended)
            # timeout hook persisted the last frame + facts under
            # the repo-local smoke/hg_failures (ROOT patched to tmp)
            expected = len(hg.build_pattern(1280, 720)[1])
            msg = str(cm.exception)
            self.assertIn("detected 0/%d known markers" % expected,
                          msg)
            self.assertIn("near-white fraction", msg)
            diag_dir = os.path.join(tmp, "smoke", "hg_failures")
            dumps = [f for f in os.listdir(diag_dir)
                     if f.endswith(".json")]
            self.assertEqual(1, len(dumps))
            with open(os.path.join(diag_dir, dumps[0])) as fh:
                meta = json.load(fh)
            self.assertEqual(0, meta["known_count"])
            self.assertEqual(expected, meta["expected_count"])
            self.assertEqual([640, 480], meta["camera_size"])
            self.assertEqual("Table_01", meta["surface_id"])
            self.assertTrue(os.path.isfile(meta["frame_path"]))
            self.assertIn("hg_failures", meta["frame_path"])
            self.assertAlmostEqual(
                0.0, meta["near_white_fraction"])
            # no profile/publish on failure
            self.assertFalse(os.path.exists(
                os.path.join(tmp, "c0_p0.json")))
            self.assertFalse(os.path.isdir(
                os.path.join(tmp, "tools", "pipe")))

    def test_live_failure_diag_error_does_not_mask(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        client = FakeTdClient(frame=frame)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(hg, "ROOT", tmp):
                with mock.patch(
                        "hg_diagnostics.save_detection_failure",
                        side_effect=OSError("disk full")):
                    with self.assertRaises(hg.HgError) as cm:
                        hg.run_live(self._args(tmp), client=client,
                                    settle=0, timeout=0.5)
            self.assertTrue(client.ended)
            msg = str(cm.exception)
            self.assertIn("fewer than 4 known markers", msg)
            self.assertIn("diagnostic unavailable", msg)

    def test_live_failure_preserves_existing_profile(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        client = FakeTdClient(frame=frame)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(hg, "ROOT", tmp):
                _prof, ppath = hg._profiles().publish_geometry(
                    tmp, dict(self.H_DOC), "Table_01", 0)
                with open(ppath, "rb") as fh:
                    before = fh.read()
                with self.assertRaises(hg.HgError):
                    hg.run_live(self._args(tmp), client=client,
                                settle=0, timeout=0.5)
                with open(ppath, "rb") as fh:
                    self.assertEqual(before, fh.read())
            self.assertTrue(client.ended)

    def test_live_restore_failure_no_artifact(self):
        client = FakeTdClient(frame=self._frame())
        real_rpc = client.rpc

        def flaky(payload):
            if payload.get("action") == "end":
                raise hg.HgError("TD: restore incomplete: boom")
            return real_rpc(payload)
        client.rpc = flaky
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(hg.HgError):
                hg.run_live(self._args(tmp), client=client,
                            settle=0, timeout=5)
            self.assertFalse(os.path.exists(
                os.path.join(tmp, "c0_p0.json")))

    def test_live_no_active_end_is_silent(self):
        client = FakeTdClient(frame=self._frame())
        real_rpc = client.rpc

        def flaky(payload):
            if payload.get("action") == "end":
                raise hg.HgError("TD: no active hg session to restore")
            return real_rpc(payload)
        client.rpc = flaky
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(hg, "ROOT", tmp):
                hg.run_live(self._args(tmp), client=client,
                            settle=0, timeout=5)
            self.assertTrue(os.path.exists(
                os.path.join(tmp, "c0_p0.json")))


    H_DOC = {"schema": "ff.hg", "schema_version": 1,
             "kind": "homography",
             "source": {"id": "0", "name": "camLogi", "type": "camera",
                        "resolution": [640, 480],
                        "top_path": "/project1/camLogi",
                        "capture_backend": "touchdesigner"},
             "destination": {"id": "P0", "name": "p0",
                             "type": "projector",
                             "resolution": [1280, 720]},
             "source_space": "raw_distorted_pixels",
             "lens_policy":
                 "planar_approximation_no_distortion_correction",
             "H_source_to_destination": np.eye(3).tolist()}

    def test_live_registers_shared_profile(self):
        client = FakeTdClient(frame=self._frame())
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(hg, "ROOT", tmp):
                hg.run_live(self._args(tmp), client=client,
                            settle=0, timeout=5)
            begin = [c for c in client.rpc_calls
                     if c.get("action") == "begin"][0]
            self.assertEqual(0, begin["camera_id"])
            self.assertEqual("Table_01", begin["surface_id"])
            profile_path = os.path.join(
                tmp, "tools", "pipe", "p0", "Table_01", "c0",
                "profile.json")
            with open(profile_path, encoding="utf-8") as fh:
                prof = json.load(fh)
            self.assertEqual("ff.projective.profile", prof["schema"])
            self.assertIsNone(prof["TC"])           # geometry publish
            self.assertIsNone(prof["photometric_revision"])
            reg = [c for c in client.rpc_calls
                   if c.get("action") == "register"][0]
            self.assertEqual(prof["geometry_sha256"],
                             reg["profile"]["geometry_sha256"])
            self.assertEqual(prof, reg["profile"])   # exact P preserved
            self.assertEqual(
                "/project1/camLogi",
                reg["warp_params"]["source_top"])

    def test_live_register_failure_is_honest(self):
        client = FakeTdClient(frame=self._frame())
        real_rpc = client.rpc

        def flaky(payload):
            if payload.get("action") == "register":
                raise hg.HgError("TD: register refused")
            return real_rpc(payload)
        client.rpc = flaky
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(hg, "ROOT", tmp):
                with self.assertRaises(hg.HgError) as cm:
                    hg.run_live(self._args(tmp), client=client,
                                settle=0, timeout=5)
            self.assertIn("TD load failed", str(cm.exception))
            # the saved artifacts stay valid on disk
            self.assertTrue(os.path.isfile(os.path.join(
                tmp, "tools", "pipe", "p0", "Table_01", "c0",
                "profile.json")))

    def test_run_warp_publishes_registers_activates(self):
        client = FakeTdClient(frame=np.zeros((2, 2, 3), np.uint8))
        with tempfile.TemporaryDirectory() as tmp:
            cal = os.path.join(tmp, "c0_p0.json")
            with open(cal, "w") as fh:
                json.dump(self.H_DOC, fh)
            args = hg.build_parser().parse_args(
                ["-c", "0", "-p", "0", "--warp", "--calibration", cal])
            with mock.patch.object(hg, "ROOT", tmp):
                hg.run_warp(args, client=client)
        reg = [c for c in client.rpc_calls
               if c.get("action") == "register"][0]
        self.assertEqual(self.H_DOC, reg["profile"]["P"])
        self.assertIsNone(reg["profile"]["TC"])
        self.assertEqual("Table_01",
                         reg["profile"]["surface_id"])
        act = [c for c in client.rpc_calls
               if c.get("action") == "activate"][0]
        self.assertEqual(0, act["camera_id"])
        self.assertEqual("Table_01", act["surface_id"])
        self.assertTrue(act["select"])
        # mismatched -c refused before rpc
        args = hg.build_parser().parse_args(
            ["-c", "1", "-p", "0", "--warp", "--calibration", cal])
        client.rpc_calls.clear()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(hg, "ROOT", tmp):
                with self.assertRaises(hg.HgError):
                    hg.run_warp(args, client=client)
        self.assertEqual([], [c for c in client.rpc_calls
                              if c.get("action") == "activate"])
        # missing profile without --calibration refused with hint
        client.rpc_calls.clear()
        args = hg.build_parser().parse_args(
            ["-c", "0", "-p", "0", "--warp"])
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(hg, "ROOT", tmp):
                with self.assertRaises(hg.HgError) as cm:
                    hg.run_warp(args, client=client)
        self.assertIn("run " + ".\\hg" + " -c 0 -p 0 -s_id "
                      "Table_01 first", str(cm.exception))
        # missing file refused
        args = hg.build_parser().parse_args(
            ["-c", "0", "-p", "0", "--warp",
             "--calibration", "no/such.json"])
        with self.assertRaises(hg.HgError):
            hg.run_warp(args, client=client)

    def test_init_profile_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            cal = os.path.join(tmp, "c0_p0.json")
            with open(cal, "w") as fh:
                json.dump(self.H_DOC, fh)
            args = hg.build_parser().parse_args(
                ["-c", "0", "-p", "0", "--init-profile", cal])
            with mock.patch.object(hg, "ROOT", tmp):
                hg.run_init_profile(args)
                ppath = os.path.join(
                    tmp, "tools", "pipe", "p0", "Table_01", "c0",
                    "profile.json")
                with open(ppath) as fh:
                    prof = json.load(fh)
                self.assertEqual(self.H_DOC, prof["P"])
                self.assertIsNone(prof["TC"])
                # second run: same geometry hash -> no new revision
                rev = prof["revision"]
                hg.run_init_profile(args)
                with open(ppath) as fh:
                    self.assertEqual(rev, json.load(fh)["revision"])


class PipeStore(unittest.TestCase):
    """tools/pipe store layout + legacy-root isolation."""

    def test_default_output_paths(self):
        self.assertEqual(
            os.path.join(hg.ROOT, "tools", "pipe", "p0", "Table_01",
                         "c0", "P.json"),
            hg.default_output("ordinary", [0], 0, "Table_01"))
        self.assertEqual(
            os.path.join(hg.ROOT, "tools", "pipe", "p0", "Wall_2",
                         "c1", "P.json"),
            hg.default_output("ordinary", [1], 0, "Wall_2"))
        self.assertEqual(
            os.path.join(hg.ROOT, "tools", "pipe", "maps",
                         "c1_c2.json"),
            hg.default_output("map", [1, 2]))
        self.assertEqual(
            os.path.join(hg.ROOT, "tools", "pipe", "maps",
                         "c2_p0.json"),
            hg.default_output("dense", [2], 0, "Table_01"))

    def test_rehydrate_pipe_root_only(self):
        """A profile stranded in the legacy out/calibration backup is
        never pushed to TD; after migration it rehydrates normally."""
        import shutil
        client = FakeTdClient()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(hg, "ROOT", tmp):
                prof = hg._profiles()
                prof.publish_geometry(tmp, dict(TdLivePath.H_DOC),
                                      "Table_01", 0)
                legacy = prof.legacy_calibration_root(tmp)
                os.makedirs(os.path.dirname(legacy), exist_ok=True)
                shutil.move(prof.calibration_root(tmp), legacy)
                # backup tree alone is invisible to TD rehydration
                self.assertEqual([], hg._rehydrate_profiles(client))
                self.assertFalse(client.rpc_calls)
                # approved migration moves the data; then it registers
                rep = prof.migrate_legacy_profiles(tmp)
                self.assertTrue(rep["copied"])
                res = hg._rehydrate_profiles(client)
                self.assertEqual([("c0/Table_01", None)], res)
                reg = [c for c in client.rpc_calls
                       if c.get("action") == "register"]
                self.assertEqual(1, len(reg))
                self.assertEqual("Table_01",
                                 reg[0]["profile"]["surface_id"])

    def test_migrate_pipe_mode_and_runner(self):
        self.assertEqual(
            "migrate_pipe",
            hg.validate(hg.build_parser().parse_args(
                ["--migrate-pipe"])))
        with self.assertRaises(hg.HgError):
            hg.validate(hg.build_parser().parse_args(
                ["--migrate-pipe", "-c", "0"]))
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(hg, "ROOT", tmp):
                hg.run_migrate_pipe()      # empty store: no error


class TdConversions(unittest.TestCase):
    def test_td_pointcloud_depth_mm(self):
        rgba = np.zeros((424, 512, 4), np.float32)
        rgba[:, :, 2] = 0.8                       # 0.8 m everywhere
        rgba[100, 200, 2] = 0.5                   # -> depth row 323
        rgba[10, 10, 2] = np.nan
        rgba[11, 10, 2] = 0.0
        rgba[12, 10, 2] = 100.0                   # > 65535 mm sentinel
        rgba[13, 10, 2] = -1.0
        depth = hg_geometry.td_pointcloud_depth_mm(rgba)
        self.assertEqual(np.uint16, depth.dtype)
        self.assertEqual(500, depth[423 - 100, 200])
        self.assertEqual(800, depth[423 - 50, 50])
        for r in (10, 11, 12, 13):
            self.assertEqual(0, depth[423 - r, 10])
        with self.assertRaises(ValueError):
            hg_geometry.td_pointcloud_depth_mm(np.zeros((4, 4, 4)))

    def test_resized_source_homography_half_pixel(self):
        H = np.eye(3)
        Hn = hg_geometry.resized_source_homography(
            H, [1920, 1080], [1280, 720])
        # native px (960,540) -> resized px centre convention
        sx, sy = 1280 / 1920, 720 / 1080
        expect = [sx * 960 + (sx - 1) / 2, sy * 540 + (sy - 1) / 2]
        np.testing.assert_allclose(
            hg_geometry.transform_points(Hn, [[960, 540]])[0],
            expect, atol=1e-9)


class TdTransport(unittest.TestCase):
    """Intercept subprocess.run and eval the emitted TD expression so the
    exact wire path (json.dumps -> repr -> json.loads -> rpc) is tested."""

    def _run_expr(self, payload):
        captured = {}
        calls = []

        def _rpc(p):
            captured["payload"] = p
            return {"ok": True, "echo": p}

        def fake_import(name, *a, **k):
            if name == "json":
                return json
            if name == "runpy":
                def run_path(path, init_globals=None):
                    captured["helper_path"] = path
                    captured["globals"] = init_globals
                    return {"rpc": _rpc}
                return types.SimpleNamespace(run_path=run_path)
            return __import__(name, *a, **k)

        def fake_run(cmd, **kw):
            calls.append(cmd)
            # stdin request carries {"op": "eval", "code": <expr>}
            request = json.loads(kw["input"])
            captured["request"] = request
            expr = request["code"]
            g = {"__import__": fake_import, "op": object(),
                 "scriptTOP": object(), "textDAT": object(),
                 "kinectTOP": object(), "glslTOP": object(),
                 "baseCOMP": object(), "ParMode": object(),
                 "constantTOP": object(), "switchTOP": object(),
                 "tableDAT": object(), "executeDAT": object()}
            return types.SimpleNamespace(
                returncode=0, stdout=eval(expr, g), stderr="")

        with mock.patch.object(hg.subprocess, "run", fake_run):
            res = hg.TdClient().rpc(payload)
        return res, captured, calls

    def test_uses_json_stdin_not_eval_arg(self):
        _, cap, calls = self._run_expr({"action": "inspect"})
        cmd = calls[0]
        self.assertEqual("-", cmd[-1])
        self.assertIn("--json", cmd)
        self.assertNotIn("--eval", cmd)
        self.assertEqual("eval", cap["request"]["op"])
        self.assertEqual(hg.TD_HELPER, cap["helper_path"])
        self.assertNotIn("\\", cap["helper_path"])

    def test_payload_roundtrips_literals(self):
        payload = {"action": "capture", "top": "/project1/camLogi",
                   "format": "bgr",
                   "flag": True, "off": False, "none": None,
                   "s": "it's \"quoted\" ünïcödé  spaces",
                   "nested": {"a": [1, None, "x\\y\\z"]}}
        res, cap, calls = self._run_expr(payload)
        self.assertTrue(calls)
        got = cap["payload"]
        for k in ("top", "format", "flag", "off", "none", "s", "nested"):
            self.assertEqual(payload[k], got[k], k)
        self.assertTrue(res["ok"])

    def test_large_wire_pattern_travels_stdin(self):
        # 1280x720x3 base64 exceeds the ~32K Windows cmdline limit
        img = np.random.randint(0, 255, (720, 1280, 3), np.uint8)
        payload = {"action": "begin", "owner": "o",
                   "pattern": hg_wire.encode_array(img)}
        res, cap, _ = self._run_expr(payload)
        self.assertTrue(res["ok"])
        np.testing.assert_array_equal(
            img, hg_wire.decode_array(cap["payload"]["pattern"]))
        self.assertGreater(len(cap["request"]["code"]), 32768)

    def test_non_dict_bridge_result_rejected(self):
        def fake_run(cmd, **kw):
            return types.SimpleNamespace(
                returncode=0, stdout="[1, 2]", stderr="")
        with mock.patch.object(hg.subprocess, "run", fake_run):
            with self.assertRaises(hg.HgError):
                hg.TdClient().rpc({"action": "inspect"})


class WireCodec(unittest.TestCase):
    def test_roundtrip_uint8_and_float32(self):
        for arr in (np.random.randint(0, 255, (8, 9, 3), np.uint8),
                    np.arange(24, dtype=np.float32).reshape(2, 3, 4)):
            out = hg_wire.decode_array(hg_wire.encode_array(arr))
            self.assertEqual(arr.dtype, out.dtype)
            self.assertEqual(arr.shape, out.shape)
            np.testing.assert_array_equal(arr, out)

    def test_roundtrip_nan_inf_and_noncontiguous(self):
        a = np.arange(6 * 7 * 3, dtype=np.float32).reshape(6, 7, 3)
        a[0, 0, 0] = np.nan
        a[1, 1, 1] = np.inf
        view = a[::2, :, :]                       # non-contiguous
        out = hg_wire.decode_array(hg_wire.encode_array(view))
        self.assertEqual("<f4", out.dtype.str)
        np.testing.assert_array_equal(view, out)

    def test_malformed_messages_rejected(self):
        good = hg_wire.encode_array(np.zeros((2, 2, 3), np.uint8))
        for bad in ("x", {}, {**good, "encoding": "raw"},
                    {**good, "dtype": "float64"},
                    {**good, "shape": [2, 2]},
                    {**good, "shape": [0, 2, 3]},
                    {**good, "data": "not_base64!"},
                    {**good, "data": good["data"][:10]},
                    {**good, "data": good["data"] +
                        good["data"][-8:]},
                    {**good, "shape": [2, 2, 4]}):
            with self.assertRaises(ValueError, msg=repr(bad)[:60]):
                hg_wire.decode_array(bad)
        with self.assertRaises(ValueError):
            hg_wire.encode_array(np.zeros((2, 2, 3), np.int32))
        with self.assertRaises(ValueError):
            hg_wire.encode_array(np.zeros((4, 4), np.uint8))  # 2-D

    def test_bgr_rgba_orientation_exact(self):
        bgr = np.random.randint(0, 255, (6, 7, 3), np.uint8)
        rgba = hg_wire.bgr_to_td_rgba(bgr)
        self.assertEqual(np.float32, rgba.dtype)
        self.assertEqual((6, 7, 4), rgba.shape)
        self.assertTrue((rgba[:, :, 3] == 1).all())
        np.testing.assert_array_equal(
            hg_wire.td_color_to_bgr(rgba), bgr)   # exact roundtrip

    def test_td_color_to_bgr_flips_and_converts(self):
        rgba = np.zeros((2, 1, 4), np.float32)
        rgba[0, 0, :3] = [1, 0, 0]        # bottom row, R=1
        rgba[1, 0, :3] = [0, 0, 1]        # top row, B=1
        bgr = hg_wire.td_color_to_bgr(rgba)
        np.testing.assert_array_equal(bgr[0, 0], [255, 0, 0])
        np.testing.assert_array_equal(bgr[1, 0], [0, 0, 255])


class CmdLauncher(unittest.TestCase):
    def test_hg_cmd_help(self):
        rc = subprocess.run(["cmd", "/c", os.path.join(ROOT, "hg.cmd"),
                             "--help"], capture_output=True, text=True)
        self.assertEqual(0, rc.returncode)
        self.assertIn("--map", rc.stdout)


if __name__ == "__main__":
    unittest.main()
