"""hg.py - homography / device-map utility for the Fulcrum toolchain.

Logical devices (all live frames arrive through TouchDesigner, never
through OpenCV device capture or desktop windows):

    c0  /project1/camLogi  VideoDeviceInTOP (Logitech), TD raster.
    c1  /project1/kinectC  KinectTOP color, TD raster 1280x720.
    c2  Kinect v2 depth, captured in TD as /project1/hg_depth_xyz
        (KinectTOP image='depthpointcloud', metres XYZ float RGBA,
        512x424), converted by hg_geometry.td_pointcloud_depth_mm.
        /project1/kinectD is the user's depth display input; metric SDK
        mapping still uses the separate XYZ helper, not display intensities.
    p0  Fengmi M055MGN projector. ADB address 192.168.0.170:5555 is
        recorded only; hg makes no ADB calls. Pixels leave through the
        OART -> /project1/hg -> final_switch input 7 branch onto the
        HDMI display during an explicitly user-run calibration.

Homography artifacts are raw distorted-pixel planar approximations: no
lens calibration is applied or implied, and they hold only for the
measured plane and fixed device configuration. Pure SDK coordinate
mapper outputs are labelled native_sdk_pixels instead.
"""
import argparse
import ctypes
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(TOOLS_DIR, "td"))
sys.path.insert(0, os.path.join(TOOLS_DIR, "cc"))
KINECT_DIR = os.path.join(TOOLS_DIR, "kinect")
TD_DIR = os.path.join(TOOLS_DIR, "td")
TD_MJS = os.path.join(TD_DIR, "td.mjs")
TD_HELPER = pathlib.Path(
    TD_DIR, "projective_td.py").resolve().as_posix()
PROTOCOL_VERSION = 3
CAM_SWITCH_TOP = "/project1/cam_and_kinect"


def _wire():
    """Lossless wire codec and image conversion (tools/hg_wire.py)."""
    import hg_wire
    return hg_wire

SCHEMA = "ff.hg"
SCHEMA_VERSION = 1

CAMERAS = {
    0: {"id": "0", "name": "webcam via TD /project1/camLogi",
        "type": "camera", "resolution": None},   # actual TD raster
    1: {"id": "1", "name": "Kinect color via TD /project1/kinectC",
        "type": "camera", "resolution": [1920, 1080]},  # native SDK space
    2: {"id": "2", "name": "Kinect depth via TD /project1/hg_depth_xyz",
        "type": "camera", "resolution": [512, 424]},
}
CAMERA_TOPS = {0: "/project1/camLogi", 1: "/project1/kinectC"}
DEPTH_TOP = "/project1/hg_depth_xyz"
TD_ROOT = "/project1"
TD_SWITCH = "/project1/final_switch"
TD_SWITCH_INPUT = 7
TD_OART = "/project1/OART"
TD_HG = "/project1/hg"
TD_NATIVE_COLOR = [1920, 1080]
TD_KNOWN_COLOR_RASTERS = ([1920, 1080], [1280, 720])

PROJECTORS = {
    0: {"id": "P0", "name": "Fengmi M055MGN projector",
        "type": "projector", "resolution": None,  # actual TD branch raster
        "adb": "192.168.0.170:5555",
        "adb_note": "address recorded only; hg makes no ADB calls",
        "image_path": "TD branch /project1/OART -> /project1/hg -> "
                      "final_switch input 7 (HDMI)"},
}

# Pattern recipe: uniquely identified ArUco markers, no
# symmetric-chessboard orientation ambiguity.
ARUCO_DICT = "DICT_4X4_250"
ARUCO_IDS = list(range(100, 112))       # 4 columns x 3 rows
ARUCO_COLS, ARUCO_ROWS = 4, 3
MIN_MARKER_SIDE = 20

SETTLE_S = 2.0
CAPTURE_TIMEOUT_S = 10.0
RETRY_S = 0.25

SOURCE_SPACE = "raw_distorted_pixels"
LENS_POLICY = "planar_approximation_no_distortion_correction"
SDK_SOURCE_SPACE = "native_sdk_pixels"
SDK_LENS_POLICY = "factory_sdk_coordinate_mapper"
SDK_VALIDITY = ("observed depth snapshot; includes SDK factory mapping; "
                "no global planar homography")
VALID_MASK_NOTE = (
    "valid mask checks finiteness/bounds/nonzero depth only - no "
    "visibility, occlusion or confidence guarantee; no camera or "
    "projector rays are modelled")

COORD_CONVENTION = {
    "origin": "top-left",
    "x": "right",
    "y": "down",
    "pixels": "integer pixel centres",
    "vectors": "column vectors",
    "uv": "bottom-left origin, half-pixel centre convention "
          "(see tools/hg_geometry.py uv_to_pixel)",
}


class HgError(Exception):
    """User-facing failure; printed without a traceback."""


# ---------------------------------------------------------------- helpers

def _np():
    import numpy as np
    return np


def _cv2():
    import cv2
    return cv2


def _geometry():
    import hg_geometry
    return hg_geometry


def _diagnostics():
    import hg_diagnostics
    return hg_diagnostics


def _profiles():
    import projective_profiles
    return projective_profiles


def _kin():
    sys.path.insert(0, KINECT_DIR)
    import kincap
    return kincap


def _open_mapper():
    """Kinect sensor + CoordinateMapper only - no stream readers."""
    return _kin().Kin()


def _close_kin(kin):
    try:
        for kind, reader in getattr(kin, "_readers", {}).items():
            handle = kin._handles.get(kind)
            if handle:
                try:
                    reader.UnsubscribeFrameArrived(handle)
                except Exception:
                    pass
        kin._sensor.Close()
    except Exception:
        pass


# ------------------------------------------------------------- ArUco grid

def build_pattern(width, height):
    """Black-backed grid with one white quiet-zone module per marker.

    A 4x4 code and its black border occupy six modules. The eight-module
    white tile sits on a nine-module pitch, leaving a black gap between
    tiles. Correspondences describe the marker, not its white surround.
    """
    cv2 = _cv2()
    np = _np()
    module = int(min(width // (ARUCO_COLS * 9),
                     height // (ARUCO_ROWS * 9)))
    side = 6 * module
    if side < MIN_MARKER_SIDE:
        raise HgError(
            f"marker side {side}px < {MIN_MARKER_SIDE}px on "
            f"{width}x{height} raster")
    tile, pitch = 8 * module, 9 * module
    gw = (ARUCO_COLS - 1) * pitch + tile
    gh = (ARUCO_ROWS - 1) * pitch + tile
    ox, oy = (width - gw) // 2, (height - gh) // 2
    img = np.zeros((height, width, 3), np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_250)
    corners = {}
    for i, marker_id in enumerate(ARUCO_IDS):
        r, c = divmod(i, ARUCO_COLS)
        tx, ty = ox + c * pitch, oy + r * pitch
        img[ty:ty + tile, tx:tx + tile] = 255
        x, y = tx + module, ty + module
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side)
        img[y:y + side, x:x + side] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        corners[marker_id] = [[x, y], [x + side - 1, y],
                              [x + side - 1, y + side - 1],
                              [x, y + side - 1]]
    return img, corners


def detect_pattern(frame, known_corners, min_ids=4):
    """Detect known ArUco ids; return (cam_pts, proj_pts) or None."""
    cv2 = _cv2()
    np = _np()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(
        dictionary, params).detectMarkers(frame)
    if ids is None:
        return None
    found = {}
    for marker_corners, marker_id in zip(corners, ids.ravel()):
        marker_id = int(marker_id)
        if marker_id in known_corners:
            found[marker_id] = marker_corners.reshape(4, 2)
    if len(found) < min_ids:
        return None
    cam_pts, proj_pts = [], []
    for marker_id in sorted(found):
        cam_pts.extend(found[marker_id])
        proj_pts.extend(known_corners[marker_id])
    return np.asarray(cam_pts, np.float64), np.asarray(proj_pts, np.float64)


# --------------------------------------------------------------- TD client

class TdClient:
    """Structured JSON rpc into TouchDesigner through tools/td/td.mjs.

    TD_WS_URL is honoured by td.mjs itself. Images travel as compressed
    array messages inside JSON; no temporary image files are used."""

    def __init__(self, helper=None):
        self.helper = helper or TD_HELPER
        if not os.path.isfile(TD_MJS):
            raise HgError(f"td.mjs not found at {TD_MJS}")
        if not os.path.isfile(self.helper):
            raise HgError(f"TD helper not found at {self.helper}")

    def rpc(self, payload):
        payload = dict(payload)
        # The payload crosses as data, not code: json.dumps then
        # json.loads inside TD (JSON true/false/null are not Python
        # literals). repr() quotes both literals for the eval parser.
        # The whole eval request travels on stdin (--json -): base64
        # array payloads exceed the Windows command-line limit.
        expr = (
            "__import__('json').dumps("
            "__import__('runpy').run_path(%r, init_globals={'op':op,"
            "'scriptTOP':scriptTOP,'textDAT':textDAT,'kinectTOP':kinectTOP,"
            "'ParMode':ParMode,'glslTOP':glslTOP,'baseCOMP':baseCOMP,"
            "'constantTOP':constantTOP,'switchTOP':switchTOP,"
            "'tableDAT':tableDAT,'executeDAT':executeDAT}"
            ")['rpc']("
            "__import__('json').loads(%r)))"
            % (self.helper, json.dumps(payload)))
        request = json.dumps({"op": "eval", "code": expr})
        try:
            proc = subprocess.run(
                ["node", TD_MJS, "--json", "-"], input=request,
                capture_output=True, text=True, timeout=60)
        except FileNotFoundError:
            raise HgError("node not found on PATH (required by td.mjs)")
        except subprocess.TimeoutExpired:
            raise HgError("TD bridge timed out")
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise HgError("TD bridge failed: " + detail)
        out = proc.stdout.strip()
        try:
            res = json.loads(out)
        except ValueError:
            # wscontrol 'eval' sends back a Python repr of the expression
            # result; our expression returns a JSON string, so decode the
            # literal first, then the JSON inside it.
            try:
                import ast
                res = json.loads(ast.literal_eval(out))
            except Exception:
                raise HgError("TD bridge returned non-JSON: " + out[:300])
        if not isinstance(res, dict):
            raise HgError(
                "TD bridge returned unexpected result: " + repr(res)[:300])
        if res.get("ok") is False:
            raise HgError("TD: " + str(res.get("error")))
        return res

    def inspect(self):
        return self.rpc({"action": "inspect"})["inspect"]

    def capture_color(self, top):
        """Wire capture -> top-down uint8 BGR [H,W,3]."""
        res = self.rpc({"action": "capture", "top": top,
                        "format": "bgr"})
        np = _np()
        try:
            arr = _wire().decode_array(res.get("array"))
        except ValueError as e:
            raise HgError("TD color capture malformed: " + str(e))
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
            raise HgError("TD returned malformed color frame")
        return arr

    def capture_depth_rgba(self):
        """Wire capture -> raw float32 TD bottom-up RGBA point cloud."""
        res = self.rpc({"action": "capture", "top": DEPTH_TOP,
                        "format": "raw"})
        try:
            return _wire().decode_array(res.get("array"))
        except ValueError as e:
            raise HgError("TD depth capture malformed: " + str(e))


def _td_raster(info, name):
    node = info.get(name) or {}
    if not node.get("exists"):
        raise HgError(f"TD node {name} missing; run .\\hg --setup-td")
    w, h = node.get("width"), node.get("height")
    if not w or not h:
        raise HgError(f"TD node {name} reports no raster size")
    return [int(w), int(h)]


# ------------------------------------------------------------ JSON output

def _write_json_atomic(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{uuid.uuid4().hex[:12]}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, allow_nan=False)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _device_entry(dev, resolution=None, extra=None):
    entry = dict(dev)
    if resolution is not None:
        entry["resolution"] = list(resolution)
    if extra:
        entry.update({k: v for k, v in extra.items() if v is not None})
    return entry


def _base_doc(kind, source, destination,
              source_space=SOURCE_SPACE, lens_policy=LENS_POLICY,
              validity=("fixed plane and fixed device configuration only; "
                        "no lens distortion correction applied")):
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "created_utc": datetime.datetime.now(datetime.timezone.utc)
            .isoformat(timespec="seconds"),
        "source": source,
        "destination": destination,
        "coordinate_convention": dict(COORD_CONVENTION),
        "source_space": source_space,
        "lens_policy": lens_policy,
        "validity": validity,
    }


def save_homography(path, source, destination, H, H_inv, diagnostics, plane,
                    warnings):
    g = _geometry()
    doc = _base_doc("homography", source, destination)
    doc["H_source_to_destination"] = H.tolist()
    doc["H_destination_to_source"] = H_inv.tolist()
    doc["H_source_uv_to_destination_uv"] = g.uv_homography(
        H, source["resolution"], destination["resolution"]).tolist()
    doc["H_destination_uv_to_source_uv"] = g.uv_homography(
        H_inv, destination["resolution"], source["resolution"]).tolist()
    doc["plane"] = plane
    doc["diagnostics"] = diagnostics
    doc["warnings"] = warnings
    _write_json_atomic(path, doc)
    return doc


def save_dense_map(path, source, destination, xy, valid, depth_mm, extra,
                   sdk_space=False):
    """Write the .npz sibling first (complete), then atomically write JSON."""
    np = _np()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    npz_name = (os.path.splitext(os.path.basename(path))[0]
                + "." + uuid.uuid4().hex[:12] + ".npz")
    npz_path = os.path.join(os.path.dirname(path) or ".", npz_name)
    with open(npz_path, "wb") as fh:
        np.savez(fh, xy=xy.astype(np.float32),
                 valid=valid.astype(np.uint8),
                 depth_mm=depth_mm.astype(np.uint16))
    if sdk_space:
        doc = _base_doc("dense_map", source, destination,
                        source_space=SDK_SOURCE_SPACE,
                        lens_policy=SDK_LENS_POLICY,
                        validity=SDK_VALIDITY)
    else:
        doc = _base_doc("dense_map", source, destination)
    doc["map_file"] = npz_name
    doc["map_direction"] = "forward_source_to_destination"
    doc["map_layout"] = {
        "xy": "float32 [sourceH, sourceW, 2] destination pixels; "
              "[-1, -1] marks invalid",
        "valid": "uint8 [sourceH, sourceW]",
        "depth_mm": "uint16 [424, 512] snapshot used for this map",
    }
    doc["snapshot_only"] = True
    doc["depth_required"] = True
    doc["regenerate_when"] = "scene depth or device configuration changes"
    doc["sampling_note"] = (
        "forward scatter map; do NOT feed to cv2.remap - run the reverse "
        "--map for pull sampling")
    doc.update(extra)
    _write_json_atomic(path, doc)
    return doc, npz_path


def default_output(mode, cams=None, projector=None,
                   surface_id="Table_01"):
    """Shared calibration artifacts live in tools/pipe/: standalone P
    exports sit beside their profile (p0/<surface>/c<id>/P.json) while
    SDK and dense-composed maps stay under tools/pipe/maps/. An
    explicit -o still overrides the export path, never the canonical
    profile path."""
    if mode == "map":
        name = f"c{cams[0]}_c{cams[1]}.json"
        return os.path.join(ROOT, "tools", "pipe", "maps", name)
    if mode == "dense":
        name = f"c{cams[0]}_p{projector}.json"
        return os.path.join(ROOT, "tools", "pipe", "maps", name)
    pdir = _profiles().profile_dir(ROOT, surface_id, cams[0])
    return os.path.join(pdir, "P.json")


# ------------------------------------------------------------ points mode

def run_points(args):
    np = _np()
    g = _geometry()
    if not os.path.isfile(args.points):
        raise HgError(f"--points file not found: {args.points}")
    with open(args.points, encoding="utf-8") as fh:
        spec = json.load(fh)
    try:
        cam_size = g.image_size(spec["camera_size"])
        proj_size = g.image_size(spec["projector_size"])
        cam_pts = g.points_array(spec["camera_points"])
        proj_pts = g.points_array(spec["projector_points"])
    except (KeyError, TypeError, ValueError) as e:
        raise HgError(f"bad --points file: {e}")
    for pts, size, label in ((cam_pts, cam_size, "camera_points"),
                             (proj_pts, proj_size, "projector_points")):
        if np.any(pts[:, 0] < 0) or np.any(pts[:, 0] >= size[0]) \
                or np.any(pts[:, 1] < 0) or np.any(pts[:, 1] >= size[1]):
            raise HgError(f"{label} outside {size[0]}x{size[1]} image bounds")
    try:
        H, H_inv, diag = g.fit_homography(cam_pts, proj_pts)
    except ValueError as e:
        raise HgError(str(e))
    source = _device_entry(
        CAMERAS[args.cameras[0]], cam_size,
        extra={"capture_backend": "offline_correspondences"})
    destination = _device_entry(PROJECTORS[args.projector], proj_size)
    out = args.output or default_output(
        "ordinary", args.cameras, args.projector, args.surface_id)
    doc = save_homography(
        out, source, destination, H, H_inv, diag, args.plane,
        warnings=["offline correspondence mode; no live "
                  "TD branch or camera used",
                  "no independent validation performed"])
    # explicit offline mode: publish the geometry profile locally, but
    # nothing is sent to TD (no registration without --warp/activate)
    profile, profile_path = _profiles().publish_geometry(
        ROOT, doc, args.surface_id, args.cameras[0])
    print(f"profile -> {profile_path} (not registered; offline)")
    print(f"saved -> {out}")
    print(f"fit: {diag['inlier_count']}/{diag['point_count']} inliers, "
          f"mean {diag['inlier_mean_destination_px']:.3f}px, "
          f"max {diag['inlier_max_destination_px']:.3f}px")


# ---------------------------------------------------------------- TD modes

def _rehydrate_profiles(client):
    """Push every valid CURRENT profile file into the TD registry.

    Only <surface>/c<id>/profile.json current files are considered -
    never revisions archives - and each is fully validated on the PC
    before the register rpc (no TD filesystem reads). Malformed files
    are a clear error, not a silent skip."""
    prof = _profiles()
    root = prof.calibration_root(ROOT)
    results = []
    if not os.path.isdir(root):
        return results
    for sid in sorted(os.listdir(root)):
        sdir = os.path.join(root, sid)
        if not os.path.isdir(sdir):
            continue
        for cdir in sorted(os.listdir(sdir)):
            if cdir not in ("c0", "c1"):
                continue                    # revisions/, c2 skipped
            path = os.path.join(sdir, cdir, "profile.json")
            if not os.path.isfile(path):
                continue
            try:
                with open(path, encoding="utf-8") as fh:
                    profile = json.load(fh)
                bundle = prof.profile_bundle_for_td(profile)
                if bundle.get("camera_id") != int(cdir[1]):
                    raise HgError(
                        f"profile {path} camera_id "
                        f"{bundle.get('camera_id')} does not match "
                        f"directory {cdir}")
                params = _geometry().warp_parameters(bundle["P"])
            except HgError:
                raise
            except Exception as e:
                raise HgError(
                    f"profile {path} is malformed; fix or remove it: {e}")
            res = client.rpc({
                "action": "register", "profile": bundle,
                "warp_params": params,
                "provenance": os.path.abspath(path)})
            results.append((f"{cdir}/{sid}", res.get("status")))
    return results


def run_setup_td():
    client = TdClient()
    res = client.rpc({"action": "setup"})
    info = res.get("inspect", {})
    print("setup complete:")
    for name in ("hg_pattern", "hg_pattern_callbacks", "hg_depth_xyz"):
        node = info.get(name) or {}
        if node.get("exists"):
            print(f"  {name:22} exists type={node.get('type')} "
                  f"hg_owned={node.get('hg_owned')}")
    inputs = info.get("final_switch_inputs", [])
    if len(inputs) > TD_SWITCH_INPUT:
        print(f"  final_switch input {TD_SWITCH_INPUT} <- "
              f"{inputs[TD_SWITCH_INPUT]}")
    idx = info.get("final_switch_index", {})
    print(f"  final_switch.index unchanged: mode={idx.get('mode')} "
          f"val={idx.get('val')}")
    for label, status in _rehydrate_profiles(client):
        print(f"  profile {label:18} -> TD registry ({status})")
    return info


def _register_profile(client, profile, provenance):
    params = _geometry().warp_parameters(profile["P"])
    return client.rpc({
        "action": "register",
        "profile": _profiles().profile_bundle_for_td(profile),
        "warp_params": params,
        "provenance": provenance})


def run_warp(args, client=None):
    """Compatibility profile activation: optionally publish a legacy
    homography JSON into the shared profile first, then register the
    bundle and activate c{cam}/{surface}. Never shows a new source with
    old uniforms - a profile without accepted color correction gates
    black instead of applying geometry-only compensation."""
    cam_id = args.cameras[0]
    prof = _profiles()
    sid = args.surface_id
    provenance = None
    if args.calibration:
        path = args.calibration
        if not os.path.isfile(path):
            raise HgError(f"--calibration not found: {path}")
        try:
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError) as e:
            raise HgError(f"--calibration unreadable: {e}")
        try:
            params = _geometry().warp_parameters(doc)
        except ValueError as e:
            raise HgError(f"--calibration invalid for warp: {e}")
        if str(params["source_id"]) != str(cam_id):
            raise HgError(
                f"calibration is for camera {params['source_id']}; use "
                f"-c {params['source_id']} or a matching --calibration")
        profile, ppath = prof.publish_geometry(
            ROOT, doc, sid, cam_id)
        provenance = os.path.abspath(ppath)
        print(f"profile -> {ppath}")
    profile = prof.load_profile(ROOT, sid, cam_id)
    if profile is None:
        raise HgError(
            f"no geometry profile for c{cam_id}/{sid} - run "
            f".\\hg -c {cam_id} -p 0 -s_id {sid} first (or pass "
            "--calibration to publish a legacy artifact)")
    client = client or TdClient()
    _ensure_td(client)
    res = _register_profile(
        client, profile,
        provenance or "profile %s" % prof.profile_path(
            ROOT, sid, cam_id))
    act = client.rpc({"action": "activate", "camera_id": cam_id,
                      "surface_id": sid, "select": True})
    ready = act.get("ready")
    route = (f"{CAM_SWITCH_TOP} -> {TD_ROOT}/hg_warp -> "
             f"{TD_ROOT}/hg_table_color -> {TD_ROOT}/hg_output -> "
             f"{TD_OART} -> {TD_HG} -> "
             f"{TD_SWITCH}[{TD_SWITCH_INPUT}] (selected)")
    if ready:
        print(f"profile active: {route}")
    else:
        print(f"gate black: {route}")
        print("profile has no accepted color correction - run "
              f".\\cc -c {cam_id} -p 0 -s_id {sid} to complete it")
    print(f"calibration profile: c{cam_id}/{sid} "
          f"(registry {res.get('status')})")


def run_migrate_pipe():
    """Reversibly copy the legacy out/calibration store into
    tools/pipe. Byte-exact; conflicting destinations are refused and
    reported, never overwritten."""
    prof = _profiles()
    rep = prof.migrate_legacy_profiles(ROOT)
    if not any(rep[k] for k in rep):
        print("nothing to migrate - no legacy "
              + prof.legacy_calibration_root(ROOT))
    for key in ("copied", "skipped", "refused", "ignored"):
        for rel in rep[key]:
            print(f"  {key:8} {rel}")
    if rep["refused"]:
        raise HgError("migration refused %d file(s): %s - resolve "
                      "manually; nothing was overwritten"
                      % (len(rep["refused"]),
                         ", ".join(rep["refused"])))
    if rep["copied"]:
        print("pipe store: " + prof.calibration_root(ROOT))
        print("legacy tree kept as read-only backup: "
              + prof.legacy_calibration_root(ROOT))


def run_init_profile(args, client=None):
    """Publish a legacy homography JSON into the shared profile store
    without any TD contact (idempotent: a matching geometry hash reuses
    the current profile instead of writing a new revision)."""
    cam_id = args.cameras[0]
    prof = _profiles()
    path = args.init_profile
    if not os.path.isfile(path):
        raise HgError(f"--init-profile file not found: {path}")
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as e:
        raise HgError(f"--init-profile unreadable: {e}")
    try:
        params = _geometry().warp_parameters(doc)
    except ValueError as e:
        raise HgError(f"--init-profile invalid: {e}")
    if str(params["source_id"]) != str(cam_id):
        raise HgError(
            f"artifact is for camera {params['source_id']}; use "
            f"-c {params['source_id']}")
    sha = prof.geometry_sha256(doc)
    existing = prof.load_profile(ROOT, args.surface_id, cam_id)
    if existing and existing.get("geometry_sha256") == sha:
        print("profile already current for this geometry: "
              + prof.profile_path(ROOT, args.surface_id, cam_id))
        return
    profile, ppath = prof.publish_geometry(
        ROOT, doc, args.surface_id, cam_id)
    print(f"profile -> {ppath} (not registered; "
          "run .\\hg --warp or .\\cc to activate)")



def run_table_color(args, client=None):
    """Compatibility alias for `cc --load`: bind a reviewed model to the
    shared profile and register it. No color math lives in hg."""
    import cc as _cc
    client = client or TdClient()
    _ensure_td(client)
    _cc.run_load_path(args.table_color, args.cameras[0],
                      args.surface_id, client=client)
    print("model registered via shared profile; activate with "
          f".\\hg --warp -c {args.cameras[0]} -p 0 "
          f"-s_id {args.surface_id}")


def run_restore_td():
    client = TdClient()
    res = client.rpc({"action": "restore"})
    print("restored previous session state:")
    print(json.dumps(res.get("restored"), indent=2))


def _ensure_td(client):
    """Idempotent setup + inspect for every live path. setup is
    read-only preflight + owned-helper refresh; it refuses an active
    session and never changes output selection."""
    res = client.rpc({"action": "setup"})
    info = res.get("inspect") or client.inspect()
    if info.get("protocol_version") != PROTOCOL_VERSION:
        raise HgError(
            "TD helper protocol v%r, expected v%d - "
            "tools/td/projective_td.py deployment is stale"
            % (info.get("protocol_version"), PROTOCOL_VERSION))
    _rehydrate_profiles(client)     # restore the registry from disk
    return info


def _td_depth_mm(client):
    try:
        return _geometry().td_pointcloud_depth_mm(
            client.capture_depth_rgba())
    except ValueError as e:
        raise HgError(str(e))


def _projector_entry_td(info):
    raster = _td_raster(info, "OART")
    return _device_entry(
        PROJECTORS[0], raster,
        extra={"td_branch": {"pattern_top": TD_ROOT + "/hg_pattern",
                             "oart": TD_OART, "hg": TD_HG,
                             "switch": TD_SWITCH,
                             "switch_input": TD_SWITCH_INPUT,
                             "raster_source": "TD output raster, not an "
                                              "assumed HDMI mode"}}), raster


def _timeout_message(frame, known_corners, camera_id, surface_id,
                     projector_size, timeout):
    """Detection-timeout error text; persists the last failed frame
    through the hg_diagnostics helper when possible. A diagnostic
    IO/encode failure only appends a note - it never masks the
    timeout or the session restore in the caller's finally."""
    msg = ("fewer than 4 known markers detected within "
           f"{timeout:.0f}s - aim the camera at the projected pattern "
           "on a flat stationary surface")
    try:
        summary = _diagnostics().save_detection_failure(
            frame, known_corners, camera_id, surface_id,
            projector_size, ROOT)
        msg += ("; detected %d/%d known markers; frame saved %s; "
                "full-frame near-white fraction %.3f; inspect "
                "framing/occlusion and camera exposure/ambient light"
                % (summary["known_count"], summary["expected_count"],
                   summary["frame_path"],
                   summary["near_white_fraction"]))
    except Exception as e:
        msg += "; diagnostic unavailable: %s" % e
    return msg


def run_live(args, client=None, settle=SETTLE_S,
             timeout=CAPTURE_TIMEOUT_S):
    cam_id = args.cameras[0]
    if cam_id == 2:
        return run_depth_to_projector(args, client=client)
    g = _geometry()
    client = client or TdClient()
    info = _ensure_td(client)
    destination, proj_raster = _projector_entry_td(info)
    top = CAMERA_TOPS[cam_id]
    cam_res = _td_raster(info, top.rsplit("/", 1)[-1])
    source_extra = {"capture_backend": "touchdesigner", "top_path": top}
    if cam_id == 1:
        node = info.get("kinectC", {})
        if node.get("image") != "color":
            raise HgError("kinectC image is not 'color'; refusing "
                          "unknown raster")
        if node.get("mirrorimage") is not False:
            raise HgError("kinectC is mirrored; raw pixel space "
                          "assumption broken")
        source_extra["mirrorimage"] = False
        source_extra["native_color_mapping"] = {
            "model": "full_frame_resize",
            "native_resolution": TD_NATIVE_COLOR,
            "td_resolution": cam_res}
    pattern, proj_corners = build_pattern(*proj_raster)
    owner = uuid.uuid4().hex
    attempted = False
    try:
        attempted = True          # end is attempted even if the begin
        client.rpc({"action": "begin", "owner": owner,  # ack is lost
                    "camera_id": cam_id,
                    "surface_id": args.surface_id,
                    "pattern": _wire().encode_array(pattern)})
        deadline = time.time() + settle + timeout
        settle_until = time.time() + settle
        detect_result = None
        while True:
            frame = client.capture_color(CAM_SWITCH_TOP)
            cam_res = [int(frame.shape[1]), int(frame.shape[0])]
            if cam_id == 1:
                source_extra["native_color_mapping"][
                    "td_resolution"] = cam_res
            if time.time() >= settle_until:
                detect_result = detect_pattern(frame, proj_corners)
                if detect_result is not None:
                    break
            if time.time() > deadline:
                raise HgError(_timeout_message(
                    frame, proj_corners, cam_id, args.surface_id,
                    proj_raster, timeout))
            time.sleep(RETRY_S)
        cam_pts, proj_pts = detect_result
        try:
            H, H_inv, diag = g.fit_homography(cam_pts, proj_pts)
        except ValueError as e:
            raise HgError(str(e))
    finally:
        if attempted:
            try:
                client.rpc({"action": "end", "owner": owner})
            except HgError as e:
                if "no active" in str(e):
                    pass
                else:
                    raise HgError(
                        f"TD restore failed: {e} - fix the cause and "
                        "run .\\hg --restore-td (snapshot retained in TD "
                        "storage)") from e
    source = _device_entry(
        CAMERAS[cam_id], cam_res, extra=source_extra)
    out = args.output or default_output(
        "ordinary", args.cameras, args.projector, args.surface_id)
    doc = save_homography(
        out, source, destination, H, H_inv, diag, args.plane,
        warnings=["camera feed is the TD node raster; TD-side "
                  "crops/flips/resizes would change this space",
                  "no independent validation performed"])
    # Publish the shared per-camera/surface profile (new geometry
    # revision; any prior photometric model is invalidated), then push
    # the bundle to the TD registry over the wire.
    prof = _profiles()
    profile, profile_path = prof.publish_geometry(
        ROOT, doc, args.surface_id, cam_id)
    print(f"profile -> {profile_path}")
    try:
        res = client.rpc({
            "action": "register",
            "profile": prof.profile_bundle_for_td(profile),
            "warp_params": g.warp_parameters(doc),
            "provenance": os.path.abspath(profile_path)})
    except HgError as e:
        raise HgError(
            f"saved {out} and {profile_path} but TD load failed: {e}")
    print(f"registered c{cam_id}/{args.surface_id} in TD "
          f"({res.get('status')})")
    print(f"saved -> {out}")
    print(f"fit: {diag['inlier_count']}/{diag['point_count']} inliers, "
          f"mean {diag['inlier_mean_destination_px']:.3f}px, "
          f"max {diag['inlier_max_destination_px']:.3f}px")


# ----------------------------------------------------------- Kinect maps

def _sdk_map(kin, depth, source_cam, dest_size):
    """Native SDK CoordinateMapper forward XY lookup -> float32 [H,W,2]."""
    kincap = _kin()
    np = _np()
    from pykinect2 import PyKinectV2
    DW, DH = kincap.DW, kincap.DH
    CW, CH = kincap.CW, kincap.CH
    depth_ptr = depth.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
    if source_cam == 1:                     # color -> depth
        pts = (PyKinectV2._DepthSpacePoint * (CW * CH))()
        kin.mapper.MapColorFrameToDepthSpace(
            ctypes.c_uint(DW * DH), depth_ptr, ctypes.c_uint(CW * CH), pts)
        return np.frombuffer(pts, np.float32).reshape(CH, CW, 2).copy()
    pts = (PyKinectV2._ColorSpacePoint * (DW * DH))()
    kin.mapper.MapDepthFrameToColorSpace(
        ctypes.c_uint(DW * DH), depth_ptr, ctypes.c_uint(DW * DH), pts)
    return np.frombuffer(pts, np.float32).reshape(DH, DW, 2).copy()


def run_map(args, client=None):
    s, d = args.cameras
    g = _geometry()
    client = client or TdClient()
    _ensure_td(client)
    kin = _open_mapper()
    try:
        depth = _td_depth_mm(client)
        raw = _sdk_map(kin, depth, s, CAMERAS[d]["resolution"])
        xy, valid = g.sanitize_sdk_map(raw, depth, s,
                                       CAMERAS[d]["resolution"])
    finally:
        _close_kin(kin)
    source = _device_entry(
        CAMERAS[s], extra={"depth_source": "TD " + DEPTH_TOP})
    destination = _device_entry(CAMERAS[d])
    out = args.output or default_output("map", args.cameras)
    doc, npz_path = save_dense_map(
        out, source, destination, xy, valid, depth,
        {"diagnostics": {
            "valid_fraction": float(valid.mean()),
            "independent_validation": False},
         "depth_source": {"top_path": DEPTH_TOP,
                          "conversion": "hg_geometry.td_pointcloud_depth_mm",
                          "orientation": "TD bottom-up RGBA -> top-down "
                                         "uint16 mm"},
         "td_raster_note": ("emitted coordinates are native SDK "
                            "(color 1920x1080 / depth 512x424); TD color "
                            "rasters are 1280x720 and need explicit "
                            "pixel-centre rescaling"),
         "warnings": [
            "snapshot only: the SDK used the observed depth frame; "
            "regenerate when scene or depth changes",
            "not a reusable calibration and not a homography",
            VALID_MASK_NOTE]},
        sdk_space=True)
    print(f"saved -> {out}")
    print(f"map   -> {npz_path} "
          f"({int(valid.sum())}/{valid.size} valid, "
          f"{100.0 * valid.mean():.1f}%)")


def _load_via(path):
    if not path or not os.path.isfile(path):
        raise HgError(
            "--via requires an existing c1->P0 homography JSON; "
            "calibrate first, e.g.  .\\hg -c 1 -p 0")
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as e:
        raise HgError(f"--via unreadable: {e}")
    if not isinstance(doc, dict):
        raise HgError("--via root must be a JSON object")
    try:
        src, dst = doc["source"], doc["destination"]
        g = _geometry()
        src_res = g.image_size(src["resolution"])
        dest_res = g.image_size(dst["resolution"])
        ok = (doc.get("schema") == SCHEMA
              and doc.get("schema_version") == SCHEMA_VERSION
              and doc.get("kind") == "homography"
              and isinstance(src, dict) and isinstance(dst, dict)
              and src.get("id") == "1" and src.get("type") == "camera"
              and dst.get("id") == "P0" and dst.get("type") == "projector"
              and list(src_res) in TD_KNOWN_COLOR_RASTERS
              and doc.get("source_space") == SOURCE_SPACE
              and doc.get("lens_policy") == LENS_POLICY
              and src.get("mirrorimage") is not True
              and _via_resize_metadata_ok(src, src_res))
    except (KeyError, TypeError, ValueError):
        ok = False
        src_res = dest_res = None
    if not ok:
        raise HgError(
            "--via must be a c1 -> P0 homography artifact in raw pixel "
            "space (schema ff.hg v1, source id 1 camera at a known 16:9 "
            "TD/native raster, destination id P0 projector, "
            "planar_approximation policy; unknown crop/mirror spaces "
            "refused)")
    np = _np()
    H = np.asarray(doc.get("H_source_to_destination"), np.float64)
    if (H.shape != (3, 3) or not np.isfinite(H).all()
            or np.linalg.matrix_rank(H) != 3):
        raise HgError("--via homography is malformed, non-finite or "
                      "singular")
    H_native = _geometry().resized_source_homography(
        H, TD_NATIVE_COLOR, src_res)
    return path, doc, H, H_native, list(dest_res), list(src_res)


def _via_resize_metadata_ok(src, src_res):
    """Metadata policy for --via sources; no dimension guessing.

    TD kinectC sources need the recorded raw/mirror/full-frame-resize
    fields; offline/native 1920x1080 passes; offline 1280x720 needs an
    explicit full_frame_resize declaration."""
    backend = src.get("capture_backend")
    ncm = src.get("native_color_mapping")
    ncm_ok = (isinstance(ncm, dict)
              and ncm.get("model") == "full_frame_resize"
              and list(ncm.get("native_resolution") or [])
              == TD_NATIVE_COLOR
              and list(ncm.get("td_resolution") or []) == list(src_res))
    if backend == "touchdesigner":
        return (src.get("top_path") == "/project1/kinectC"
                and src.get("mirrorimage") is False and ncm_ok)
    if tuple(src_res) == tuple(TD_NATIVE_COLOR):
        return True
    return ncm_ok                       # offline 1280x720: explicit only


def _transform_safe(g, H, pts):
    """transform_points in chunks so a few horizon points don't kill all."""
    np = _np()
    out = np.full((len(pts), 2), np.nan)
    step = 4096
    for a in range(0, len(pts), step):
        chunk = pts[a:a + step]
        try:
            out[a:a + step] = g.transform_points(H, chunk)
            continue
        except ValueError:
            pass
        for i, p in enumerate(chunk):
            try:
                out[a + i] = g.transform_points(H, [p])[0]
            except ValueError:
                pass
    return out


def run_depth_to_projector(args, client=None):
    np = _np()
    g = _geometry()
    via_path, via_doc, H, H_native, proj_res, src_res = _load_via(args.via)
    client = client or TdClient()
    _ensure_td(client)
    kin = _open_mapper()
    try:
        depth = _td_depth_mm(client)
        raw = _sdk_map(kin, depth, 2, TD_NATIVE_COLOR)
        xy, valid = g.sanitize_sdk_map(raw, depth, 2, TD_NATIVE_COLOR)
    finally:
        _close_kin(kin)
    # Compose: depth px -> native color px -> projector px.
    ys, xs = np.nonzero(valid)
    color_pts = xy[ys, xs]
    proj = _transform_safe(g, H_native, color_pts)
    pw, ph = proj_res
    good = (np.isfinite(proj).all(axis=1)
            & (proj[:, 0] >= 0) & (proj[:, 0] < pw)
            & (proj[:, 1] >= 0) & (proj[:, 1] < ph))
    dense = np.full(xy.shape, -1, np.float32)
    new_valid = np.zeros(valid.shape, bool)
    dense[ys[good], xs[good]] = proj[good]
    new_valid[ys[good], xs[good]] = True
    source = _device_entry(CAMERAS[2],
                           extra={"depth_source": "TD " + DEPTH_TOP})
    destination = _device_entry(PROJECTORS[args.projector], proj_res)
    out = args.output or default_output(
        "dense", args.cameras, args.projector, args.surface_id)
    doc, npz_path = save_dense_map(
        out, source, destination, dense, new_valid, depth,
        {"provenance": {
            "via": os.path.abspath(via_path),
            "via_created_utc": via_doc.get("created_utc"),
            "via_source_resolution": src_res,
            "fullframe_resize_assumption": tuple(src_res)
                != tuple(TD_NATIVE_COLOR),
            "homography_copy": {
                "H_source_to_destination": H.tolist(),
                "H_native_1920x1080_to_destination": H_native.tolist(),
                "source": via_doc["source"],
                "destination": via_doc["destination"],
                "coordinate_convention": via_doc.get(
                    "coordinate_convention"),
                "source_space": via_doc.get("source_space"),
                "lens_policy": via_doc.get("lens_policy"),
                "plane": via_doc.get("plane"),
                "diagnostics": via_doc.get("diagnostics"),
            }},
         "warnings": [
            "composed snapshot: depth->native-color SDK map frozen through "
            "a copy of the via c1->P0 homography; not a reusable "
            "calibration",
            "ONLY points on the via calibration plane are geometrically "
            "registered; observed depth elsewhere does not make the "
            "homography a 3D warp",
            VALID_MASK_NOTE,
            "native SDK top-left pixel convention; not mirrored/TD space"]})
    print(f"saved -> {out}")
    print(f"map   -> {npz_path} "
          f"({int(new_valid.sum())}/{new_valid.size} valid)")


# -------------------------------------------------------------------- list

def run_list():
    print("logical devices (live frames via TouchDesigner):")
    for cid, cam in CAMERAS.items():
        res = cam["resolution"] or "actual TD raster"
        print(f"  c{cid}  {cam['name']:45} native={res}")
    proj = PROJECTORS[0]
    print(f"  p{0}   {proj['name']:45} via {proj['image_path']}")
    try:
        info = TdClient().inspect()
    except HgError as e:
        print(f"TD bridge unavailable ({e}); showing static registry only")
        return
    print("TD nodes:")
    for name in ("camLogi", "kinectC", "kinectD", "hg_depth_xyz",
                 "OART", "hg", "final_switch"):
        node = info.get(name) or {}
        if not node.get("exists"):
            print(f"  {name:18} MISSING")
            continue
        res = (f"{node.get('width')}x{node.get('height')}"
               if node.get("width") else "-")
        print(f"  {name:18} {str(node.get('type')):14} {res:10} "
              f"inputs={node.get('inputs')}")
    idx = info.get("final_switch_index", {})
    print(f"  final_switch.index: mode={idx.get('mode')} "
          f"val={idx.get('val')} expr={idx.get('expr')!r}")
    mode = info.get("output_mode", {})
    if mode:
        print(f"  output_mode.Mode: {mode.get('val')} "
              f"(mode={mode.get('mode')})")


# -------------------------------------------------------------------- CLI

def build_parser():
    p = argparse.ArgumentParser(
        prog="hg",
        description="Homography / device-map utility over the TD camera "
                    "branch (see tools/hg_README.md).")
    p.add_argument("-c", "--camera", dest="cameras", type=int,
                   action="append", metavar="ID",
                   help="logical camera id (0 camLogi, 1 kinectC, "
                        "2 depth); repeat for --map")
    p.add_argument("-p", "--projector", type=int, metavar="ID",
                   help="logical projector index (0); the TD hg branch, "
                        "NOT a monitor index")
    p.add_argument("--map", action="store_true",
                   help="dense SDK map; requires exactly two -c, only "
                        "1->2 or 2->1, no -p")
    p.add_argument("--points", metavar="JSON",
                   help="offline correspondences for c0/c1 -> P0: "
                        "{camera_size, projector_size, camera_points, "
                        "projector_points}; no hardware/TD access")
    p.add_argument("--via", metavar="JSON",
                   help="existing c1->P0 homography artifact; required for "
                        "-c 2 -p 0")
    p.add_argument("--setup-td", action="store_true",
                   help="create hg helper nodes and wire final_switch "
                        "input 7 <- hg; changes no selection")
    p.add_argument("--restore-td", action="store_true",
                   help="restore a stored session left by a killed run")
    p.add_argument("--warp", action="store_true",
                   help="route raw camera through hg_warp (inverse-UV "
                        "GLSL) into OART and select final_switch 7")
    p.add_argument("--calibration", metavar="JSON",
                   help="homography artifact for --warp: published into "
                        "the shared profile before activation")
    p.add_argument("--init-profile", metavar="JSON",
                   help="publish a legacy homography JSON into the "
                        "shared profile store only (no TD contact)")
    p.add_argument("--table-color", metavar="JSON",
                   help="compat alias for 'cc --load': bind a reviewed "
                        "ff.hg.table_color model to the shared profile")
    p.add_argument("-s_id", "--surface-id", "--surface",
                   dest="surface_id", default="Table_01",
                   metavar="ID",
                   help="named calibration surface (default Table_01); "
                        "profiles live under tools/pipe/p0/<id>/")
    p.add_argument("--monitor", type=int, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--webcam-index", type=int, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("-o", "--output", metavar="JSON",
                   help="export path (default "
                        "tools/pipe/p0/<surface>/c<id>/P.json or "
                        "tools/pipe/maps/c{s}_c{d}.json)")
    p.add_argument("--plane", default="unspecified_flat_surface",
                   help="label for the measured plane")
    p.add_argument("--list", action="store_true",
                   help="list logical devices and TD branch state; "
                        "activates nothing")
    p.add_argument("--migrate-pipe", action="store_true",
                   help="copy legacy out/calibration/p0 profiles into "
                        "tools/pipe/p0 byte-exact (read-only backup "
                        "kept; conflicting files are refused)")
    return p


def validate(args):
    cams = args.cameras or []
    try:
        args.surface_id = _profiles().validate_surface_id(
            args.surface_id)
    except Exception as e:
        raise HgError(str(e))
    if args.list:
        return "list"
    if args.setup_td:
        if cams or args.projector is not None:
            raise HgError("--setup-td takes no -c/-p")
        return "setup_td"
    if args.restore_td:
        return "restore_td"
    if args.migrate_pipe:
        if cams or args.projector is not None or args.warp \
                or args.map or args.points or args.via \
                or args.table_color or args.init_profile \
                or args.output or args.calibration:
            raise HgError("--migrate-pipe takes no other arguments")
        return "migrate_pipe"
    if args.monitor is not None:
        raise HgError("--monitor was removed: hg now projects through the "
                      "TD branch (OART -> hg -> final_switch input 7); "
                      "see tools/hg/README.md")
    if args.webcam_index is not None:
        raise HgError("--webcam-index was removed: c0 is the TD node "
                      "/project1/camLogi, not a DirectShow index")
    if args.calibration and not args.warp:
        raise HgError("--calibration applies only to --warp")
    if args.init_profile:
        if args.warp or args.map or args.points or args.via \
                or args.table_color:
            raise HgError("--init-profile takes no other operations")
        if len(cams) != 1 or cams[0] not in (0, 1):
            raise HgError("--init-profile requires -c 0 or -c 1")
        if args.projector != 0:
            raise HgError("--init-profile requires -p 0")
        return "init_profile"
    if args.warp:
        if args.map or args.points or args.via or args.table_color:
            raise HgError("--warp takes no --map/--points/--via/"
                          "--table-color")
        if len(cams) != 1 or cams[0] not in (0, 1):
            raise HgError("--warp requires exactly one -c 0 or -c 1")
        if args.projector != 0:
            raise HgError("--warp requires -p 0")
        return "warp"
    if args.table_color:
        if args.map or args.points or args.via:
            raise HgError("--table-color takes no --map/--points/--via")
        if len(cams) != 1 or cams[0] != 0:
            raise HgError("--table-color requires -c 0 (camLogi model)")
        if args.projector != 0:
            raise HgError("--table-color requires -p 0")
        return "table_color"
    for c in cams:
        if c not in CAMERAS:
            raise HgError(
                f"unknown camera {c} (valid: {sorted(CAMERAS)})")
    if args.map:
        if len(cams) != 2:
            raise HgError("--map requires exactly two -c ids")
        if sorted(cams) != [1, 2]:
            raise HgError("--map supports only -c 1 -c 2 or -c 2 -c 1")
        if args.projector is not None:
            raise HgError("--map takes no -p")
        if args.points or args.via:
            raise HgError("--map takes no --points/--via")
        return "map"
    if len(cams) != 1:
        raise HgError("ordinary mode requires exactly one -c")
    if args.projector is None:
        raise HgError("ordinary mode requires -p (use -p 0)")
    if args.projector != 0:
        raise HgError(f"unknown projector {args.projector} (only p0)")
    if args.via:
        if cams[0] != 2:
            raise HgError("--via applies only to -c 2 -p 0")
        if args.points:
            raise HgError("--via cannot combine with --points")
        return "live"
    if cams[0] == 2:
        raise HgError(
            "camera 2 (depth) cannot see a projected pattern; use "
            "--via <c1_p0.json> - e.g. first run .\\hg -c 1 -p 0")
    if args.points:
        return "points"
    return "live"


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        mode = validate(args)
    except HgError as e:
        print(f"hg: error: {e}", file=sys.stderr)
        return 2
    try:
        if mode == "list":
            run_list()
        elif mode == "setup_td":
            run_setup_td()
        elif mode == "restore_td":
            run_restore_td()
        elif mode == "migrate_pipe":
            run_migrate_pipe()
        elif mode == "warp":
            run_warp(args)
        elif mode == "init_profile":
            run_init_profile(args)
        elif mode == "table_color":
            run_table_color(args)
        elif mode == "points":
            run_points(args)
        elif mode == "map":
            run_map(args)
        else:
            run_live(args)
    except HgError as e:
        print(f"hg: error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("hg: interrupted", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"hg: error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
