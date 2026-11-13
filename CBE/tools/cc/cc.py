"""cc.py - table color calibration / model registration CLI.

    cc -c 0 -p 0 -s_id Table_01            measure + publish a model
    cc --load model.json -c 0 -p 0 -s_id Table_01
                                           register a reviewed model

The measurement loop and acceptance gate live in
tools/cc/measurement.py; cc only resolves profiles, stages inputs and
publishes results. Profiles are shared with hg under
tools/pipe/p0/<surface>/c<id>/.
"""
import argparse
import datetime
import json
import os
import sys
import tempfile
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, os.path.join(TOOLS_DIR, "hg"))
sys.path.insert(0, os.path.join(TOOLS_DIR, "td"))
sys.path.insert(0, HERE)

import hg                                # noqa: E402
import hg_geometry                       # noqa: E402
import projective_profiles as profiles   # noqa: E402

CAMERA_TOPS = {0: "/project1/camLogi", 1: "/project1/kinectC"}


class CcError(Exception):
    """User-facing failure; printed without a traceback."""


def _wrap(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except profiles.ProfileError as e:
        raise CcError(str(e))
    except hg.HgError as e:
        raise CcError(str(e))


def _staging_calibration(staging_dir, P):
    """Write the profile's P doc verbatim as the calibration staging
    file measurement.calibrate reads (same artifact bytes, parsed).
    Must live OUTSIDE the measurement directory: Measurement.__init__
    owns mkdir(exist_ok=False) of it."""
    path = os.path.join(staging_dir, "calibration_staging.json")
    hg._write_json_atomic(path, P)
    return path


def run_measure(args, client=None):
    camera_id = args.cameras[0]
    sid = args.surface_id
    profile = _wrap(profiles.load_profile, ROOT, sid, camera_id)
    if profile is None:
        raise CcError(
            "no geometry profile for c%d/%s - run "
            ".\\hg -c %d -p 0 -s_id %s first"
            % (camera_id, sid, camera_id, sid))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = args.output or os.path.join(
        ROOT, "out", "cc", sid, "c%d" % camera_id, stamp)
    client = client or hg.TdClient()
    import measurement
    # staging P copy lives in a private temp dir; Measurement creates
    # and owns `directory` itself.
    with tempfile.TemporaryDirectory(prefix="cc_staging_") as staging_dir:
        staging = _staging_calibration(staging_dir, profile["P"])
        try:
            model = measurement.calibrate(
                directory, calibration=staging, camera_id=camera_id,
                surface_id=sid, client=client)
        except (ValueError, hg.HgError, OSError) as e:
            # failed measurements never replace the shared profile
            raise CcError("measurement did not produce an accepted "
                          "model (profile unchanged): %s" % e)
    profile, ppath = _wrap(
        profiles.publish_color, ROOT, model, sid, camera_id,
        provenance=os.path.abspath(
            os.path.join(directory, "model.json")))
    _register_or_fail(client, profile, ppath)
    print(f"model   -> {os.path.join(directory, 'model.json')}")
    print(f"profile -> {ppath}")
    print("registered c%d/%s; the TD watcher now drives it via "
          "cam_and_kinect + hg_pair_controls automatically" % (camera_id, sid))


def run_load_path(path, camera_id, surface_id, client=None):
    """Bind a reviewed ff.hg.table_color JSON to the shared profile
    (legacy models missing destination_id/surface_id may attach to the
    explicitly requested pair only) and register it in TD."""
    if not os.path.isfile(path):
        raise CcError("model file not found: %s" % path)
    try:
        with open(path, encoding="utf-8") as fh:
            model = json.load(fh)
    except (OSError, ValueError) as e:
        raise CcError("model unreadable: %s" % e)
    profile, ppath = _wrap(
        profiles.publish_color, ROOT, model, surface_id, camera_id,
        allow_migrate=True, provenance=os.path.abspath(path))
    client = client or hg.TdClient()
    _register_or_fail(client, profile, ppath)
    print(f"profile -> {ppath}")


def _register_or_fail(client, profile, profile_path):
    """Push the validated bundle to TD; a valid file stays on disk and
    the failure is reported honestly instead of claimed success."""
    try:
        res = client.rpc({
            "action": "register",
            "profile": profiles.profile_bundle_for_td(profile),
            "warp_params": hg_geometry.warp_parameters(profile["P"]),
            "provenance": os.path.abspath(profile_path)})
    except hg.HgError as e:
        raise CcError("saved %s but TD load failed: %s"
                      % (profile_path, e))
    print("TD registry: %s (%s)" % (res.get("key"), res.get("status")))


def run_load(args, client=None):
    client = client or hg.TdClient()
    hg._ensure_td(client)
    run_load_path(args.load, args.cameras[0], args.surface_id,
                  client=client)
    print("model registered; the TD watcher applies it automatically "
          "when the matching camera/surface is selected "
          "(.\\hg --warp remains the manual activation path)")


def build_parser():
    p = argparse.ArgumentParser(
        prog="cc",
        description="Table color calibration over the shared projective "
                    "profile (see tools/cc/README.md).")
    p.add_argument("-c", "--camera", dest="cameras", type=int,
                   action="append", metavar="ID",
                   help="logical camera id (0 camLogi, 1 kinectC)")
    p.add_argument("-p", "--projector", type=int, metavar="ID",
                   help="logical projector index (0)")
    p.add_argument("-s_id", "--surface-id", "--surface",
                   dest="surface_id", default="Table_01",
                   metavar="ID",
                   help="named calibration surface (default Table_01)")
    p.add_argument("--load", metavar="JSON",
                   help="register an existing reviewed ff.hg.table_color"
                        " model without re-measuring")
    p.add_argument("-o", "--output", metavar="DIR",
                   help="measurement output directory (default "
                        "out/cc/<surface>/c<id>/<utc>)")
    p.add_argument("--list", action="store_true",
                   help="list profile paths; activates nothing")
    return p


def validate(args):
    cams = args.cameras or []
    try:
        args.surface_id = profiles.validate_surface_id(args.surface_id)
    except profiles.ProfileError as e:
        raise CcError(str(e))
    if args.list:
        return "list"
    for c in cams:
        if c == 2:
            raise CcError("camera 2 is depth; cc requires an RGB camera "
                          "(0 or 1) - use hg --map/--via for depth")
        if c not in (0, 1):
            raise CcError("unknown camera %r (valid: 0, 1)" % c)
    if len(cams) != 1:
        raise CcError("cc requires exactly one -c 0 or -c 1")
    if args.projector != 0:
        raise CcError("cc requires -p 0 (only p0)")
    return "load" if args.load else "measure"


def run_list():
    root = profiles.calibration_root(ROOT)
    if not os.path.isdir(root):
        print("no profiles yet - run .\\hg -c 0 -p 0 -s_id <surface>")
        return
    for sid in sorted(os.listdir(root)):
        sdir = os.path.join(root, sid)
        if not os.path.isdir(sdir):
            continue
        for cdir in sorted(os.listdir(sdir)):
            path = os.path.join(sdir, cdir, "profile.json")
            if not os.path.isfile(path):
                continue
            try:
                with open(path, encoding="utf-8") as fh:
                    prof = json.load(fh)
                tc = "TC" if prof.get("TC") else "geometry only"
                print(f"  {cdir}/{sid}: {tc} "
                      f"rev {str(prof.get('revision'))[:8]}")
            except (OSError, ValueError):
                print(f"  {cdir}/{sid}: unreadable profile")


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        mode = validate(args)
    except CcError as e:
        print(f"cc: error: {e}", file=sys.stderr)
        return 2
    try:
        if mode == "list":
            run_list()
        elif mode == "load":
            run_load(args)
        else:
            run_measure(args)
    except CcError as e:
        print(f"cc: error: {e}", file=sys.stderr)
        return 1
    except hg.HgError as e:
        print(f"cc: error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("cc: interrupted", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"cc: error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
