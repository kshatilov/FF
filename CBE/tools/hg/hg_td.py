"""hg_td.py - TD-side geometric warp install for the projective branch.

runpy-loaded by tools/td/projective_td.py with that module's globals
in scope (op, ParMode, _require, _storage_*, _create_owned, CAM_SWITCH,
WARP_TOP, WARP_SHADER, ROOT, RpcError). No rpc entry here; projective_td
owns dispatch.
"""
import os


def install_warp(params):
    """Ensure the owned hg_warp GLSL TOP + shader, fed by the shared
    cam_and_kinect selector, with inverse-UV rows from params.

    Returns (warp_top, shader_dat). Does not cook, check errors or
    route; callers do that before exposing output."""
    rows = params["inverse_uv_rows"]
    dst_size = [int(params["destination_size"][0]),
                int(params["destination_size"][1])]
    base = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(base, "hg_warp.frag"),
                  encoding="utf-8") as fh:
            src_text = fh.read()
    except OSError as e:
        raise RpcError("cannot read hg_warp.frag: " + str(e))
    root = op(ROOT)
    sh = _create_owned(root, textDAT, "hg_warp_shader", "textdat",
                       WARP_SHADER + " exists and is not hg-owned")
    w = _create_owned(root, glslTOP, "hg_warp", "glsltop",
                      WARP_TOP + " exists and is not hg-owned")
    sh.text = src_text
    w.par.pixeldat = "hg_warp_shader"
    w.par.outputresolution = "custom"
    w.par.resolutionw, w.par.resolutionh = dst_size
    w.inputConnectors[0].connect(_require(CAM_SWITCH))
    for i, row in enumerate(rows):
        setattr(w.par, "vec%dname" % i, "uH%d" % i)
        for axis, val in zip("xyz", row):
            setattr(w.par, "vec%dvalue%s" % (i, axis), float(val))
    return w, sh
