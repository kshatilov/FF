"""cc_td.py - TD-side table-color install for the projective branch.

runpy-loaded by tools/td/projective_td.py with that module's globals in
scope (op, ParMode, _require, _storage_*, _create_owned, _has_par,
_first_par, TABLE_COLOR_*, WARP_TOP, ROOT, RpcError). Contains the model
validator, node/controls creation and uniform application moved out of
the old hg_td.py - no color math lives here beyond the settled
gain*headroom composite written verbatim into uniforms.
"""
import os


def validate_color_model(model):
    """Strict ff.hg.table_color v1 gate. Returns parsed fields."""
    import math
    try:
        ok = isinstance(model, dict)
        ok = ok and model.get("schema") == "ff.hg.table_color"
        ok = ok and model.get("schema_version") == 1
        ok = ok and (model.get("method")
                     == "global_attenuation_gray_balance")
        source_id = str(model["source_id"])
        ok = ok and source_id in ("0", "1")
        ok = ok and model.get("source_top") == (
            "/project1/camLogi" if source_id == "0"
            else "/project1/kinectC")
        sizes = (list(model["source_size"])
                 + list(model["destination_size"]))
        ok = ok and all(isinstance(v, (int, float))
                        and math.isfinite(v) and v > 0
                        and float(v).is_integer() for v in sizes)
        gains = [float(v) for v in model["gain_rgb"]]
        headroom = float(model["headroom"])
        ok = ok and len(gains) == 3 and all(map(math.isfinite, gains))
        ok = ok and all(0.0 <= v <= 1.0 for v in gains)
        ok = ok and math.isfinite(headroom) and 0.0 < headroom <= 1.0
    except (KeyError, TypeError, ValueError, IndexError):
        ok = False
    if not ok:
        raise RpcError("table_color model malformed")
    if model.get("accepted") is not True:
        raise RpcError("table_color model has no top-level "
                       "accepted=true decision")
    return {"source_id": source_id, "gains": gains,
            "headroom": headroom,
            "src_size": [int(sizes[0]), int(sizes[1])],
            "dst_size": [int(sizes[2]), int(sizes[3])]}


def ensure_color_nodes():
    """Create/reuse the owned color GLSL, shader DAT and controls COMP."""
    base = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(base, "hg_table_color.frag"),
                  encoding="utf-8") as fh:
            src_text = fh.read()
    except OSError as e:
        raise RpcError("cannot read hg_table_color.frag: " + str(e))
    root = op(ROOT)
    sh = _create_owned(
        root, textDAT, "hg_table_color_shader", "textdat",
        TABLE_COLOR_SHADER + " exists and is not hg-owned")
    color = _create_owned(
        root, glslTOP, "hg_table_color", "glsltop",
        TABLE_COLOR_TOP + " exists and is not hg-owned")
    ctl = _create_owned(
        root, baseCOMP, "hg_table_color_controls", "basecomp",
        TABLE_COLOR_CONTROLS + " exists and is not hg-owned")
    # user-adjustable controls; defaults only for pars that are absent,
    # created independently so a partial page is repaired correctly
    if not (_has_par(ctl, "Enabled") and _has_par(ctl, "Strength")):
        page = ctl.appendCustomPage("Table Color")
        if not _has_par(ctl, "Enabled"):
            _first_par(page.appendToggle(
                "Enabled", label="Enabled")).val = True
        if not _has_par(ctl, "Strength"):
            st = _first_par(page.appendFloat(
                "Strength", label="Strength"))
            st.val = 1.0
            try:
                st.normMin, st.normMax = 0.0, 1.0
            except Exception:
                pass
    sh.text = src_text
    color.par.pixeldat = "hg_table_color_shader"
    return color, sh, ctl


def install_color_model(model):
    """Ensure color nodes and apply a validated model's uniforms.

    Returns (color_top, controls_comp). Does not connect the input,
    cook or route; callers do that before exposing output."""
    check = validate_color_model(model)
    color, _sh, ctl = ensure_color_nodes()
    color.par.outputresolution = "custom"
    color.par.resolutionw, color.par.resolutionh = check["dst_size"]
    composite = [float(g) * check["headroom"]
                 for g in check["gains"]]
    setattr(color.par, "vec0name", "uTableGain")
    for axis, val in zip("xyz", composite):
        setattr(color.par, "vec0value%s" % axis, val)
    setattr(color.par, "vec1name", "uTableOptions")
    setattr(color.par, "vec1valuex", 1.0)
    ox = color.par.vec1valuex
    try:
        ox.mode = ParMode.EXPRESSION
    except Exception:
        pass
    try:
        ox.expr = ("op('%s').par.Enabled.eval() * "
                   "op('%s').par.Strength.eval()"
                   % (TABLE_COLOR_CONTROLS, TABLE_COLOR_CONTROLS))
    except Exception:
        raise RpcError("cannot bind uTableOptions expression")
    return color, ctl
