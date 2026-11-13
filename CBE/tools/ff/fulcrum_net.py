"""fulcrum_net.py - printable A4 net for the ArUco frustum calibration fulcrum.

Generates tools/fulcrum/fulcrum_net.pdf - a single-page cross/star net of a
square frustum: print at 100%, cut solid lines, fold dashed, print side
OUT.  Each face carries a unique DICT_4X4_50 marker (top id10, sides
id11..14); marker corners are given in TL,TR,BR,BL order in the spec.

Also writes tools/pipe/ff/fulcrum_spec.json with the marker-corner 3D
coordinates in the fulcrum frame (origin = base centre, +z up; +y face =
id11, +x = id12, -y = id13, -x = id14).  The spec is generated from the
MEAS block - the physically measured built object (base 155, top 75,
height ~45, markers 55/50); the print params below only regenerate the
net itself.

Geometry: base 155mm, top 75mm, height 40mm -> faces tilt 45 deg, slant
height 56.6mm.  Net spans 188mm horizontally (~11mm side margins).  45 deg
is the compromise slope: a camera at 30-60 deg elevation sees every face
within ~15-35 deg of its normal, and the projector (steep) still
illuminates all faces.

Preview render of page 1 -> smoke/fulcrum_net_preview.png
"""
import json
import os

import cv2
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as rl_canvas

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.abspath(os.path.join(HERE, "..", "fulcrum"))
PDF = os.path.join(DOC, "fulcrum_net.pdf")
SPEC = os.path.abspath(os.path.join(
    HERE, "..", "pipe", "ff", "fulcrum_spec.json"))
PREVIEW = os.path.abspath(
    os.path.join(HERE, "..", "..", "smoke", "fulcrum_net_preview.png"))

# ---- nominal geometry (mm) --------------------------------------------
A_BASE = 155.0          # bottom edge (open)
B_TOP = 75.0            # top face edge
HEIGHT = 40.0           # vertical height
SLANT = float(np.hypot(HEIGHT, (A_BASE - B_TOP) / 2.0))   # 56.57
M_TOP = 60.0            # marker side, top face
M_SIDE = 52.0           # marker side, side faces
GAP = 2.0               # marker top gap below the top fold edge
TAB = 9.0               # glue tab max width on slant edges (tapers in wedge)
TAB_GAP = 2.0           # clearance between tab edge and neighbour trapezoid
FLAP = 10.0             # base flap height (+y and -y faces only)
DICT = cv2.aruco.DICT_4X4_50
ID_TOP, ID_N, ID_E, ID_S, ID_W = 10, 11, 12, 13, 14

PAGE_W, PAGE_H = A4[0] / mm, A4[1] / mm     # 210 x 297 in mm
CX, CY = 105.0, 148.5   # net centre (square centre) on page 1

# ---- measured physical parameters of the BUILT fulcrum -----------------
# user-measured on the printed+folded object - the spec/solver uses THESE,
# the print params above only regenerate the net.  h~42.5 vs the 40 design
# means the real face tilt is atan(42.5/40) ~ 46.7 deg, not 45 deg.
MEAS = {"base": 155.0, "top": 75.0, "height": 42.5,   # measured 42-43
        "m_top": 55.0, "m_side": 50.0,
        "gap": 4.0}     # marker gap below top fold - assumed (unmeasured)

# marker placements in NET coords (origin = square centre, y up):
# id -> (box (x0,y0,x1,y1), rot90-k, face)
_HB, _MS, _G = B_TOP / 2, M_SIDE / 2, GAP + M_SIDE
MARKERS = {
    ID_TOP: ((-M_TOP / 2, -M_TOP / 2, M_TOP / 2, M_TOP / 2), 0, "top"),
    ID_N: ((-_MS, _HB + GAP, _MS, _HB + _G), 2, "+y"),
    ID_E: ((_HB + GAP, -_MS, _HB + _G, _MS), 1, "+x"),
    ID_S: ((-_MS, -_HB - _G, _MS, -_HB - GAP), 0, "-y"),
    ID_W: ((-_HB - _G, -_MS, -_HB - GAP, _MS), 3, "-x"),
}
# page-coordinate corner that is the marker TL, per rot90-k


# ---- net geometry ------------------------------------------------------
def rot_pts(pts, k):
    """Rotate (x,y) points by k*90 deg CCW about origin."""
    R = {0: (1, 0, 0, 1), 1: (0, -1, 1, 0),
         2: (-1, 0, 0, -1), 3: (0, 1, -1, 0)}[k]
    return [(R[0] * x + R[1] * y, R[2] * x + R[3] * y) for x, y in pts]


def wedge_tab(P0, P1, M0, M1, inset=2.0):
    """Glue tab on edge P0->P1 (a trapezoid slant edge).  The wedge between
    this edge and the neighbour trapezoid's edge M0->M1 opens away from the
    shared square corner, so the tab must TAPER: its outer boundary follows
    the neighbour edge offset inwards by TAB_GAP, capped at TAB width.
    Returns quad [A0, A1, B1, B0] (edge A0->A1 is the fold)."""
    def cross2(a, b):
        return a[0] * b[1] - a[1] * b[0]
    P0, P1, M0, M1 = [np.float64(p) for p in (P0, P1, M0, M1)]
    e = (P1 - P0) / np.linalg.norm(P1 - P0)
    n = np.array([e[1], -e[0]])
    mid = (P0 + P1) / 2
    md = M1 - M0
    mlen = np.linalg.norm(md)
    # sign n toward the M-edge side (wedge interior)
    if cross2(md, mid + n - M0) / mlen < cross2(md, mid - M0) / mlen:
        n = -n
    # M edge shifted by TAB_GAP toward the wedge interior (P side)
    perp = np.array([-md[1], md[0]]) / mlen     # toward positive side
    side = np.sign(cross2(md, mid - M0) / mlen)
    Ms = M0 + TAB_GAP * side * perp
    A0, A1 = P0 + e * inset, P1 - e * inset
    pts = [A0, A1]
    for A in (A1, A0):                           # B1, B0 on shifted M line
        t = cross2(Ms - A, md) / cross2(n, md)
        pts.append(A + min(max(t, 0.0), TAB) * n)
    return [tuple(p) for p in pts]


def build_elements():
    """Return list of dicts: {pts, folds:set(edge idx), label?}.
    Net coords, origin = square centre.  Edge i = pts[i] -> pts[i+1]."""
    hb, ha = B_TOP / 2, A_BASE / 2
    els = [{"pts": [(-hb, -hb), (hb, -hb), (hb, hb), (-hb, hb)],
            "folds": {0, 1, 2, 3}, "id": "square"}]
    trap = [(-hb, hb), (hb, hb), (ha, hb + SLANT), (-ha, hb + SLANT)]
    flap = [(-ha + 2, hb + SLANT), (ha - 2, hb + SLANT),
            (ha - 12, hb + SLANT + FLAP), (-ha + 12, hb + SLANT + FLAP)]
    names = ("N", "W", "S", "E")                 # index = rot90 k
    traps = {nm: rot_pts(trap, k) for k, nm in enumerate(names)}
    for k, name in enumerate(names):
        pts = traps[name]
        nxt = traps[names[(k - 1) % 4]]          # clockwise neighbour
        tab = wedge_tab(pts[1], pts[2], nxt[3], nxt[0])
        folds = {0, 1}                           # square edge + tab edge
        if name in ("N", "S"):
            els.append({"pts": rot_pts(flap, k), "folds": {0},
                        "id": f"flap_{name}"})
            folds.add(2)
        els.append({"pts": pts, "folds": folds, "id": f"trap_{name}"})
        els.append({"pts": tab, "folds": {0}, "id": f"tab_{name}"})
    return els


# ---- marker images -----------------------------------------------------
def marker_img(i, side_mm, rot_k):
    px = int(round(side_mm * 12))               # ~12 px/mm
    dic = cv2.aruco.getPredefinedDictionary(DICT)
    try:
        m = dic.generateImageMarker(i, px)
    except (AttributeError, TypeError):
        m = cv2.aruco.generateImageMarker(dic, i, px)
    pad = px // 30
    m = np.pad(m, pad, constant_values=255)
    return np.rot90(m, rot_k), pad / (side_mm * 12) * side_mm


# ---- spec JSON ---------------------------------------------------------
def spec_corners():
    """Marker TL,TR,BR,BL in fulcrum 3D frame (base centre, z up).
    Built from the MEASURED folded geometry - face tilt is derived from
    (base-top)/2 inset and measured height, not assumed 45 deg."""
    z = np.array([0.0, 0.0, 1.0])
    inset = (MEAS["base"] - MEAS["top"]) / 2.0            # 40
    H = MEAS["height"]
    slant3d = float(np.hypot(H, inset))                  # 60.2
    out = {}
    m = MEAS["m_top"] / 2
    out[str(ID_TOP)] = [[-m, m, H], [m, m, H],
                        [m, -m, H], [-m, -m, H]]
    for mid, n_h in ((ID_N, np.array([0, 1, 0])),
                     (ID_E, np.array([1, 0, 0])),
                     (ID_S, np.array([0, -1, 0])),
                     (ID_W, np.array([-1, 0, 0]))):
        # in-plane up (base->top along the face) and outward normal
        u = np.array([-n_h[0] * inset, -n_h[1] * inset, H]) / slant3d
        nrm = np.array([n_h[0] * H, n_h[1] * H, inset]) / slant3d
        r = np.cross(z, nrm).astype(float)               # in-plane right
        r /= np.linalg.norm(r)
        T = np.array([n_h[0] * MEAS["top"] / 2,
                      n_h[1] * MEAS["top"] / 2, H])
        C = T - (MEAS["gap"] + MEAS["m_side"] / 2) * u
        c = MEAS["m_side"] / 2
        cs = [C - c * r + c * u, C + c * r + c * u,
              C + c * r - c * u, C - c * r - c * u]
        out[str(mid)] = [list(np.round(p, 2)) for p in cs]
    return out


def write_spec():
    inset = (MEAS["base"] - MEAS["top"]) / 2.0
    tilt = np.degrees(np.arctan2(MEAS["height"], inset))
    spec = {
        "units": "mm",
        "frame": ("origin = base centre, +z up. Side faces: id11=+y, "
                  "id12=+x, id13=-y, id14=-x, id10=top. Marker corners "
                  "in TL,TR,BR,BL order (matches cv2.aruco)."),
        "dict": "DICT_4X4_50",
        "measured": {"base_edge": MEAS["base"], "top_edge": MEAS["top"],
                     "height": MEAS["height"],
                     "face_tilt_deg": round(float(tilt), 1),
                     "marker_side_top": MEAS["m_top"],
                     "marker_side_faces": MEAS["m_side"],
                     "marker_gap_below_top_fold": MEAS["gap"]},
        "markers": {},
        "note": ("MEASURED on the built object (base 155, top 75, "
                 "height ~45, markers 55/50).  Face tilt derived from "
                 "measured geometry; 'gap' assumed 4mm (unmeasured)."),
    }
    corners = spec_corners()
    for mid, cs in corners.items():
        spec["markers"][mid] = {
            "face": MARKERS[int(mid)][2],
            "size_mm": (MEAS["m_top"] if int(mid) == ID_TOP
                        else MEAS["m_side"]),
            "corners_TL_TR_BR_BL": cs}
    with open(SPEC, "w") as fh:
        json.dump(spec, fh, indent=2)
    print("wrote", SPEC)


# ---- PDF rendering -----------------------------------------------------
def to_page(pts):
    return [(x + CX, y + CY) for x, y in pts]


def draw_elements(c):
    for el in build_elements():
        pts = el["pts"]
        n = len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            c.setDash(2, 2) if i in el["folds"] else c.setDash()
            c.line((a[0] + CX) * mm, (a[1] + CY) * mm,
                   (b[0] + CX) * mm, (b[1] + CY) * mm)
    c.setDash()


def draw_markers(c):
    for mid, (box, k, _) in MARKERS.items():
        img, pad_mm = marker_img(mid, box[2] - box[0], k)
        w = box[2] - box[0] + 2 * pad_mm
        c.drawImage(ImageReader(Image.fromarray(img)),
                    (box[0] - pad_mm + CX) * mm, (box[1] - pad_mm + CY) * mm,
                    w * mm, w * mm)


def draw_page1(c):
    # net only, ~11mm side margins, no text
    draw_elements(c)
    draw_markers(c)
    c.showPage()


def render_pdf():
    c = rl_canvas.Canvas(PDF, pagesize=A4)
    c.setLineWidth(0.4)
    draw_page1(c)
    c.save()
    print("wrote", PDF)


# ---- drawio export ------------------------------------------------------
DRAWIO = os.path.join(DOC, "fulcrum_net.drawio")


def write_drawio():
    """Editable diagrams.net version of the net.  Lines are mxCell edges
    (solid = cut, dashed = fold), markers are embedded PNG images - every
    piece is individually selectable/movable.
    drawio units: 100 px per inch, A4 page = 827 x 1169 px, y down."""
    import base64
    S = 100.0 / 25.4

    def px(x, y):                          # net coords -> drawio px
        return (x + CX) * S, (PAGE_H - y - CY) * S

    cells = []
    n = [1]

    def nid():
        n[0] += 1
        return f"c{n[0]}"

    for el in build_elements():
        pts = el["pts"]
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            x0, y0 = px(*a)
            x1, y1 = px(*b)
            dash = "dashed=1;" if i in el["folds"] else ""
            cells.append(
                f'<mxCell id="{nid()}" value="" style="endArrow=none;'
                f'html=1;rounded=0;strokeWidth=1;strokeColor=#000000;'
                f'{dash}" edge="1" parent="1"><mxGeometry relative="1" '
                f'as="geometry"><mxPoint x="{x0:.2f}" y="{y0:.2f}" '
                f'as="sourcePoint"/><mxPoint x="{x1:.2f}" y="{y1:.2f}" '
                f'as="targetPoint"/></mxGeometry></mxCell>')
    for mid, (box, k, face) in MARKERS.items():
        img, pad_mm = marker_img(mid, box[2] - box[0], k)
        _, png = cv2.imencode(".png", img)
        b64 = base64.b64encode(png.tobytes()).decode()
        w = (box[2] - box[0] + 2 * pad_mm) * S
        x = (box[0] - pad_mm + CX) * S
        y = (PAGE_H - (box[3] + pad_mm) - CY) * S   # top-left in y-down
        cells.append(
            f'<mxCell id="{nid()}" value="" style="shape=image;imageAspect=1;'
            f'aspect=fixed;image=data:image/png;base64,{b64}" vertex="1" '
            f'parent="1"><mxGeometry x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" '
            f'height="{w:.2f}" as="geometry"/></mxCell>')
    xml = (
        '<mxfile host="app.diagrams.net" type="device">\n'
        '  <diagram name="fulcrum-net" id="fulcrum-net">\n'
        '    <mxGraphModel dx="800" dy="500" grid="1" gridSize="10" '
        'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" '
        'pageScale="1" pageWidth="827" pageHeight="1169" math="0" '
        'shadow="0">\n      <root>\n'
        '        <mxCell id="0"/>\n'
        '        <mxCell id="1" parent="0"/>\n'
        + "".join("        " + c + "\n" for c in cells) +
        '      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n')
    with open(DRAWIO, "w") as fh:
        fh.write(xml)
    print("wrote", DRAWIO)


# ---- PNG preview of page 1 ---------------------------------------------
def render_preview():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(0, PAGE_H)
    ax.set_aspect("equal")
    ax.add_patch(plt.Rectangle((0, 0), PAGE_W, PAGE_H,
                               fill=True, fc="white", ec="black", zorder=-1))
    for el in build_elements():
        pts = np.array(to_page(el["pts"] + [el["pts"][0]]))
        for i in range(len(el["pts"])):
            ax.plot(pts[i:i + 2, 0], pts[i:i + 2, 1],
                    "k--" if i in el["folds"] else "k-", lw=0.6)
    for mid, (box, k, _) in MARKERS.items():
        img, pad_mm = marker_img(mid, box[2] - box[0], k)
        w = box[2] - box[0] + 2 * pad_mm
        ax.imshow(img, extent=(box[0] - pad_mm + CX, box[0] - pad_mm + CX + w,
                               box[1] - pad_mm + CY, box[1] - pad_mm + CY + w),
                  cmap="gray", vmin=0, vmax=255, zorder=3)
    ax.set_title("fulcrum net preview")
    os.makedirs(os.path.dirname(PREVIEW), exist_ok=True)
    fig.savefig(PREVIEW, dpi=110)
    print("wrote", PREVIEW)


if __name__ == "__main__":
    render_pdf()
    write_spec()
    write_drawio()
    render_preview()
