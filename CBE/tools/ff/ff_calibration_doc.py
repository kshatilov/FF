"""Generate ff_calibration.pdf — algebraic formalization of fulcrum-frustum
multi-device calibration (cameras, depth cameras, projectors as inverse
cameras).  Equations are rendered with matplotlib mathtext and embedded in a
reportlab document.  Run:  C:\\Python314\\python.exe ff_calibration_doc.py
"""
import io
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (BaseDocTemplate, Frame, Image, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Spacer,
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parent
PIPE_FF = ROOT.parent / 'pipe' / 'ff'
SPEC = json.loads((PIPE_FF / 'fulcrum_spec.json').read_text())
GEOM = json.loads((PIPE_FF / 'calib_geometry.json').read_text())
OUT_PDF = ROOT.parent / 'fulcrum' / 'ff_calibration.pdf'

# ---------------------------------------------------------------- mathtext
import hashlib
import tempfile

_EQ_DIR = Path(tempfile.mkdtemp(prefix='ffeq_'))

# DejaVu (bundled with matplotlib) for bracket-extension glyphs ⎡ ⎢ ⎣ ⎤ ⎥ ⎦
_MPL_TTF = Path(matplotlib.__file__).parent / 'mpl-data' / 'fonts' / 'ttf'
pdfmetrics.registerFont(TTFont('DejaVu', str(_MPL_TTF / 'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DejaVu-Bold',
                                str(_MPL_TTF / 'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFontFamily('DejaVu', normal='DejaVu', bold='DejaVu-Bold',
                              italic='DejaVu', boldItalic='DejaVu-Bold')


def eq(tex, max_w=455, fs=13):
    """Render a LaTeX-mathtext equation to a centered reportlab Image."""
    key = hashlib.md5(f'{fs}|{tex}'.encode()).hexdigest()[:16]
    png = _EQ_DIR / f'{key}.png'
    if not png.exists():
        fig = plt.figure(figsize=(6, 1))
        fig.text(0.5, 0.5, f'${tex}$', fontsize=fs, ha='center', va='center')
        fig.savefig(png, format='png', dpi=200, bbox_inches='tight',
                    pad_inches=0.06, transparent=True)
        plt.close(fig)
    from PIL import Image as PILImage
    with PILImage.open(png) as im:
        pw_px, ph_px = im.size
    pw = pw_px * 72 / 200                    # natural width in points
    ph = ph_px * 72 / 200
    img = Image(str(png), hAlign='CENTER')
    if pw > max_w:
        s = max_w / pw
        img.drawWidth, img.drawHeight = pw * s, ph * s
    else:
        img.drawWidth, img.drawHeight = pw, ph
    return img


def eqs(*texes, gap=4):
    flow = []
    for t in texes:
        flow.append(eq(t))
        flow.append(Spacer(1, gap))
    return flow


def mat(rows, fs=12):
    """Matrix as a reportlab Table: cells are mathtext images, flanked by
    real bracket glyphs (⎡ ⎢ ⎣ / ⎤ ⎥ ⎦) from DejaVu Sans."""
    n, m = len(rows), len(rows[0])
    TOP_L, MID_L, BOT_L = '⎡', '⎢', '⎣'
    TOP_R, MID_R, BOT_R = '⎤', '⎥', '⎦'
    cell_ps = ParagraphStyle('mc', fontName='Helvetica', fontSize=fs,
                             leading=fs + 2, alignment=TA_CENTER)
    br_ps = ParagraphStyle('br', fontName='DejaVu', fontSize=fs + 9,
                           leading=fs + 3, alignment=TA_CENTER)

    def glyph(ps, top, mid, bot, i):
        if n == 1:
            return Paragraph('[' if ps == 'l' else ']', br_ps)
        ch = top if i == 0 else (bot if i == n - 1 else mid)
        return Paragraph(ch, br_ps)

    data = []
    for i, row in enumerate(rows):
        r = [glyph('l', TOP_L, MID_L, BOT_L, i)]
        r += [eq(c, fs=fs) for c in row]
        r.append(glyph('r', TOP_R, MID_R, BOT_R, i))
        data.append(r)
    t = Table(data, colWidths=[10] + [None] * m + [10], hAlign='CENTER',
              style=TableStyle([
                  ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                  ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                  ('TOPPADDING', (0, 0), (-1, -1), 1),
                  ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
                  ('LEFTPADDING', (0, 0), (-1, -1), 2),
                  ('RIGHTPADDING', (0, 0), (-1, -1), 2)]))
    return t


# ---------------------------------------------------------------- styles
BODY = ParagraphStyle('body', fontName='Helvetica', fontSize=9.5, leading=13.5,
                      alignment=TA_JUSTIFY, spaceAfter=5)
SMALL = ParagraphStyle('small', parent=BODY, fontSize=8, leading=11,
                       textColor=colors.HexColor('#333333'))
H1 = ParagraphStyle('h1', fontName='Helvetica-Bold', fontSize=14, leading=17,
                    spaceBefore=14, spaceAfter=6, textColor=colors.HexColor('#123a5c'))
H2 = ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=11, leading=14,
                    spaceBefore=10, spaceAfter=4, textColor=colors.HexColor('#1d5a8a'))
TITLE = ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=20, leading=24,
                       alignment=TA_CENTER, spaceAfter=4,
                       textColor=colors.HexColor('#123a5c'))
SUB = ParagraphStyle('sub', fontName='Helvetica', fontSize=10.5, leading=14,
                     alignment=TA_CENTER, textColor=colors.HexColor('#444444'),
                     spaceAfter=10)
CAP = ParagraphStyle('cap', parent=SMALL, alignment=TA_CENTER, spaceBefore=2,
                     spaceAfter=8)
NOTE = ParagraphStyle('note', parent=SMALL, backColor=colors.HexColor('#f2f6fa'),
                      borderPadding=5, spaceBefore=4, spaceAfter=8)


def tbl(data, widths, header=True, font=8, align='CENTER'):
    rows = [[Paragraph(str(c), ParagraphStyle('c', fontName='Helvetica-Bold'
                 if (header and i == 0) else 'Courier', fontSize=font,
                 leading=font + 2, alignment={'LEFT': 0, 'CENTER': 1,
                 'RIGHT': 2}[align])) for c in row] for i, row in
            enumerate(data)]
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#9db4c8')),
          ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dce8f2')),
          ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
          ('TOPPADDING', (0, 0), (-1, -1), 2),
          ('BOTTOMPADDING', (0, 0), (-1, -1), 2)]
    if header:
        st.append(('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dce8f2')))
    t.setStyle(TableStyle(st))
    return t


# ---------------------------------------------------------------- examples
def look_at(C, f, up=(0, 0, 1.0)):
    """World->camera rotation R and translation t for camera centre C (mm),
    forward direction f (unit), world up reference.  Columns of R_c2w are
    (right, down, forward); returns R_w2c, t = -R C."""
    z = np.asarray(f, float); z /= np.linalg.norm(z)
    u = np.asarray(up, float)
    if abs(np.dot(z, u)) > 0.99:
        u = np.array([0.0, 1.0, 0.0])
    x = np.cross(z, u); x /= np.linalg.norm(x)      # right = fwd x up
    y = np.cross(z, x)                              # down  = fwd x right
    R_c2w = np.column_stack([x, y, z])
    R = R_c2w.T
    return R, -R @ np.asarray(C, float)


def project(K, dist, R, t, X):
    """Full OpenCV-style projection: world point -> distorted pixel."""
    Xc = R @ np.asarray(X, float) + t
    x, y = Xc[0] / Xc[2], Xc[1] / Xc[2]
    r2 = x * x + y * y
    k1, k2, k3, p1, p2 = dist
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.array([K[0, 0] * xd + K[0, 1] * yd + K[0, 2],
                     K[1, 1] * yd + K[1, 2]])


# example device set-up (32 x 16 px, mm world)
W_IMG, H_IMG = 32, 16
FX = FY = 80.0
CX, CY = W_IMG / 2, H_IMG / 2
K_EX = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])
D_EX = np.zeros(5)

# fulcrum pose in the table world frame, as stored by ff_calibrate.py:
# X_world = R_OF X_object + t_OF
FUL = GEOM['fulcrum']
R_OF = cv2.Rodrigues(np.float64(FUL['rvec']))[0]
t_OF = np.float64(FUL['pos_mm'])

# example camera looking at the fulcrum
C_EX = np.array([300.0, -100.0, 550.0])
FF_C = t_OF + R_OF @ np.array([0.0, 0.0, 21.0])   # ff mid-height centre
f_EX = FF_C - C_EX
R_EX, t_EX = look_at(C_EX, f_EX)

# project marker 10 (top) corners into the example camera
mk10 = np.array(SPEC['markers']['10']['corners_TL_TR_BR_BL'])
px10 = [project(K_EX, D_EX, R_EX, t_EX, R_OF @ X + t_OF) for X in mk10]

# Rodrigues worked example on the real stored rvec
rv = np.float64(FUL['rvec'])
theta = float(np.linalg.norm(rv))
axis = rv / theta
R_rodrigues_check = cv2.Rodrigues(rv)[0]

# depth-camera back-projection example
u_d, v_d, d_dep = 10.0, 5.0, 400.0
Kd = K_EX
X_depth = d_dep * np.linalg.inv(Kd) @ np.array([u_d, v_d, 1.0])

# ---------------------------------------------------------------- diagram
def make_diagram():
    fig = plt.figure(figsize=(7.2, 2.9))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    colors_m = plt.cm.tab10(np.linspace(0, 1, 5))
    for i, (mid, m) in enumerate(sorted(SPEC['markers'].items(),
                                       key=lambda kv: int(kv[0]))):
        pts = np.array(m['corners_TL_TR_BR_BL'])
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=[colors_m[i]], s=18)
        c = pts.mean(0)
        ax.text(*c, f'  id{mid}', fontsize=7)
    ax.plot([0, 0], [0, 0], [0, 42.5], 'k--', lw=0.8)
    ax.set_xlabel('x (mm)', fontsize=7); ax.set_ylabel('y (mm)', fontsize=7)
    ax.set_zlabel('z (mm)', fontsize=7)
    ax.set_title('marker corners, object frame', fontsize=8)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=18, azim=-55)

    ax2 = fig.add_subplot(1, 2, 2)
    # side elevation (+y face): trapezoid from (y,z) corners of base/top
    base = [(-77.5, 0), (77.5, 0), (37.5, 42.5), (-37.5, 42.5)]
    xs = [p[0] for p in base] + [base[0][0]]
    zs = [p[1] for p in base] + [base[0][1]]
    ax2.fill(xs, zs, fc='#dce8f2', ec='#1d5a8a', lw=1.2)
    ax2.annotate('', xy=(77.5, -4), xytext=(-77.5, -4),
                 arrowprops=dict(arrowstyle='<->', lw=0.9))
    ax2.text(0, -8, 'base 155 mm', ha='center', fontsize=7.5)
    ax2.annotate('', xy=(37.5, 46), xytext=(-37.5, 46),
                 arrowprops=dict(arrowstyle='<->', lw=0.9))
    ax2.text(0, 50, 'top 75 mm', ha='center', fontsize=7.5)
    ax2.annotate('', xy=(82, 0), xytext=(82, 42.5),
                 arrowprops=dict(arrowstyle='<->', lw=0.9))
    ax2.text(87, 21, 'h = 42.5', fontsize=7.5, va='center')
    ax2.text(60, 30, 'tilt 46.7°', fontsize=7.5, color='#a33')
    ax2.set_xlim(-95, 110); ax2.set_ylim(-12, 58)
    ax2.set_aspect('equal'); ax2.axis('off')
    ax2.set_title('side elevation', fontsize=8)
    fig.savefig(_EQ_DIR / 'diagram.png', format='png', dpi=200,
                bbox_inches='tight')
    plt.close(fig)
    return _EQ_DIR / 'diagram.png'


def code_block(lines, width=488):
    """Monospace command block as a shaded table."""
    import html
    txt = '<br/>'.join(html.escape(l) for l in lines)
    p = Paragraph(f'<font face="Courier" size="8.4">{txt}</font>',
                  ParagraphStyle('code', fontName='Courier', fontSize=8.4,
                                  leading=11, textColor=colors.HexColor('#1a1a1a')))
    t = Table([[p]], colWidths=[width])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f4f4f2')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#bbbbbb')),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
    return t


# ---------------------------------------------------------------- document
story = []
story.append(Paragraph('Fulcrum-Frustum Multi-Device Calibration', TITLE))
story.append(Paragraph('Algebraic formalization — cameras, depth cameras, '
                       'projectors as inverse cameras', SUB))
story.append(Paragraph('Project Synchronicity · 2026-11-05 · '
                       'based on measured FF v1 geometry in '
                       '<font face="Courier">fulcrum_spec.json</font>', SUB))
story.append(Spacer(1, 6))

story.append(Paragraph('1. Problem statement', H1))
story.append(Paragraph(
    'Three device sets share one physical calibration object, the Fulcrum '
    'Frustum (FF): colour cameras <b>C</b> = {C<sub>1</sub>…}, depth cameras '
    '<b>D</b> = {D<sub>1</sub>…}, and projectors <b>P</b> = {P<sub>1</sub>…}. '
    'Every camera detects the FF\'s ArUco markers; every projector can display '
    'a known planar ArUco pattern. The FF geometry is measured once and fixed. '
    'The goal: recover, for every device, its position and orientation in a '
    'common world frame plus its lens intrinsics and distortion — from '
    '2D↔3D correspondences alone. <font face="Courier">solvePnP</font> is the '
    'core primitive; everything else is bookkeeping of frames.'))

story.append(Paragraph('2. Frames and notation', H1))
story.append(Paragraph(
    'Three frames are used throughout. The <b>world frame</b> '
    '<b>W</b> is an arbitrary fixed room frame (e.g. the table-board frame '
    'already used by <font face="Courier">calib_geometry.json</font>). The '
    '<b>object frame</b> <b>O</b> is rigidly attached to the FF: origin at the '
    'base centre, +z up, units mm. Each device has its own frame '
    '<b>D<sub>i</sub></b> (for a camera, the OpenCV convention: origin at the '
    'optical centre, +z along the pointing direction, +x right, +y down).'))
story.append(Paragraph(
    'A rigid transform is written T<sub>A←B</sub> = [R | t] and maps '
    'coordinates in frame B to frame A:'))
story += eqs(
    r'\mathbf{X}_A = \mathbf{R}_{A\leftarrow B}\,\mathbf{X}_B + \mathbf{t}_{A\leftarrow B}, '
    r'\qquad \mathbf{R}_{A\leftarrow B}\in SO(3),\ \ \mathbf{t}_{A\leftarrow B}\in\mathbb{R}^3',
    r'T_{A\leftarrow C} = T_{A\leftarrow B}\;T_{B\leftarrow C} '
    r'\;\Longrightarrow\; \mathbf{R}_{A\leftarrow C}=\mathbf{R}_{A\leftarrow B}\mathbf{R}_{B\leftarrow C},\quad '
    r'\mathbf{t}_{A\leftarrow C}=\mathbf{R}_{A\leftarrow B}\mathbf{t}_{B\leftarrow C}+\mathbf{t}_{A\leftarrow B}')
story.append(Paragraph(
    'The inverse of a pose is exact: T<sub>B←A</sub> = T<sub>A←B</sub><sup>−1</sup> '
    'with R<sup>T</sup> = R<sup>−1</sup> and t′ = −R<sup>T</sup>t. A camera '
    'centre expressed in world coordinates is C<sub>w</sub> = −R<sup>T</sup>t '
    'when [R|t] is the camera-from-world pose.'))

story.append(Paragraph('3. The calibration object', H1))
story.append(Paragraph(
    'FF v1 is a square frustum: base 155 mm, top 75 mm, height 42.5 mm, face '
    'tilt ≈ 46.7° derived from the measured geometry. Five faces carry '
    'DICT_4X4_50 markers: id 10 on top (55×55 mm), ids 11–14 on the four side '
    'faces (50×50 mm). Marker ids 10–14 are reserved for the FF.'))
story.append(tbl(
    [['measured', 'value'],
     ['base edge', '155.0 mm'], ['top edge', '75.0 mm'],
     ['height', '42.5 mm'], ['face tilt', '≈ 46.7° (derived)'],
     ['marker, top (id 10)', '55 × 55 mm'],
     ['markers, sides (ids 11–14)', '50 × 50 mm'],
     ['frame', 'origin = base centre, +z up, mm'],
     ['marker corner order', 'TL, TR, BR, BL (cv2.aruco order)']],
    [55 * mm, 95 * mm]))
story.append(Spacer(1, 4))
story.append(Paragraph(
    'Each marker yields four known 3D points (its corners) in frame O. The '
    'full measured corner table lives in <font face="Courier">'
    'fulcrum_spec.json</font>; representative values:'))
mk_rows = [['id', 'face', 'size', 'TL corner (x, y, z) mm',
            'TR corner (x, y, z) mm']]
for mid in ['10', '11', '12', '13', '14']:
    m = SPEC['markers'][mid]
    c = m['corners_TL_TR_BR_BL']
    mk_rows.append([mid, m['face'], f"{m['size_mm']:.0f} mm",
                    '(' + ', '.join(f'{v:.2f}' for v in c[0]) + ')',
                    '(' + ', '.join(f'{v:.2f}' for v in c[1]) + ')'])
story.append(tbl(mk_rows, [10 * mm, 14 * mm, 17 * mm, 52 * mm, 52 * mm],
                 font=7.5))
story.append(Paragraph(
    'BR and BL follow the same pattern (see the spec file for all 20 corners). '
    'Because the side faces are tilted planes, a side marker\'s top corners '
    'sit at smaller y/x and larger z than its bottom corners — e.g. id 11 '
    '(face +y) spans y = 40.24 → 74.51, z = 39.59 → 3.18.', SMALL))
story.append(Spacer(1, 4))
story.append(Image(str(make_diagram()), width=460, height=185))
story.append(Paragraph('Figure 1 — FF marker corners in the object frame '
                       '(left) and side elevation with measured dimensions '
                       '(right).', CAP))

story.append(Paragraph('4. Device model: intrinsics', H1))
story.append(Paragraph(
    'Every device is modelled by the same pinhole camera equations. The '
    '<b>intrinsic matrix</b> maps normalized image coordinates to pixels:'))
story.append(mat([[r'f_x', r's', r'c_x'], [r'0', r'f_y', r'c_y'],
                  [r'0', r'0', r'1']]))
story.append(Spacer(1, 4))
story += eqs(
    r'x = \frac{X_c}{Z_c},\qquad y = \frac{Y_c}{Z_c},\qquad r^2 = x^2 + y^2')
story.append(Paragraph(
    'where (X<sub>c</sub>, Y<sub>c</sub>, Z<sub>c</sub>) is a point in the '
    'device frame and (x, y) are normalized (distortion-free) coordinates. '
    'f<sub>x</sub>, f<sub>y</sub> are focal lengths in pixels, (c<sub>x</sub>, '
    'c<sub>y</sub>) the principal point, s the skew (usually 0). Lens '
    'distortion — radial k<sub>1..3</sub>, tangential p<sub>1..2</sub> — warps '
    'the normalized coordinates before projection:'))
story += eqs(
    r'x\' = x\left(1 + k_1 r^2 + k_2 r^4 + k_3 r^6\right) + 2 p_1 x y + p_2\left(r^2 + 2x^2\right)',
    r'y\' = y\left(1 + k_1 r^2 + k_2 r^4 + k_3 r^6\right) + p_1\left(r^2 + 2y^2\right) + 2 p_2 x y',
    r'u = f_x x\' + s\,y\' + c_x,\qquad v = f_y y\' + c_y')
story.append(Paragraph(
    'Collecting the pose (R, t) — i.e. T<sub>cam←world</sub> — the full '
    'projection of a world point X<sub>w</sub> to pixel q = (u, v) is '
    '(writing q̄ = (u, v, 1)<sup>T</sup> and X̄<sub>w</sub> = '
    '(X<sub>w</sub>, Y<sub>w</sub>, Z<sub>w</sub>, 1)<sup>T</sup>):'))
story += eqs(
    r's\,\bar{\mathbf{q}} = '
    r'\mathbf{K}\,[\mathbf{R}\ |\ \mathbf{t}]\,\bar{\mathbf{X}}_w,\qquad s = Z_c > 0')
story.append(Paragraph(
    'The scalar s is the depth: the equation is homogeneous, so only the '
    'ratio u/s, v/s is observed. Distortion coefficients are bundled as '
    'd = (k<sub>1</sub>, k<sub>2</sub>, k<sub>3</sub>, p<sub>1</sub>, p<sub>2</sub>) '
    'and are the OpenCV convention used by <font face="Courier">cv2.</font>'))

story.append(Paragraph('5. Device pose parameterization', H1))
story.append(Paragraph(
    'A pose has 6 degrees of freedom. Three parameterizations are used: the '
    'rotation vector rvec (Rodrigues), the position/pointing pair, and the '
    'full matrix [R | t].'))
story.append(Paragraph('5.1 Rotation vector ↔ matrix (Rodrigues)', H2))
story.append(Paragraph(
    'A rotation is encoded as rvec ρ = θ·n̂ — axis n̂ (unit) times angle θ. '
    'With the skew-symmetric matrix'))
story.append(mat([[r'0', r'-n_z', r'n_y'],
                  [r'n_z', r'0', r'-n_x'],
                  [r'-n_y', r'n_x', r'0']]))
story.append(Spacer(1, 2))
story.append(Paragraph('(denoted [n̂]<sub>×</sub>), the rotation matrix is the '
                        'Rodrigues formula — the matrix exponential of the '
                        'skew matrix:'))
story += eqs(
    r'\mathbf{R} = \mathbf{I} + \sin\theta\,[\hat{\mathbf{n}}]_\times '
    r'+ (1-\cos\theta)\,[\hat{\mathbf{n}}]_\times^{\,2}')
rv_show = ', '.join(f'{v:.4f}' for v in FUL['rvec'])
pos_show = ', '.join(f'{v:.2f}' for v in FUL['pos_mm'])
story.append(Paragraph(
    '<font face="Courier">cv2.Rodrigues</font> converts both ways. Worked '
    'example — the FF pose stored in <font face="Courier">'
    f'calib_geometry.json</font>, rvec = ({rv_show}):  '
    f'θ = ‖ρ‖ = {theta:.3f} rad = {np.degrees(theta):.1f}°, '
    f'axis n̂ = ρ/θ = ({axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}).'))
story.append(mat([[f'{v:.3f}' for v in row] for row in R_OF], fs=11))
story.append(Paragraph(
    'R<sub>OF</sub> ≈ the matrix above (orthonormal, det = +1 — verify '
    'R<sup>T</sup>R = I), and maps object-frame vectors to world-frame '
    'vectors (X<sub>w</sub> = R<sub>OF</sub> X<sub>o</sub> + t<sub>OF</sub>), '
    f'with t<sub>OF</sub> = ({pos_show}) mm.'))

story.append(Paragraph('5.2 Position and pointing vector', H2))
story.append(Paragraph(
    'Equivalently a camera pose is given by its centre C = (x, y, z) in the '
    'world frame, a unit pointing vector p (where the optical axis goes), '
    'and a roll angle φ about that axis. Build an orthonormal camera basis '
    '(x̂ = right, ŷ = down, ẑ = forward) and rotate it by φ:'))
story += eqs(
    r'\hat{\mathbf{z}} = \hat{\mathbf{p}},\qquad '
    r'\hat{\mathbf{x}} = \frac{\hat{\mathbf{z}}\times\mathbf{u}_{\mathrm{ref}}}'
    r'{\|\hat{\mathbf{z}}\times\mathbf{u}_{\mathrm{ref}}\|},\qquad '
    r'\hat{\mathbf{y}} = \hat{\mathbf{z}}\times\hat{\mathbf{x}}',
    r'\hat{\mathbf{x}}(\varphi) = \cos\varphi\,\hat{\mathbf{x}} + \sin\varphi\,\hat{\mathbf{y}},\qquad '
    r'\hat{\mathbf{y}}(\varphi) = -\sin\varphi\,\hat{\mathbf{x}} + \cos\varphi\,\hat{\mathbf{y}}',
    r'\mathbf{R}_{c2w} = \left[\hat{\mathbf{x}}\ \hat{\mathbf{y}}\ \hat{\mathbf{z}}\right],\qquad '
    r'\mathbf{R} = \mathbf{R}_{c2w}^{\top},\qquad \mathbf{t} = -\mathbf{R}\,\mathbf{C}')
story.append(Paragraph(
    'u<sub>ref</sub> is any reference up direction not parallel to p (world +z '
    'works for level rigs; fall back to +y when p is vertical). This is the '
    'parameterization stored per device as <font face="Courier">cam_pos_world_mm'
    '</font>, <font face="Courier">fwd_world</font> and roll in '
    '<font face="Courier">calib_geometry.json</font>.'))

story.append(Paragraph('6. From detections to correspondences', H1))
story.append(Paragraph(
    'A camera frame containing the FF produces, per visible marker j, four '
    'sub-pixel image corners q<sub>j,k</sub> = (u, v) (cv2.aruco detects the '
    'black-border square; corners come in TL,TR,BR,BL order). Each corner has '
    'a known object-frame coordinate X<sub>o,j,k</sub> from the spec table. '
    'Mapping object → world through the (known or previously solved) FF pose '
    'T<sub>W←O</sub> gives the correspondence set:'))
story += eqs(
    r'\mathcal{S}_i = \left\{\,(\mathbf{X}_{w,j,k},\ \mathbf{q}_{j,k})\,\right\},\qquad '
    r'\mathbf{X}_{w,j,k} = \mathbf{R}_{OF}\mathbf{X}_{o,j,k} + \mathbf{t}_{OF}')
story.append(Paragraph(
    'A single view of all five markers yields 20 correspondences — well '
    'above the 4 non-coplanar points PnP needs for a unique solution.'))

story.append(Paragraph('7. Pose estimation: solvePnP', H1))
story.append(Paragraph(
    'Given K, distortion d, and correspondences 𝒮, find (R, t) minimizing '
    'the reprojection error — the geometric distance in pixels between each '
    'observed corner and where its 3D point projects under the candidate '
    'pose:'))
story += eqs(
    r'\hat{\mathbf{q}}_i = \pi(\mathbf{K},\mathbf{d};\mathbf{R},\mathbf{t},\mathbf{X}_i),\qquad '
    r'\mathbf{r}_i = \hat{\mathbf{q}}_i - \mathbf{q}_i',
    r'\min_{\mathbf{R}\in SO(3),\,\mathbf{t}}\ E(\mathbf{R},\mathbf{t}) '
    r'= \sum_{i}\rho\!\left(\|\mathbf{r}_i\|^2\right)')
story.append(Paragraph(
    'ρ is a robust kernel (Huber) or plain squares; ‖·‖ is the Euclidean '
    'pixel distance. Three solver families implement this:'))
story.append(tbl(
    [['solver', 'idea', 'role here'],
     ['EPnP / SQPNP', 'non-iterative; expresses the 3D points as a weighted '
      'sum of 4 control points → linear system solved by SVD',
      'fast seed pose per view'],
     ['iterative (LM)', 'Gauss–Newton refinement of the same objective; '
      'each step solves (JᵀJ)δ = −Jᵀr for a pose increment',
      'polish: cv2.solvePnPRefineLM'],
     ['RANSAC', 'repeat the solve on random minimal subsets, keep the '
      'hypothesis with most inliers', 'reject false marker detections']],
    [26 * mm, 78 * mm, 46 * mm], font=7.5))
story.append(Paragraph(
    'The Jacobian J is the 2×6 derivative of π with respect to a local '
    'parameterization of the pose (3 rotation + 3 translation increments). '
    'This is exactly the loop in <font face="Courier">_paired_pose()</font> '
    'of <font face="Courier">ff_calibrate.py</font>: SQPNP seed → LM refine '
    '→ report mean/max residual; residuals above ~3 px mean bad detections '
    'or a moved FF.'))

story.append(Paragraph('8. Intrinsics and distortion estimation', H1))
story.append(Paragraph(
    'solvePnP takes K and d as given. To *estimate* them, collect many '
    'views of the FF (or any known board) per device and solve one joint '
    'nonlinear least-squares problem over all correspondences of that '
    'device — the camera-calibration analogue of §7 with a bigger '
    'parameter vector:'))
story += eqs(
    r'\boldsymbol{\theta}_i = \left(f_x, f_y, c_x, c_y, s,\ k_1, k_2, k_3, p_1, p_2,\ '
    r'\mathbf{R}_i, \mathbf{t}_i\right),\qquad '
    r'\min_{\boldsymbol{\theta}_i}\ \sum_{v}\sum_{j,k}\ '
    r'\rho\left\|\pi(\boldsymbol{\theta}_i;\mathbf{X}_{j,k}) - \mathbf{q}_{v,j,k}\right\|^2')
story.append(Paragraph(
    'Per view the pose (R<sub>v</sub>, t<sub>v</sub>) is free (10 intrinsic '
    'parameters shared across views + 6 pose parameters per view); if the '
    'device is static, a single pose serves all views — this is what '
    '<font face="Courier">cv2.calibrateCamera</font> does internally. '
    'Observability: intrinsics need parallax, so vary the FF distance and '
    'angle across views; a tiny object at fixed distance constrains pose '
    'well but leaves focal length and distance partially correlated. '
    'Distortion is only observable off-axis — include views where markers '
    'sit near the image borders.'))

story.append(Paragraph('9. Depth cameras', H1))
story.append(Paragraph(
    'A depth camera adds a per-pixel range d(u, v). Two uses — '
    '(a) back-projection of a pixel to a 3D point in the device frame '
    '(after undistortion):'))
story += eqs(
    r'\mathbf{X}_c = d\,\mathbf{K}_d^{-1}\,\mathbf{m},\qquad '
    r'\mathbf{m} = (u, v, 1)^{\top}')
story.append(Paragraph(
    '(b) registration: the depth frame and the colour frame of the same '
    'device are related by a fixed rigid transform T<sub>color←depth</sub> '
    '(factory or solved once by observing the FF in both streams). Points '
    'back-projected in the depth frame are then expressed in the colour '
    'frame — or directly in world once the device pose is known. Worked '
    'example with the §12 intrinsics: pixel (10, 5) at depth 400 mm gives'))
story += eqs(
    r'\mathbf{X}_c = 400\left(\frac{10-16}{80},\ \frac{5-8}{80},\ 1\right)^{\top}'
    r' = \left(-30,\ -15,\ 400\right)^{\top}\ \mathrm{mm}')
story.append(Paragraph(
    'Depth also makes pose estimation easier: back-projected marker '
    'corners give direct 3D↔3D correspondences, so the pose can be fit '
    'without the projective ambiguity of a single RGB view.'))

story.append(Paragraph('10. Projectors as inverted cameras', H1))
story.append(Paragraph(
    'A projector obeys the same pinhole equation, read in reverse. The '
    '"image" is the emitted pattern: pattern pixel (u, v) corresponds to a '
    'ray leaving the projector\'s exit pupil. With the same intrinsics '
    'K<sub>p</sub> and pose [R<sub>p</sub> | t<sub>p</sub>] = T<sub>proj←world</sub>:'))
story += eqs(
    r's\,\bar{\mathbf{q}} = '
    r'\mathbf{K}_p\,[\mathbf{R}_p\ |\ \mathbf{t}_p]\,\bar{\mathbf{X}}_w',
    r'\mathbf{X}_w(\lambda) = \mathbf{C}_p + \lambda\,\mathbf{R}_p^{\top}\,'
    r'\mathbf{K}_p^{-1}\,\mathbf{m},\qquad '
    r'\mathbf{m} = (u, v, 1)^{\top},\ \lambda > 0')
story.append(Paragraph(
    "The second form is the operational one: it says that projector pixel "
    "(u,v) illuminates the world point where this ray lands. Two "
    "calibration routes follow from what is known:"))
story.append(tbl(
    [['route', 'known data', 'solve'],
     ['planar board', 'projector displays an ArUco plane; the plane\'s pose '
      'T<sub>W←plane</sub> is known (measured wall, FF face, or solved by a '
      'camera)', 'each pattern pixel ↔ known 3D point → same PnP problem, '
      'image points = pattern pixels'],
     ['camera-in-the-loop', 'a calibrated camera photographs the projected '
      'pattern on a known-geometry surface (e.g. FF faces, which are '
      'planes)', 'reconstruct pattern-pixel ↔ 3D-point correspondences → '
      'projector PnP']],
    [24 * mm, 72 * mm, 54 * mm], font=7.5))
story.append(Paragraph(
    'For a perfectly planar pattern↔plane mapping the projection degenerates '
    'to a homography — 8 DOF instead of full PnP:'))
story += eqs(
    r's\,\bar{\mathbf{q}} = \mathbf{H}\,\bar{\mathbf{m}}_{\pi},\qquad '
    r'\bar{\mathbf{m}}_{\pi} = (U, V, 1)^{\top},\qquad '
    r'\mathbf{H} = \mathbf{K}_p\left[\mathbf{r}_1\ \mathbf{r}_2\ \mathbf{t}\right]')
story.append(Paragraph(
    'where (U, V) are coordinates on the pattern plane. This is exactly the '
    'ray-cast used in <font face="Courier">ff_calibrate.py</font> '
    '(§export_viewer): pattern pixels are cast as rays to the z = 0 plane '
    'of the FF frame and fitted with <font face="Courier">cv2.findHomography'
    '</font> → H<sub>px→mm</sub>.'))

story.append(Paragraph('11. Global frame assembly', H1))
story.append(Paragraph(
    'PnP solves each device relative to whatever frame its correspondences '
    'live in — the FF object frame O. To express everything in the world '
    'frame W, compose with the FF\'s world pose T<sub>W←O</sub> (itself '
    'obtained once, e.g. from surveyed placement or a board solve):'))
story += eqs(
    r'[\mathbf{R}\ |\ \mathbf{t}]_{c\leftarrow w} = '
    r'[\mathbf{R}\ |\ \mathbf{t}]_{c\leftarrow o}\;'
    r'[\mathbf{R}\ |\ \mathbf{t}]_{o\leftarrow w},\qquad '
    r'\mathbf{X}_o = \mathbf{R}_{O\leftarrow W}\mathbf{X}_w + \mathbf{t}_{O\leftarrow W},\quad '
    r'\mathbf{R}_{O\leftarrow W} = \mathbf{R}_{OF}^{\top},\quad '
    r'\mathbf{t}_{O\leftarrow W} = -\mathbf{R}_{OF}^{\top}\mathbf{t}_{OF}')
story.append(Paragraph(
    'Choosing W = O is always valid for a first pass (all device poses then '
    'share the FF frame); upgrading to a room frame later is a single rigid '
    'transform applied to every stored pose — no re-solve needed. Scale is '
    'not ambiguous: the FF\'s millimetre dimensions fix the metric, which is '
    'why a measured physical object beats a printed pattern of unknown size.'))

story.append(Paragraph('12. Worked example (32 × 16 px devices)', H1))
story.append(Paragraph(
    'All devices share W = 32, H = 16. Example intrinsics (no distortion '
    'initially):'))
story.append(mat([['80', '0', '16'], ['0', '80', '8'], ['0', '0', '1']],
                 fs=12))
story.append(Paragraph(
    'with distortion d = (0, 0, 0, 0, 0).'))
story.append(Paragraph(
    'Example camera: centre C = (300, −100, 550) mm in the table world '
    'frame, pointed at the FF mid-height. Its pose from §5.2, and the '
    'projection of the four top-marker (id 10) corners '
    '(X<sub>w</sub> = R<sub>OF</sub>X<sub>o</sub> + t<sub>OF</sub>, then '
    'X<sub>c</sub> = R X<sub>w</sub> + t, then §4):'))
ex_rows = [['corner', 'world X (mm)', 'world Y (mm)', 'world Z (mm)',
            'pixel u', 'pixel v']]
for k, (Xo, q) in enumerate(zip(mk10, px10)):
    Xw = R_OF @ Xo + t_OF
    ex_rows.append([['TL', 'TR', 'BR', 'BL'][k],
                    f'{Xw[0]:8.2f}', f'{Xw[1]:8.2f}', f'{Xw[2]:8.2f}',
                    f'{q[0]:6.2f}', f'{q[1]:6.2f}'])
story.append(tbl(ex_rows, [12 * mm, 24 * mm, 24 * mm, 24 * mm, 14 * mm, 14 * mm],
                 font=7.5))
story.append(Paragraph(
    'All four corners land inside the 32×16 frame — the FF subtends only a '
    'few pixels at ~600 mm, which is why real deployments use higher '
    'resolution and/or multiple FF positions. The same math at 640×480 with '
    'f ≈ 516 px is what <font face="Courier">calib_geometry.json</font> '
    'stores for the live webcam.'))

story.append(Paragraph('13. Calibration process — formalized', H1))
story.append(Paragraph('13.1 Operating modes', H2))
story.append(tbl(
    [['mode', 'when', 'what runs', 'output'],
     ['S — static survey', 'FF parked; one-time or after hardware change',
      'offline calibrate (§13.2)', 'K, d, pose per device in W'],
     ['R — recalibrate', 'FF moved to a new position; want world-frame '
      'poses again', 'same command as S — it re-solves the FF pose itself',
      'updated calib_geometry.json'],
     ['L — live', 'performance; devices and/or FF may move', 'per-frame '
      'tracker + EKF (§13.4)', 'fulcrum_poses.json @ frame rate']],
    [20 * mm, 52 * mm, 60 * mm, 38 * mm], font=7.5))
story.append(Paragraph(
    'The key structural fact: <b>every solve is performed in the FF object '
    'frame O</b>. The FF\'s own position in W, T<sub>W←O</sub>, enters only '
    'at the final composition step. Therefore:'))
story += eqs(
    r'T_{c\leftarrow w} = T_{c\leftarrow o}\;T_{o\leftarrow w}')
story.append(Paragraph(
    'If the FF moves but a consumer only needs the pose <i>relative to the '
    'FF</i> (e.g. projecting onto its faces), nothing needs re-solving — '
    'T<sub>c←o</sub> is already correct. If world-frame poses are needed, '
    'only T<sub>o←w</sub> must be refreshed (re-measure the mount, or solve '
    'it from a fixed reference camera that still sees the FF). If instead a '
    '<i>device</i> moved, its T<sub>c←o</sub> changes and must be re-solved '
    '— which the live tracker does every frame anyway.'))

story.append(Paragraph('13.2 Offline pipeline (modes S and R)', H2))
for i, s in enumerate([
    '<b>A0 — freeze the object.</b> MEAS → fulcrum_spec.json: 5 markers, '
    '20 corners in frame O (mm). Done once; ids 10–14 reserved.',
    '<b>A1 — acquire.</b> For each device, capture views containing ≥3 FF '
    'markers (more is better; vary distance/angle for intrinsics).',
    '<b>A2 — pose per view.</b> Detections → correspondences 𝒮 (§6) → '
    'solvePnP (§7) with current K, d → (R<sub>v</sub>, t<sub>v</sub>) per view.',
    '<b>A3 — intrinsics.</b> Joint nonlinear refinement (§8) over all views '
    'of a device → K<sub>i</sub>, d<sub>i</sub>; then re-run A2 once with '
    'the refined values.',
    '<b>A4 — world assembly.</b> Solve or measure T<sub>W←O</sub> for the '
    'FF\'s current placement; compose every device pose (§11); convert to '
    '(C, p, φ) storage form; write calib_geometry.json.',
    '<b>A5 — validate.</b> Reprojection stats per device (mean < 1 px, '
    'max < 3 px); cross-checks: two cameras viewing the FF must agree on '
    'its pose; fulcrum_check.py confirms the live frame.'], 1):
    story.append(Paragraph(s, BODY))
story.append(Paragraph(
    'Mode R is literally the same command: the solver re-derives the FF '
    'pose from the markers it sees, so a moved FF produces an updated '
    'T<sub>W←O</sub> automatically. No manual measurement step exists.'))

story.append(Paragraph('13.3 Live tracking loop (mode L)', H2))
story.append(Paragraph(
    'Each frame t, for each device i, the tracker runs a detect → solve → '
    'filter cycle. The filter is a constant-velocity EKF over the 12-dim '
    'pose state (position p, rotation vector r, linear velocity v, angular '
    'velocity ω) — the PoseEKF in <font face="Courier">pose_ekf.py</font>:'
))
story += eqs(
    r'\mathbf{x} = [\mathbf{p}^{\top}\ \mathbf{r}^{\top}\ \mathbf{v}^{\top}\ '
    r'\boldsymbol{\omega}^{\top}]^{\top} \in \mathbb{R}^{12}')
story.append(Paragraph('<b>Predict</b> (every frame, Δt = frame interval):'))
story += eqs(
    r'\hat{\mathbf{x}}^{-} = \mathbf{F}\,\mathbf{x},\qquad '
    r'\mathbf{P}^{-} = \mathbf{F}\,\mathbf{P}\,\mathbf{F}^{\top} + \mathbf{Q},\qquad '
    r'\mathbf{F} = \mathrm{blkdiag}(\mathbf{F}_{pv},\ \mathbf{F}_{r\omega})')
story.append(mat([[r'\mathbf{I}', r'\Delta t\,\mathbf{I}'],
                  [r'\mathbf{0}', r'\mathbf{I}']], fs=11))
story.append(Paragraph(
    'where the shown 2×2 block is F<sub>pv</sub> = F<sub>rω</sub> acting on '
    '(position, velocity) and (rotation vector, angular velocity); Q injects '
    'process noise (acceleration and angular-acceleration terms — tuned for '
    'handheld motion). <b>Update</b> only when a PnP fix passes the gates:'))
story += eqs(
    r'\mathbf{S} = \mathbf{H}\mathbf{P}^{-}\mathbf{H}^{\top} + \mathbf{R}_m,\qquad '
    r'\mathbf{G} = \mathbf{P}^{-}\mathbf{H}^{\top}\mathbf{S}^{-1},\qquad '
    r'\mathbf{x} = \hat{\mathbf{x}}^{-} + \mathbf{G}\,\mathbf{y},\quad '
    r'\mathbf{y} = \mathbf{z}_t - \mathbf{H}\hat{\mathbf{x}}^{-}')
story.append(Paragraph(
    'where z<sub>t</sub> is the PnP measurement (position + rotation '
    'vector), H selects the measured pose components, and R<sub>m</sub> is '
    'the measurement covariance (≈ 2 mm position, 0.015 rad rotation in '
    'the current tuning). <b>Gate:</b> a fix is accepted only if its '
    'Mahalanobis distance is small and its reprojection error is small:'))
story += eqs(
    r'g^2 = \mathbf{y}^{\top}\mathbf{S}^{-1}\mathbf{y} < g_{\max}^2,\qquad '
    r'\overline{e}_{\mathrm{px}} < \varepsilon_{\mathrm{gate}}')
story.append(Paragraph(
    'Rejected fixes are counted; after a reacquire timeout the filter '
    'hard-resets onto the newest accepted measurement — this is what lets '
    'a moved camera or a moved FF "jump" instead of being dragged by stale '
    'velocity.'))

story.append(Paragraph('13.4 Dropouts and moving devices', H2))
story.append(tbl(
    [['situation', 'detection', 'filter behaviour', 'consumer sees'],
     ['FF visible, device still', '≥3 markers, low residual', 'update every '
      'frame', 'TRACKING — fresh pose'],
     ['FF briefly lost (occlusion, motion blur, gimbal whip)', '<3 markers '
      'or solve fails', 'predict-only coast; P grows via Q',
      'COASTING — extrapolated pose, bounded by T<sub>lost</sub>'],
     ['FF or device moved while unseen', 'fix returns with large '
      'innovation', 'gate rejects until reacquire timeout → hard reset',
      'jump to new pose (no stale smoothing)'],
     ['FF moved, still visible', 'solve in O unaffected', 'world pose '
      'changes only through T<sub>o←w</sub>', 'update T<sub>W←O</sub>, '
      're-compose (§13.1)']],
    [42 * mm, 42 * mm, 52 * mm, 38 * mm], font=7.2))
story.append(Paragraph(
    'Additional guards already in the tracker: RANSAC/iterative solve with '
    'a reprojection-error report, the projected-canvas footprint polygon '
    '(FIELD in <font face="Courier">fulcrum_track.py</font>) rejecting '
    'solves that land outside the lit area, and a minimum-marker-count '
    'rule (non-planar PnP needs ≥3 markers = 12 correspondences, well '
    'posed against the 5-face geometry). For a phone/gimbal camera the '
    'practical limits are motion blur and rolling shutter — the coast '
    'window (≈0.5 s in the current tuning) should stay shorter than the '
    'expected time-to-reacquire.'))

story.append(Paragraph('14. Command reference', H1))
story.append(Paragraph(
    'All commands run from the repo root <font face="Courier">'
    'C:\\dev\\ProjectSynchronicity</font> with '
    '<font face="Courier">C:\\Python314\\python.exe</font>.'))
story.append(Paragraph('14.1 Full calibration — static or moved FF (modes S/R)', H2))
story.append(code_block([
    '.\\ff calib',
    '',
    '# phases: webcam solvePnP -> projector structured-light solve',
    '#          -> kinect registration -> writes tools\\pipe\\ff\\calib_geometry.json',
    '# safe to re-run any time; the FF may be anywhere - its pose is',
    '# re-solved from the markers, so a moved FF just updates T_W<-O']))
story.append(Paragraph('14.2 Realtime tracking (mode L)', H2))
story.append(code_block([
    'C:\\Python314\\python.exe tools\\ff\\fulcrum_track.py',
    '',
    '# one process owns webcam + kinect; per frame: aruco detect ->',
    '# solvePnP in FF frame -> EKF gate/update -> structured-light',
    '# projector dots -> writes fulcrum_poses.json + preview feeds']))
story.append(Paragraph('14.3 TouchDesigner side', H2))
story.append(code_block([
    'node tools\\td\\td.mjs --file tools\\ff\\td_ff_live.py   # live pose consumer',
    'node tools\\td\\td.mjs --file tools\\ff\\td_ff_tick.py   # per-tick updater',
    'node tools\\td\\td.mjs --file tools\\ff\\td_ff_solve.py  # TD-side solve']))
story.append(Paragraph('14.4 Diagnostics', H2))
story.append(code_block([
    'C:\\Python314\\python.exe tools\\ff\\fulcrum_check.py          # one-frame pose + residual check',
    'C:\\Python314\\python.exe tools\\ff\\fulcrum_kinect.py --frames 30   # kinect registration frames']))
story.append(Paragraph(
    'Typical session: run 14.1 once with the FF parked → validate with '
    '14.4 → start 14.2 for the performance → 14.3 consumes '
    '<font face="Courier">fulcrum_poses.json</font> inside TouchDesigner. '
    'If the FF is bumped mid-show, mode-L tracking keeps working in the '
    'FF frame automatically; re-run 14.1 afterwards if world-frame poses '
    'must be refreshed.'))

story.append(Paragraph('15. Pipeline summary', H1))
for i, s in enumerate([
    '<b>Measure once.</b> FF geometry → fulcrum_spec.json (corners in '
    'object frame, mm).',
    '<b>Per camera view.</b> Detect ArUco ids 10–14 → 2D corners; build '
    'correspondences via T<sub>W←O</sub>.',
    '<b>Pose.</b> solvePnP (SQPNP/EPnP seed → LM refine → RANSAC if '
    'outliers) → (R, t) = T<sub>cam←world</sub>; convert to '
    '(C, p, φ) for storage.',
    '<b>Intrinsics.</b> Multi-view joint refinement (§8) → K, d per '
    'device; re-solve poses with refined K, d.',
    '<b>Depth cams.</b> Back-project with K<sub>d</sub>; fix '
    'T<sub>color←depth</sub> once.',
    '<b>Projectors.</b> Display known ArUco plane or project onto known '
    'FF faces; camera-in-the-loop correspondences → projector PnP; '
    'planar case reduces to homography.',
    '<b>Assemble.</b> Compose all poses into W (§11); validate by '
    'reprojection error (target: mean < 1 px, max < 3 px).'], 1):
    story.append(Paragraph(f'{i}. {s}', BODY))

story.append(Spacer(1, 8))
story.append(Paragraph(
    'References: Hartley & Zisserman, <i>Multiple View Geometry</i>, ch. 6–7 '
    '(camera model, DLT/EPnP); OpenCV docs — solvePnP, calibrateCamera, '
    'Rodrigues; project code: tools/ff/ff_calibrate.py, '
    'fulcrum_track.py, pose_ekf.py.', SMALL))


# ---------------------------------------------------------------- build
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 7.5)
    canvas.setFillColor(colors.HexColor('#777777'))
    canvas.drawCentredString(A4[0] / 2, 12 * mm,
                             f'FF calibration formalization — page {doc.page}')
    canvas.restoreState()


doc = BaseDocTemplate(str(OUT_PDF), pagesize=A4,
                      leftMargin=18 * mm, rightMargin=18 * mm,
                      topMargin=16 * mm, bottomMargin=18 * mm,
                      title='Fulcrum-Frustum Multi-Device Calibration',
                      author='Project Synchronicity')
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
              id='main')
doc.addPageTemplates([PageTemplate(id='all', frames=[frame],
                                   onPage=footer)])
doc.build(story)
print(f'wrote {OUT_PDF}')
