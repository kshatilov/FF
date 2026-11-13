import json
import math
import time
from itertools import permutations, product
from pathlib import Path

import cv2
import numpy as np

try:
    from scipy.optimize import least_squares as _least_squares
    from scipy.ndimage import map_coordinates as _map_coordinates
except ImportError:
    _least_squares = None
    _map_coordinates = None

ROOT = Path('C:/dev/ProjectSynchronicity')
CALIBRATION = ROOT / 'tools/pipe/ff/calibration_current.json'
REGISTRATION = ROOT / 'tools/pipe/ff/td_kinect_registration.json'
SPEC = ROOT / 'tools/pipe/ff/ffv2_spec.json'

PROJECTOR_SIZE = (1280, 720)  # native cv2 projector image (w, h)
MAX_OBJECT_FRACTION = 0.20    # of the smaller image dimension
MIN_OBJECT_PX = 12            # minimum candidate bbox side
PREDICT_SECONDS = 0.15        # bounded EKF prediction while a detection is missing
HOLD_SECONDS = 1.5            # static hold + fade to black after loss
FADE_START = 0.6              # fade begins this long after the last accepted pose
INSET_EXPAND = 1.15           # fitted interior triangles sit inside the real face

# OpenCV hue is 0..179. (lo, hi) inclusive ranges, minimum saturation.
COLOR_RANGES = {
    'red':    ([(0, 11), (150, 179)], 40),
    'yellow': ([(19, 38)],           40),
    'green':  ([(40, 92)],           26),
    'violet': ([(135, 178)],         26),
}
FACE_COLOR = {'DBC': 'black', 'ADB': 'green', 'DEC': 'red', 'ADE': 'violet', 'AEC': 'yellow'}
FACE_COLOR_INV = {v: k for k, v in FACE_COLOR.items()}
ALL_FACES = ('DBC', 'ADB', 'DEC', 'ADE', 'AEC', 'ABC')
SATURATED = ('red', 'yellow', 'green', 'violet')
MESH_EDGES = [('A', 'B'), ('B', 'C'), ('C', 'A'), ('A', 'D'), ('B', 'D'),
              ('C', 'D'), ('A', 'E'), ('C', 'E'), ('D', 'E')]
EDGE_FACES = {}
for _f in ALL_FACES:
    for _i in range(3):
        _e = tuple(sorted((_f[_i], _f[(_i+1) % 3])))
        EDGE_FACES.setdefault(_e, []).append(_f)


def rt(rv, tv):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.float64(rv).reshape(3))[0]
    T[:3, 3] = np.float64(tv).reshape(3)
    return T


def look_at(camera_pos, target, up=(0, 1, 0)):
    """Proper world->camera transform: R rows are camera x,y,z in world coords."""
    z = np.float64(target)-np.float64(camera_pos)
    z /= np.linalg.norm(z)
    x = np.cross(z, np.float64(up))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)
    return cv2.Rodrigues(R)[0], (-R @ np.float64(camera_pos)).reshape(3, 1)


def load_geometry():
    spec = json.loads(SPEC.read_text())
    vertices = {k: np.float64(v) for k, v in spec['vertices'].items()}
    centroid = np.mean(list(vertices.values()), axis=0)
    faces = {}
    for name, info in spec['faces'].items():
        labels = list(name)
        pts = np.array([vertices[l] for l in labels])
        normal = np.cross(pts[1]-pts[0], pts[2]-pts[0])
        if normal @ (pts.mean(axis=0)-centroid) < 0:
            labels = labels[::-1]
            pts = pts[::-1]
            normal = -normal
        faces[name] = dict(labels=labels, points=pts,
                           normal=normal/np.linalg.norm(normal),
                           rgb=tuple(info['rgb']), track=info['track'], project=info['project'])
    return spec, vertices, faces


def color_mask(bgr, hsv, family):
    if family == 'black':
        # chromatically neutral dark only — red borders and warm shading excluded
        b, g, r = (bgr[:, :, i].astype(np.int16) for i in range(3))
        dark = bgr[:, :, :3].max(axis=2) < 125
        neutral = (np.abs(r-b) < 28) & (np.abs(r-g) < 22) & (np.abs(g-b) < 28)
        return (dark & neutral).astype(np.uint8)*255
    ranges, sat = COLOR_RANGES[family]
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, (lo, sat, 0), (hi, 255, 255))
    rb = bgr[:, :, 2].astype(np.int16)-bgr[:, :, 0].astype(np.int16)
    if family == 'red':
        # split red hue wrap from violet: red needs R clearly above B
        mask = cv2.bitwise_and(mask, mask, mask=(rb > 8).astype(np.uint8)*255)
    elif family == 'violet':
        mask = cv2.bitwise_and(mask, mask, mask=(rb <= 30).astype(np.uint8)*255)
    return mask


class CameraModel:
    """Camera contract: the switch index selects a source, never fuses them.
    Unvalidated or mismatched calibration is a hard error, not a silent default."""

    def __init__(self, name):
        self.name = name
        self.cal_mtime = CALIBRATION.stat().st_mtime
        self.reg_mtime = REGISTRATION.stat().st_mtime
        calibration = json.loads(CALIBRATION.read_text())
        if not calibration.get('validated'):
            raise RuntimeError(f'calibration {CALIBRATION} is not validated')
        projector_rt = rt(calibration['projector']['rvec'], calibration['projector']['tvec'])
        webcam_rt = rt(calibration['webcam']['rvec'], calibration['webcam']['tvec'])
        self.projector_K = np.float64(calibration['projector']['K'])
        self.calibration = calibration['timestamp']
        if name == 'webcam':
            self.source = 'camLogi'
            self.size = (640, 480)
            self.K = np.float64(calibration['webcam']['K'])
            self.D = np.float64(calibration['webcam'].get('D', np.zeros(5)))
            self.projector_from_camera = projector_rt @ np.linalg.inv(webcam_rt)
        elif name in ('kinect', 'kinect_depth'):
            registration = json.loads(REGISTRATION.read_text())
            if registration['timestamp'] != calibration['timestamp']:
                raise RuntimeError('kinect registration timestamp '
                                   f"{registration['timestamp']} != calibration "
                                   f"{calibration['timestamp']}")
            if list(registration['resolution']) != [1280, 720]:
                raise RuntimeError('kinect registration resolution mismatch')
            self.size = (1280, 720)
            self.K = np.float64(registration['K'])
            self.D = np.float64(registration['D'])
            transform = np.float64(registration['webcam_from_kinect'])
            self.projector_from_camera = projector_rt @ np.linalg.inv(webcam_rt) @ transform
            # kinect depth has no color: tracking consumes the kinectC color companion
            self.source = 'kinectC' if name == 'kinect' else 'kinect_depth+color'
        else:
            raise ValueError(f'unknown camera {name!r}')

    def stale(self):
        try:
            if CALIBRATION.stat().st_mtime != self.cal_mtime:
                return True
            return self.name in ('kinect', 'kinect_depth') and \
                REGISTRATION.stat().st_mtime != self.reg_mtime
        except OSError:
            return False


class FFv2Tracker:
    """Color-and-geometry tracker for the FFv2 painted solid.

    process_frame(frame_bgr_top_down, timestamp) ->
        (720x1280x4 uint8 RGBA projector image in true top-down order, state dict)
    """

    def __init__(self, camera):
        from pose_ekf import PoseEKF
        self.camera = CameraModel(camera)
        _, self.vertices, self.faces = load_geometry()
        self.pose_filter = PoseEKF(position_noise=4.0, rotation_noise=0.03,
                                   acceleration_noise=120, angular_acceleration_noise=2.0,
                                   gate=45, reacquire_after=0.25, stationary_enabled=False)
        self.output = np.zeros((PROJECTOR_SIZE[1], PROJECTOR_SIZE[0], 4), np.uint8)
        self.output[:, :, 3] = 255
        self.reset()

    def reset(self):
        self.pose_filter.time = None
        self.pose_filter.accepted_time = None
        self.last_accepted_pose = None
        self.last_accepted_time = None
        self.held_pose = None
        self.state = {}

    # ---------------- detection ----------------

    def candidate_masks(self, bgr, hsv):
        """Seeds are saturated colors only — dark pixels are never global seeds
        (they would glue the object onto any dark surroundings)."""
        masks = {family: color_mask(bgr, hsv, family) for family in SATURATED}
        masks['black'] = color_mask(bgr, hsv, 'black')
        union = masks['red'] | masks['yellow'] | masks['green'] | masks['violet']
        union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return masks, union

    def find_candidates(self, union, shape):
        limit = MAX_OBJECT_FRACTION*min(shape)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(union)
        out = []
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < 60 or max(w, h) > limit or min(w, h) < 6 or w < MIN_OBJECT_PX*0.5:
                continue
            out.append(dict(box=(x, y, w, h), area=area, label=i))
        return out, labels

    def face_interiors(self, masks, labels, label_id, box, origin, shape):
        """Filled colored interiors: erode so thin raised red borders vanish,
        keep regions that still hold a filled triangle. Dark pixels are only
        considered inside this candidate — never a global black search."""
        x, y, w, h = box
        x0, y0 = max(0, x-6), max(0, y-6)
        x1, y1 = min(shape[1], x+w+6), min(shape[0], y+h+6)
        # saturated seeds define the component; the black face is searched
        # only inside a small dilation of it — still a local ROI, never global
        inside = cv2.dilate((labels[y0:y1, x0:x1] == label_id).astype(np.uint8),
                            np.ones((5, 5), np.uint8))
        observations = []
        for ksize in (5, 3):
            kernel = np.ones((ksize, ksize), np.uint8)
            for family in ('red', 'yellow', 'green', 'violet', 'black'):
                if any(o['face'] == FACE_COLOR_INV[family] for o in observations):
                    continue
                sub = masks[family][y0:y1, x0:x1]
                sub = cv2.bitwise_and(sub, sub, mask=inside.astype(np.uint8))
                if family != 'black':
                    sub = cv2.erode(sub, kernel)
                n, lab, st, _ = cv2.connectedComponentsWithStats(sub)
                best = None
                min_edge = 0.18*max(w, h)
                min_area = max(10.0, 0.02*w*h)
                for i in range(1, n):
                    if st[i, 4] < min_area:
                        continue
                    region = (lab == i).astype(np.uint8)
                    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue
                    contour = max(contours, key=cv2.contourArea)
                    hull = cv2.convexHull(contour)
                    triangle = None
                    for eps in (0.03, 0.05, 0.08, 0.12, 0.18):
                        approx = cv2.approxPolyDP(hull, eps*cv2.arcLength(hull, True), True)
                        if len(approx) <= 3:
                            triangle = approx.reshape(-1, 2)
                            break
                    if triangle is None or len(triangle) != 3:
                        continue
                    edges = np.linalg.norm(np.roll(triangle, 1, axis=0)-triangle, axis=1)
                    if edges.min() < min_edge:
                        continue  # border remnant, not a filled face
                    # crop-local corner -> full image coordinates, then expand
                    # about the centroid: erosion+borders inset the fitted corners
                    corners = triangle + np.array([x0+origin[0], y0+origin[1]])
                    center = corners.mean(axis=0)
                    corners = center+(corners-center)*INSET_EXPAND
                    support = cv2.contourArea(contour)
                    if best is None or support > best['support']:
                        # observed interior pixels in the crop coordinate frame
                        pixels = np.argwhere(region > 0) + np.array([y0, x0])
                        best = dict(face=FACE_COLOR_INV[family],
                                    corners=corners.astype(np.float64), support=support,
                                    pixels=pixels)
                if best is not None:
                    observations.append(best)
            if len(observations) >= 2:
                break
        return observations

    # ---------------- pose hypotheses ----------------

    def hypotheses(self, observations, radius):
        """Assign each observed face's 3 model vertices to its 3 image corners.
        A model vertex shared by two observed faces must land on image points
        within `radius` — shared-vertex correspondences are then merged."""
        perms = list(permutations(range(3)))
        choices = []
        for obs in observations:
            verts = self.faces[obs['face']]['labels']
            local = []
            for perm in perms:
                mapping = {}
                for corner_i, vert_i in enumerate(perm):
                    mapping[verts[vert_i]] = obs['corners'][corner_i]
                local.append(mapping)
            choices.append(local)
        results, seen = [], set()
        for combo in product(*choices):
            merged = {}
            ok = True
            for mapping in combo:
                for vert, point in mapping.items():
                    if vert in merged:
                        if np.linalg.norm(merged[vert][0]-point) > radius:
                            ok = False
                            break
                        merged[vert].append(point)
                    else:
                        merged[vert] = [point]
                if not ok:
                    break
            key = tuple(sorted((v, tuple(np.round(np.mean(p, axis=0)))) for v, p in merged.items()))
            if ok and len(merged) >= 3 and key not in seen:
                seen.add(key)
                results.append({v: np.mean(p, axis=0) for v, p in merged.items()})
            if len(results) > 160:
                break
        # single-face fallbacks: a lone observed face still yields P3P hypotheses
        for local in choices:
            for mapping in local:
                key = tuple(sorted((v, tuple(np.round(p))) for v, p in mapping.items()))
                if key not in seen:
                    seen.add(key)
                    results.append({v: p for v, p in mapping.items()})
                if len(results) > 200:
                    return results
        return results

    def solve(self, correspondences):
        verts = sorted(correspondences)
        obj = np.float64([self.vertices[v] for v in verts])
        img = np.float64([correspondences[v] for v in verts])
        # this OpenCV build only supports SQPNP in solvePnPGeneric (P3P/IPPE
        # raise cv2.error); SQPNP accepts >=3 points and returns alternatives
        try:
            _, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, self.camera.K,
                                                     self.camera.D, flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:
            return []
        if rvecs is None:
            return []
        out = []
        for rv, tv in zip(rvecs, tvecs):
            try:
                rv2, tv2 = cv2.solvePnPRefineVVS(obj, img, self.camera.K, self.camera.D,
                                               rv.reshape(3, 1), tv.reshape(3, 1))
            except cv2.error:
                rv2, tv2 = rv.reshape(3, 1), tv.reshape(3, 1)
            out.append((rv2, tv2, verts, img))
        return out

    def projector_uv(self, rv, tv):
        T = self.camera.projector_from_camera @ rt(rv, tv)
        verts = np.array(list(self.vertices.values()))
        rv_p = cv2.Rodrigues(T[:3, :3])[0]
        return cv2.projectPoints(verts, rv_p, T[:3, 3].reshape(3, 1),
                                 self.camera.projector_K, np.zeros(5))[0].reshape(-1, 2)

    # ---------------- scoring (local crop coordinates) ----------------

    def visible_faces(self, rv, tv):
        """Faces front-facing to the camera with all vertices in front."""
        T = rt(rv, tv)
        out = []
        for name, face in self.faces.items():
            cam_pts = face['points'] @ T[:3, :3].T + T[:3, 3]
            if np.any(cam_pts[:, 2] <= 1.0):
                continue
            center = cam_pts.mean(axis=0)
            if (T[:3, :3] @ face['normal']) @ (-center) <= 0:
                continue
            out.append(name)
        return out

    def score_pose(self, rv, tv, bgr, hsv, cand_mask, shape, offset, observations=()):
        T = rt(rv, tv)
        verts = np.array(list(self.vertices.values()))
        cam_verts = verts @ T[:3, :3].T + T[:3, 3]
        if np.any(cam_verts[:, 2] <= 1.0):
            return None
        uv = cv2.projectPoints(verts, rv, tv, self.camera.K, self.camera.D)[0].reshape(-1, 2)
        if not np.isfinite(uv).all():
            return None
        uv = uv-offset
        h, w = shape
        if not (-0.3*w < uv[:, 0].min() and uv[:, 0].max() < 1.3*w
                and -0.3*h < uv[:, 1].min() and uv[:, 1].max() < 1.3*h):
            return None
        hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)
        silhouette = np.zeros(shape, np.uint8)
        cv2.fillConvexPoly(silhouette, np.round(hull).astype(np.int32), 255)
        sil_px = int((silhouette > 0).sum())
        if sil_px < 20:
            return None
        inside = cv2.bitwise_and(silhouette, cand_mask)
        precision = float((inside > 0).sum())/sil_px
        cover = float((inside > 0).sum())/max(int((cand_mask > 0).sum()), 1)
        geo = min(1.0, 2*precision*cover/max(precision+cover, 1e-6))
        eroded_sil = cv2.erode(silhouette, np.ones((3, 3), np.uint8))
        color_scores = []
        face_detail = {}
        for name in self.visible_faces(rv, tv):
            face = self.faces[name]
            fuv = cv2.projectPoints(face['points'], rv, tv, self.camera.K,
                                    self.camera.D)[0].reshape(-1, 2)-offset
            fc = fuv.mean(axis=0)
            inner = np.round(fc+(fuv-fc)*0.55).astype(np.int32)
            fmask = np.zeros(shape, np.uint8)
            cv2.fillConvexPoly(fmask, inner, 255)
            fmask = cv2.bitwise_and(fmask, eroded_sil)
            count = int((fmask > 0).sum())
            if count < 6:
                continue
            family = FACE_COLOR.get(name)
            if family is None:
                match = cv2.inRange(hsv, (0, 0, 0), (179, 60, 255))
            else:
                match = color_mask(bgr, hsv, family)
            frac = float((cv2.bitwise_and(match, fmask) > 0).sum()/count)
            face_detail[name] = round(frac, 3)
            color_scores.append((frac, 1.0 if face['track'] else 0.3))
        color = (sum(f*wt for f, wt in color_scores)/sum(wt for _, wt in color_scores)
                 if color_scores else 0.0)
        # every observed colored region must be covered by its predicted face:
        # a pose that puts e.g. the observed red pixels outside DEC is wrong
        obs_cover = {}
        visible = set(self.visible_faces(rv, tv))
        for obs in observations:
            if obs['face'] not in visible:
                obs_cover[obs['face']] = 0.0  # observed face must face the camera
                continue
            fuv = cv2.projectPoints(self.faces[obs['face']]['points'], rv, tv,
                                    self.camera.K, self.camera.D)[0].reshape(-1, 2)-offset
            fmask = np.zeros(shape, np.uint8)
            cv2.fillConvexPoly(fmask, np.round(fuv).astype(np.int32), 255)
            fmask = cv2.dilate(fmask, np.ones((3, 3), np.uint8))
            inside = (fmask[obs['pixels'][:, 0], obs['pixels'][:, 1]] > 0).mean()
            obs_cover[obs['face']] = round(float(inside), 3)
        return dict(score=0.5*color+0.5*geo, color=color, geo=geo,
                    precision=precision, cover=cover, faces=face_detail, uv=uv,
                    hull=hull, obs_cover=obs_cover)

    # ---------------- bounded joint refinement ----------------

    def refine_pose(self, rv, tv, verts, img, cand_mask, masks, hsv, offset):
        """Smooth bounded joint fit over four residual families, all evaluated
        by bilinear sampling of distance fields (no discrete correspondences
        to silhouette pixels, no flat integer gradients):

        a) hypothesis vertex anchors (weak — fitted triangles were inset),
        b) projected visible silhouette hull edges vs the observed outer
           contour distance field (fixed sample count, constant OOB penalty),
        c) samples on camera-visible physical edges vs the red-border field,
        d) samples inside visible face interiors vs their color fields.
        """
        if _least_squares is None or _map_coordinates is None:
            return rv, tv, False
        h, w = cand_mask.shape
        # observed outer contour (candidate mask boundary) distance field
        boundary = cv2.dilate(cand_mask, np.ones((3, 3), np.uint8)) & ~cand_mask
        dist_outer = cv2.distanceTransform(255-boundary, cv2.DIST_L2, 3).astype(np.float32)
        dist_red = cv2.distanceTransform(255-masks['red'], cv2.DIST_L2, 3).astype(np.float32)
        dist_color = {}
        for name in ALL_FACES:
            family = FACE_COLOR.get(name)
            if family == 'red':
                continue  # red borders already cover the red face
            mask = (color_mask(masks['bgr'], hsv, family) if family
                    else cv2.inRange(hsv, (0, 0, 0), (179, 60, 255)))
            dist_color[name] = cv2.distanceTransform(255-mask, cv2.DIST_L2,
                                                     3).astype(np.float32)
        verts_arr = np.array(list(self.vertices.values()))
        labels = list(self.vertices)
        hull_offset = np.float64(offset)
        x0 = np.r_[np.ravel(rv), np.ravel(tv)]
        span = np.linalg.norm(tv)
        anchor_idx = [labels.index(v) for v in verts]

        # residual topology frozen at the seed pose so the vector size is fixed
        visible0 = set(self.visible_faces(rv, tv))
        edge_samples = []
        for a, b in MESH_EDGES:
            e = tuple(sorted((a, b)))
            if not any(f in visible0 for f in EDGE_FACES.get(e, [])):
                continue
            pa, pb = self.vertices[a], self.vertices[b]
            edge_samples.append(np.float64([pa*(1-t)+pb*t
                                            for t in np.linspace(0.1, 0.9, 3)]))
        face_samples = {}
        for name in visible0:
            if name not in dist_color:
                continue
            pts3 = self.faces[name]['points']
            c3 = pts3.mean(axis=0)
            face_samples[name] = c3+(pts3-c3)*0.6
        N_HULL = 24

        def hull_samples(poly):
            """Exactly N_HULL points evenly spaced along the polyline."""
            n = len(poly)
            if n < 2:
                return np.repeat(poly[:1], N_HULL, axis=0)
            seg = np.linalg.norm(np.roll(poly, -1, axis=0)-poly, axis=1)
            total = seg.sum()
            if total < 1e-6:
                return np.repeat(poly[:1], N_HULL, axis=0)
            out = []
            dist = 0.0
            j = 0
            for s in np.linspace(0, total, N_HULL, endpoint=False):
                while dist+seg[j] < s:
                    dist += seg[j]
                    j = (j+1) % n
                a, b = poly[j], poly[(j+1) % n]
                f = (s-dist)/max(seg[j], 1e-9)
                out.append(a*(1-f)+b*f)
            return np.float64(out)

        def residuals(x):
            r_ = x[:3].reshape(3, 1)
            t_ = x[3:].reshape(3, 1)
            res = []
            uv_all = cv2.projectPoints(verts_arr, r_, t_, self.camera.K,
                                       self.camera.D)[0].reshape(-1, 2)
            # (a) anchors — eroded-inset corners, so moderate weight only
            pred = uv_all[anchor_idx]
            res.extend((pred-img).ravel()*0.5)
            # (b) silhouette hull boundary vs observed outer contour
            try:
                hull = cv2.convexHull(uv_all.astype(np.float32)).reshape(-1, 2)
            except cv2.error:
                hull = uv_all
            pts = hull_samples(hull)-hull_offset
            inside = (pts[:, 0] >= 0) & (pts[:, 0] <= w-1.001) & \
                     (pts[:, 1] >= 0) & (pts[:, 1] <= h-1.001)
            vals = np.full(len(pts), 8.0)
            if inside.any():
                pi = pts[inside]
                vals[inside] = _map_coordinates(dist_outer, [pi[:, 1], pi[:, 0]],
                                                order=1, mode='constant', cval=8.0)
            res.extend(np.clip(vals-1.0, -1.0, 6.0)*0.6)
            # (c) visible physical edges vs red-border field
            for pts3 in edge_samples:
                uv = cv2.projectPoints(pts3, r_, t_, self.camera.K,
                                       self.camera.D)[0].reshape(-1, 2)-hull_offset
                okp = (uv[:, 0] >= 0) & (uv[:, 0] <= w-1.001) & \
                      (uv[:, 1] >= 0) & (uv[:, 1] <= h-1.001)
                vals = np.full(len(uv), 6.0)
                if okp.any():
                    pi = uv[okp]
                    vals[okp] = _map_coordinates(dist_red, [pi[:, 1], pi[:, 0]],
                                                 order=1, mode='constant', cval=6.0)
                res.extend(np.clip(vals-1.0, -1.0, 5.0)*0.45)
            # (d) visible face interiors vs their color fields
            for name, inner3 in face_samples.items():
                uv = cv2.projectPoints(inner3, r_, t_, self.camera.K,
                                       self.camera.D)[0].reshape(-1, 2)-hull_offset
                okp = (uv[:, 0] >= 0) & (uv[:, 0] <= w-1.001) & \
                      (uv[:, 1] >= 0) & (uv[:, 1] <= h-1.001)
                vals = np.full(len(uv), 6.0)
                if okp.any():
                    pi = uv[okp]
                    vals[okp] = _map_coordinates(dist_color[name], [pi[:, 1], pi[:, 0]],
                                                 order=1, mode='constant', cval=6.0)
                res.extend(np.clip(vals-1.0, -1.0, 5.0)*0.5)
            return np.float64(res)

        lower = x0-np.r_[[0.4]*3+[30, 30, 0.2*span]]
        upper = x0+np.r_[[0.4]*3+[30, 30, 0.2*span]]
        try:
            sol = _least_squares(residuals, x0, method='trf', bounds=(lower, upper),
                                 loss='soft_l1', f_scale=2.0, diff_step=1e-4,
                                 max_nfev=120)
        except Exception:
            return rv, tv, False
        # status 0 (max_nfev) still returns the best point found inside the
        # bounds; only failed (<0) or non-finite results are rejected
        if sol.status < 0 or not np.isfinite(sol.x).all() or not np.isfinite(sol.cost):
            return rv, tv, False
        return sol.x[:3].reshape(3, 1), sol.x[3:].reshape(3, 1), True

    def detect_pose(self, frame, timestamp):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        shape = frame.shape[:2]
        masks, union = self.candidate_masks(frame, hsv)
        rois = [None]
        if self.last_accepted_time is not None and timestamp-self.last_accepted_time < 0.35:
            rv, tv = self.last_accepted_pose
            uv = cv2.projectPoints(np.array(list(self.vertices.values())), rv, tv,
                                   self.camera.K, self.camera.D)[0].reshape(-1, 2)
            x0, y0 = np.floor(uv.min(axis=0)-25).astype(int)
            x1, y1 = np.ceil(uv.max(axis=0)+25).astype(int)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(shape[1], x1), min(shape[0], y1)
            if x1-x0 > 20 and y1-y0 > 20:
                rois = [(x0, y0, x1, y1), None]
        best = None
        runner = None
        best4 = None
        runner4 = None
        diagnostics = dict(candidates=0, hypotheses=0)
        for roi in rois:
            if roi is not None:
                x0, y0, x1, y1 = roi
                sub = np.zeros(shape, np.uint8)
                sub[y0:y1, x0:x1] = union[y0:y1, x0:x1]
            else:
                x0, y0, x1, y1 = 0, 0, shape[1], shape[0]
                sub = union
            candidates, labels = self.find_candidates(sub, shape)
            diagnostics['candidates'] += len(candidates)
            for cand in candidates:
                cx0, cy0 = max(0, cand['box'][0]-30), max(0, cand['box'][1]-30)
                cx1 = min(shape[1], cand['box'][0]+cand['box'][2]+30)
                cy1 = min(shape[0], cand['box'][1]+cand['box'][3]+30)
                crop = (slice(cy0, cy1), slice(cx0, cx1))
                hsv_c = hsv[crop]
                bgr_c = frame[crop]
                cshape = hsv_c.shape[:2]
                box_c = (cand['box'][0]-cx0, cand['box'][1]-cy0,
                         cand['box'][2], cand['box'][3])
                masks_c = {f: color_mask(bgr_c, hsv_c, f) for f in SATURATED}
                masks_c['black'] = color_mask(bgr_c, hsv_c, 'black')
                masks_c['bgr'] = bgr_c
                labels_c = labels[crop]
                observations = self.face_interiors(masks_c, labels_c, cand['label'],
                                                   box_c, (cx0, cy0), cshape)
                if not observations:
                    continue
                object_d = float(max(cand['box'][2], cand['box'][3]))
                hypos = self.hypotheses(observations, max(4.0, 0.35*object_d))
                diagnostics['hypotheses'] += len(hypos)
                cand_mask = (labels_c == cand['label']).astype(np.uint8)*255
                cand_mask = cv2.dilate(cand_mask, np.ones((3, 3), np.uint8))
                offset = np.array([cx0, cy0])
                cand_top = []
                for mapping in hypos:
                    for rv, tv, verts, img in self.solve(mapping):
                        if rv is None:
                            continue
                        scored = self.score_pose(rv, tv, bgr_c, hsv_c, cand_mask,
                                                 cshape, offset, observations)
                        if scored is None:
                            continue
                        pred = cv2.projectPoints(
                            np.float64([self.vertices[v] for v in verts]),
                            rv, tv, self.camera.K, self.camera.D)[0].reshape(-1, 2)
                        scored['reproj'] = float(np.linalg.norm(pred-img, axis=1).mean())
                        scored['rv'], scored['tv'] = rv, tv
                        scored['faces_observed'] = [o['face'] for o in observations]
                        scored['n_points'] = len(verts)
                        scored['candidate'] = cand['label']
                        scored['ctx'] = (verts, img, cand_mask, masks_c, hsv_c,
                                         offset, cshape, bgr_c)
                        scored['observations'] = observations
                        cand_top.append(scored)
                cand_top.sort(key=lambda s: -s['score'])
                # ambiguity is measured against the runner-up of the SAME
                # candidate, never a different object in the scene
                if cand_top and (best is None or cand_top[0]['score'] > best['score']):
                    best = cand_top[0]
                    runner = cand_top[1] if len(cand_top) > 1 else None
                strong = [s for s in cand_top if s['n_points'] >= 4]
                if strong and (best4 is None or strong[0]['score'] > best4['score']):
                    best4 = strong[0]
                    runner4 = strong[1] if len(strong) > 1 else None
            if best is not None:
                break
        has_prior = (self.last_accepted_time is not None
                     and timestamp-self.last_accepted_time < 0.35)
        if not has_prior and best4 is not None:
            # with no tracked prior a lone triangle cannot anchor the pose:
            # prefer >=4-anchor multi-face hypotheses even at lower scores
            best, runner = best4, runner4
        if best is None:
            return None, None, diagnostics
        # bounded joint refinement on the winning hypothesis only
        verts, img, cand_mask, masks_c, hsv_c, offset, cshape, bgr_c = best['ctx']
        rv, tv, fitted = self.refine_pose(best['rv'], best['tv'], verts, img,
                                        cand_mask, masks_c, hsv_c, offset)
        refined = self.score_pose(rv, tv, bgr_c, hsv_c, cand_mask, cshape, offset,
                                  best.get('observations', ()))
        if refined is not None:
            refined['reproj'] = float(np.linalg.norm(
                cv2.projectPoints(np.float64([self.vertices[v] for v in verts]),
                                  rv, tv, self.camera.K,
                                  self.camera.D)[0].reshape(-1, 2)-img, axis=1).mean())
            refined['rv'], refined['tv'] = rv, tv
            refined['faces_observed'] = best['faces_observed']
            refined['n_points'] = best['n_points']
            refined['candidate'] = best['candidate']
            refined['refined'] = fitted
            if refined['score'] >= best['score']-0.05:
                best = refined
        if not fitted:
            # the bounded joint fit is part of the acceptance contract
            diagnostics['reject'] = 'refinement_failed'
            diagnostics.update(faces_observed=best.get('faces_observed'),
                               score=round(best.get('score', 0), 3))
            return None, None, diagnostics
        runner_score = runner['score'] if runner is not None else 0.0
        # alternatives must be distinct in projector space too — near-identical
        # camera projections can diverge massively projector-side
        proj_gap = float('inf')
        if runner is not None:
            proj_gap = float(np.linalg.norm(
                self.projector_uv(best['rv'], best['tv'])
                - self.projector_uv(runner['rv'], runner['tv']), axis=1).mean())
        ambiguous = (best['score']-runner_score < 0.08 and runner_score > 0.4
                     and proj_gap > 10.0)
        diagnostics.update(faces_observed=best['faces_observed'],
                           score=round(best['score'], 3), color=round(best['color'], 3),
                           geo=round(best['geo'], 3), reproj=round(best['reproj'], 2),
                           face_match=best['faces'], runner_up=round(runner_score, 3),
                           proj_gap=round(proj_gap, 1), ambiguous=ambiguous,
                           refined=fitted)
        diagnostics['obs_cover'] = best.get('obs_cover', {})
        accept = (best['score'] >= 0.55 and best['color'] >= 0.45
                  and best['geo'] >= 0.4 and best['reproj'] <= 6.0)
        if accept and best.get('obs_cover'):
            if min(best['obs_cover'].values()) < 0.4:
                accept = False
                diagnostics['reject'] = 'observed_region_not_covered'
        if accept and best['n_points'] < 4:
            # a lone triangle cannot anchor the pose at all: single-face
            # ambiguity is only ever resolved inside an established track
            accept = False
            diagnostics['reject'] = 'single_face_no_support'
        if accept and ambiguous:
            accept = False
            diagnostics['reject'] = 'ambiguous_hypothesis'
        if not accept:
            diagnostics.setdefault('reject', 'low_score')
            return None, None, diagnostics
        return best['rv'], best['tv'], diagnostics

    # ---------------- render ----------------

    def render(self, rv, tv, alpha=1.0):
        """True top-down RGBA projector image: row 0 is the top scanline,
        channels are R,G,B,A. The TD adapter flips once for TD's bottom-up
        numpy layout."""
        image = np.zeros((PROJECTOR_SIZE[1], PROJECTOR_SIZE[0], 4), np.uint8)
        image[:, :, 3] = 255
        T = self.camera.projector_from_camera @ rt(rv, tv)
        rv_p = cv2.Rodrigues(T[:3, :3])[0]
        tv_p = T[:3, 3].reshape(3, 1)
        Kp = self.camera.projector_K
        order = []
        for name, face in self.faces.items():
            cam_pts = face['points'] @ T[:3, :3].T + T[:3, 3]
            if np.any(cam_pts[:, 2] <= 1.0):
                continue
            center = cam_pts.mean(axis=0)
            if (T[:3, :3] @ face['normal']) @ (-center) <= 0:  # backface for the projector
                continue
            order.append((float(center[2]), name))
        for _, name in sorted(order, reverse=True):
            face = self.faces[name]
            uv = cv2.projectPoints(face['points'], rv_p, tv_p, Kp,
                                   np.zeros(5))[0].reshape(-1, 2)
            if not np.isfinite(uv).all() or np.abs(uv).max() > 20000:
                continue
            # black DBC and the ignored white base render black: projector off
            color = face['rgb'] if name in FACE_COLOR else (0, 0, 0)
            rgba = (int(round(color[0]*alpha)), int(round(color[1]*alpha)),
                    int(round(color[2]*alpha)), 255)
            cv2.fillConvexPoly(image, np.round(uv).astype(np.int32), rgba)
        return image

    # ---------------- main entry ----------------

    def _loss_state(self, state, timestamp):
        """Bounded drop-out handling shared by missing frames and rejected
        detections: <=0.15 s bounded prediction, then static hold of the last
        coast pose fading to black by 1.5 s. Never refreshes the accepted
        timestamp from predictions, never jumps back to the older pose."""
        anchor = self.last_accepted_time
        age = float('inf') if anchor is None else timestamp-anchor
        if np.isfinite(age) and age <= PREDICT_SECONDS and self.pose_filter.time is not None:
            coast = self.pose_filter.predict(timestamp)
            if coast is not None:
                self.held_pose = coast
                self.output[:] = self.render(*coast)
                state.update(status='predicting', pose_age_s=round(age, 3))
                return
        if np.isfinite(age) and age <= HOLD_SECONDS:
            pose = self.held_pose or self.last_accepted_pose
            if pose is not None:
                fade = min(1.0, max(0.0, (HOLD_SECONDS-age)/(HOLD_SECONDS-FADE_START)))
                self.output[:] = self.render(*pose, alpha=fade)
                state.update(status='holding' if fade >= 1.0 else 'fading',
                             pose_age_s=round(age, 3), fade=round(fade, 3))
                return
        if np.isfinite(age):
            state['pose_age_s'] = round(age, 3)

    def process_frame(self, frame, timestamp):
        state = dict(t=time.time(), timestamp=float(timestamp), source=self.camera.source,
                     camera=self.camera.name, calibration=self.camera.calibration,
                     status='lost', faces={k: FACE_COLOR.get(k, 'white') for k in ALL_FACES})
        self.output.fill(0)
        self.output[:, :, 3] = 255
        if self.camera.stale():
            try:
                self.camera = CameraModel(self.camera.name)
                self.reset()
                state['calibration'] = self.camera.calibration
                state['source'] = self.camera.source
            except Exception as error:
                # never continue detecting on a stale/broken calibration
                self.reset()
                state.update(status='lost', error=f'calibration reload failed: {error}',
                             fit={})
                self.state = state
                return self.output, self._sanitize(state)
        rv, tv, diag = None, None, {}
        try:
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                diag = dict(error='no frame' if frame is None else 'malformed frame')
            else:
                if frame.shape[1] != self.camera.size[0] or frame.shape[0] != self.camera.size[1]:
                    frame = cv2.resize(frame, self.camera.size)
                rv, tv, diag = self.detect_pose(frame, timestamp)
        except Exception as error:
            rv, tv, diag = None, None, dict(error=repr(error))
            state['error'] = repr(error)
        state['fit'] = diag
        accepted = False
        try:
            if rv is not None:
                (rv, tv), accepted = self.pose_filter.update(
                    rv, tv, timestamp, noise_scale=max(1.0, diag.get('reproj', 1.0)/1.5))
            if accepted:
                self.last_accepted_pose = (rv, tv)
                self.last_accepted_time = timestamp
                self.held_pose = (rv, tv)
                self.output[:] = self.render(rv, tv)
                uv = cv2.projectPoints(np.array(list(self.vertices.values())), rv, tv,
                                       self.camera.K, self.camera.D)[0].reshape(-1, 2)
                state.update(status='tracked',
                             pose=dict(rvec=np.ravel(rv).tolist(), tvec=np.ravel(tv).tolist()),
                             debug=dict(labels=list(self.vertices),
                                        uv=np.round(uv, 1).tolist()))
        except Exception as error:
            # detection, EKF and render are guarded together: a bad input must
            # never kill the worker or leave a partial projection
            accepted = False
            state['error'] = repr(error)
            self.output.fill(0)
            self.output[:, :, 3] = 255
        if not accepted:
            rejected_with_evidence = (rv is None and diag.get('score') is not None)
            self._loss_state(state, timestamp)
            if state['status'] == 'lost' and rejected_with_evidence:
                # hypotheses existed but were honestly rejected
                state['status'] = 'uncertain'
        self.state = state
        return self.output, self._sanitize(state)

    @staticmethod
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: FFv2Tracker._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [FFv2Tracker._sanitize(v) for v in obj]
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj

    def render_debug(self, frame):
        """Camera-space overlay for the ffv2_debug path — never projector output."""
        view = frame.copy()
        if self.last_accepted_pose is not None:
            rv, tv = self.last_accepted_pose
            uv = cv2.projectPoints(np.array(list(self.vertices.values())), rv, tv,
                                   self.camera.K, self.camera.D)[0].reshape(-1, 2)
            labels = list(self.vertices)
            idx = {l: i for i, l in enumerate(labels)}
            for a, b in MESH_EDGES:
                cv2.line(view, tuple(np.round(uv[idx[a]]).astype(int)),
                         tuple(np.round(uv[idx[b]]).astype(int)), (255, 255, 255), 1)
            for l, p in zip(labels, uv):
                cv2.putText(view, l, tuple(np.round(p+[2, -2]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(view, f"ffv2 {self.state.get('status', '?')}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        return view
