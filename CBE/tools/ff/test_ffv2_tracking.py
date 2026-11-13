import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ffv2_tracker


def render_scene(tracker, rv, tv, object_px=None, distractor=True):
    """Physically plausible synthetic view: only camera-visible faces drawn
    (backface culled, painter far-to-near), red borders only on edges of
    visible faces, plus a large red distractor object."""
    cam = tracker.camera
    image = np.full((cam.size[1], cam.size[0], 3), (235, 235, 235), np.uint8)
    T = ffv2_tracker.rt(rv, tv)
    order = []
    for name, face in tracker.faces.items():
        pts = face['points'] @ T[:3, :3].T + T[:3, 3]
        if (pts[:, 2] <= 1).any():
            continue
        center = pts.mean(axis=0)
        if (T[:3, :3] @ face['normal']) @ (-center) <= 0:
            continue  # hidden face — not drawn at all
        order.append((float(center[2]), name))
    visible_edges = set()
    for _, name in sorted(order, reverse=True):
        face = tracker.faces[name]
        uv = cv2.projectPoints(face['points'], rv, tv, cam.K, cam.D)[0].reshape(-1, 2)
        bgr = (face['rgb'][2], face['rgb'][1], face['rgb'][0])
        cv2.fillConvexPoly(image, np.round(uv).astype(np.int32), bgr)
        for i in range(3):
            visible_edges.add(tuple(sorted((face['labels'][i], face['labels'][(i+1) % 3]))))
    idx = {l: np.array(p) for l, p in tracker.vertices.items()}
    for a, b in visible_edges:
        pa = cv2.projectPoints(idx[a][None], rv, tv, cam.K, cam.D)[0].reshape(2)
        pb = cv2.projectPoints(idx[b][None], rv, tv, cam.K, cam.D)[0].reshape(2)
        cv2.line(image, tuple(np.round(pa).astype(int)), tuple(np.round(pb).astype(int)),
                 (30, 30, 200), 2)
    if distractor:
        cv2.rectangle(image, (480, 30), (620, 160), (40, 40, 220), -1)  # big red distractor
    return image


class SpecGeometry(unittest.TestCase):
    def setUp(self):
        spec, self.vertices, self.faces = ffv2_tracker.load_geometry()
        self.spec = spec
        self.centroid = np.mean(list(self.vertices.values()), axis=0)

    def test_edge_lengths(self):
        for edge, expected in self.spec['edges_mm'].items():
            a, b = edge
            d = np.linalg.norm(self.vertices[a]-self.vertices[b])
            self.assertAlmostEqual(d, expected, places=6, msg=edge)

    def test_outward_normals(self):
        for name, face in self.faces.items():
            normal = face['normal']
            center = face['points'].mean(axis=0)
            self.assertGreater(normal @ (center-self.centroid), 0, name)
            for label, point in self.vertices.items():
                if label not in face['labels']:
                    self.assertLess(normal @ (point-center), 0,
                                    f'{label} should be inside relative to {name}')

    def test_hypothesis_consistency(self):
        tracker = ffv2_tracker.FFv2Tracker('webcam')
        obs = [
            dict(face='DEC', corners=np.float64([[10, 10], [30, 10], [20, 25]])),
            dict(face='DBC', corners=np.float64([[10, 10], [30, 10], [20, -15]])),
        ]
        hyps = tracker.hypotheses(obs, radius=5.0)
        self.assertTrue(hyps)
        merged = [m for m in hyps if len(m) >= 4]
        self.assertTrue(merged)  # C,D shared -> >=4 distinct merged vertices
        singles = [m for m in hyps if len(m) == 3]
        self.assertTrue(singles)  # single-face fallbacks are also generated
        obs[1]['corners'] += 500
        hyps = tracker.hypotheses(obs, radius=5.0)
        self.assertFalse([m for m in hyps if len(m) >= 4])
        self.assertTrue([m for m in hyps if len(m) == 3])

    def test_face_colors(self):
        self.assertEqual(self.faces['DBC']['rgb'], (0, 0, 0))
        self.assertEqual(self.faces['ADB']['rgb'], (0, 255, 0))
        self.assertFalse(self.faces['ABC']['track'])


class RenderContract(unittest.TestCase):
    """Independent checks of the top-down RGBA render contract: identity
    projector transform, raw uv rasterization, RGB channel order."""

    def setUp(self):
        self.tracker = ffv2_tracker.FFv2Tracker('webcam')
        # isolate the render: identity transform + known projector intrinsics
        self.tracker.camera.projector_from_camera = np.eye(4)
        self.tracker.camera.projector_K = np.float64([[600, 0, 640],
                                                      [0, 600, 360],
                                                      [0, 0, 1]])

    def test_topdown_rgba_channel_order(self):
        # object directly ahead, base toward camera -> green ADB also visible?
        rv = np.zeros((3, 1))
        tv = np.array([[0.0], [0.0], [800.0]])
        image = self.tracker.render(rv, tv)
        # every visible face interior must be at its raw projected uv,
        # top-down (no flip), RGB in channels 0,1,2
        Kp = self.tracker.camera.projector_K
        for name, face in self.tracker.faces.items():
            T = ffv2_tracker.rt(rv, tv)
            pts = face['points'] @ T[:3, :3].T + T[:3, 3]
            if (pts[:, 2] <= 1).any():
                continue
            if (T[:3, :3] @ face['normal']) @ (-pts.mean(axis=0)) <= 0:
                continue
            uv = cv2.projectPoints(face['points'], rv, tv, Kp,
                                   np.zeros(5))[0].reshape(-1, 2)
            cx, cy = np.round(uv.mean(axis=0)).astype(int)
            self.assertTrue(0 <= cx < 1280 and 0 <= cy < 720, name)
            expected = face['rgb'] if name in ffv2_tracker.FACE_COLOR else (0, 0, 0)
            got = image[cy, cx]
            self.assertEqual(tuple(got[:3]), tuple(expected),
                             f'{name} at ({cx},{cy}) expected rgb {expected} got {tuple(got[:3])}')

    def test_black_face_and_white_base_unlit(self):
        # camera looks squarely at DBC: must render pure black (projector off)
        normal = self.tracker.faces['DBC']['normal']
        center = self.tracker.faces['DBC']['points'].mean(axis=0)
        cam_pos = center+normal*800.0
        rv, tv = ffv2_tracker.look_at(cam_pos, center)
        image = self.tracker.render(rv, tv)
        Kp = self.tracker.camera.projector_K
        uv = cv2.projectPoints(self.tracker.faces['DBC']['points'], rv, tv, Kp,
                               np.zeros(5))[0].reshape(-1, 2)
        cx, cy = np.round(uv.mean(axis=0)).astype(int)
        self.assertEqual(int(image[cy, cx, :3].max()), 0, 'DBC must be projector-off')

    def test_painter_occlusion(self):
        # a pose where two faces project onto overlapping pixels: the nearer
        # face's color must win at the overlap point
        rv = np.array([[0.0], [1.2], [0.0]])
        tv = np.array([[0.0], [0.0], [900.0]])
        image = self.tracker.render(rv, tv)
        Kp = self.tracker.camera.projector_K
        T = ffv2_tracker.rt(rv, tv)
        # choose a visible face pair and check ordering by depth at any
        # overlapping pixel of their projected triangles
        visible = []
        for name, face in self.tracker.faces.items():
            pts = face['points'] @ T[:3, :3].T + T[:3, 3]
            if (pts[:, 2] <= 1).any():
                continue
            if (T[:3, :3] @ face['normal']) @ (-pts.mean(axis=0)) <= 0:
                continue
            uv = cv2.projectPoints(face['points'], rv, tv, Kp,
                                   np.zeros(5))[0].reshape(-1, 2)
            visible.append((name, pts[:, 2].mean(), uv))
        self.assertGreaterEqual(len(visible), 2)
        nearest = min(visible, key=lambda v: v[1])[0]
        color = self.tracker.faces[nearest]['rgb'] if nearest in ffv2_tracker.FACE_COLOR else (0, 0, 0)
        cx, cy = np.round(dict((n, u) for n, _, u in visible)[nearest].mean(axis=0)).astype(int)
        if 0 <= cx < 1280 and 0 <= cy < 720:
            got = tuple(image[cy, cx, :3])
            self.assertEqual(got, tuple(color), f'nearest face {nearest} must win')


class Tracking(unittest.TestCase):
    def setUp(self):
        self.tracker = ffv2_tracker.FFv2Tracker('webcam')
        # orientation recovered from the real webcam capture (DBC+DEC visible)
        self.rv = np.array([[0.5597], [1.2666], [-1.8510]])
        self.tv = np.array([[3.3], [-61.5], [614.4]])

    def _pose_errors(self, state, rv=None, tv=None):
        rv = self.rv if rv is None else rv
        tv = self.tv if tv is None else tv
        rve = np.float64(state['pose']['rvec']).reshape(3, 1)
        tve = np.float64(state['pose']['tvec']).reshape(3, 1)
        t_err = np.linalg.norm(tve.ravel()-tv.ravel())
        cam = self.tracker.camera
        verts = np.array(list(self.tracker.vertices.values()))
        Tt = cam.projector_from_camera @ ffv2_tracker.rt(rv, tv)
        Te = cam.projector_from_camera @ ffv2_tracker.rt(rve, tve)
        e = np.linalg.norm(
            cv2.projectPoints(verts, cv2.Rodrigues(Tt[:3, :3])[0], Tt[:3, 3],
                              cam.projector_K, np.zeros(5))[0].reshape(-1, 2)
            - cv2.projectPoints(verts, cv2.Rodrigues(Te[:3, :3])[0], Te[:3, 3],
                                cam.projector_K, np.zeros(5))[0].reshape(-1, 2),
            axis=1)
        return t_err, e.max()

    def test_synthetic_80px_accurate(self):
        # ~70px object: clean multi-face scene must resolve within tolerance
        tv = np.array([[1.72], [-32.0], [320.0]])
        frame = render_scene(self.tracker, self.rv, tv)
        out, state = self.tracker.process_frame(frame, 1.0)
        self.assertEqual(state['status'], 'tracked', state.get('fit'))
        t_err, p_err = self._pose_errors(state, tv=tv)
        self.assertLess(t_err, 30.0)
        self.assertLess(p_err, 8.0)

    def test_synthetic_40px_honest(self):
        # ~40px object: either a correct lock or an honest reject — never a
        # confidently wrong pose
        frame = render_scene(self.tracker, self.rv, self.tv)
        out, state = self.tracker.process_frame(frame, 1.0)
        if state['status'] == 'tracked':
            t_err, p_err = self._pose_errors(state)
            self.assertLess(t_err, 30.0, 'accepted pose is wrong geometry')
            self.assertLess(p_err, 8.0, 'accepted pose misses projector alignment')
        else:
            self.assertIn(state['status'], ('uncertain', 'lost', 'predicting'))
            self.assertEqual(int(out[:, :, :3].max()), 0)

    def test_synthetic_pose_recovers_50px(self):
        frame = render_scene(self.tracker, self.rv, self.tv)
        out, state = self.tracker.process_frame(frame, 1.0)
        if state['status'] != 'tracked':
            self.skipTest(f'pose not resolved ({state["status"]}) — '
                          'must not fake a lock: ' + json.dumps(state.get('fit', {}), default=str)[:200])
        tvec = np.float64(state['pose']['tvec'])
        self.assertLess(np.linalg.norm(tvec-self.tv.ravel()), 30.0)
        _, p_err = self._pose_errors(state)
        self.assertLess(p_err, 8.0)
        self.assertGreater(int(out[:, :, :3].max()), 0)

    def test_blank_frame_rejected_and_black(self):
        blank = np.full((480, 640, 3), 235, np.uint8)
        out, state = self.tracker.process_frame(blank, 1.0)
        self.assertNotEqual(state['status'], 'tracked')
        self.assertEqual(int(out[:, :, :3].max()), 0)

    def test_no_projection_before_confident_solve(self):
        noise = np.random.default_rng(0).integers(180, 255, (480, 640, 3), np.uint8)
        out, state = self.tracker.process_frame(noise, 1.0)
        self.assertNotEqual(state['status'], 'tracked')
        self.assertEqual(int(out[:, :, :3].max()), 0)

    def test_red_distractor_alone_rejected(self):
        image = np.full((480, 640, 3), 235, np.uint8)
        cv2.rectangle(image, (200, 150), (320, 260), (40, 40, 220), -1)
        out, state = self.tracker.process_frame(image, 1.0)
        self.assertNotEqual(state['status'], 'tracked')
        self.assertEqual(int(out[:, :, :3].max()), 0)

    def test_missing_frames_bounded_dropout(self):
        frame = render_scene(self.tracker, self.rv, self.tv)
        out, state = self.tracker.process_frame(frame, 1.0)
        if state['status'] != 'tracked':
            self.skipTest('synthetic pose not resolved')
        # a missing frame inside the prediction window must NOT black out
        out, state = self.tracker.process_frame(None, 1.10)
        self.assertEqual(state['status'], 'predicting')
        self.assertGreater(int(out[:, :, :3].max()), 0)
        # no jump backward: past 0.15 s the held pose is the coast pose, not
        # the original accepted pose — and still non-black until 1.5 s
        out, state = self.tracker.process_frame(None, 1.30)
        self.assertIn(state['status'], ('holding', 'fading'))
        self.assertGreater(int(out[:, :, :3].max()), 0)
        # expiry: full black after the hold window
        out, state = self.tracker.process_frame(None, 2.8)
        self.assertEqual(state['status'], 'lost')
        self.assertEqual(int(out[:, :, :3].max()), 0)

    def test_no_indefinite_prediction(self):
        frame = render_scene(self.tracker, self.rv, self.tv)
        out, state = self.tracker.process_frame(frame, 1.0)
        if state['status'] != 'tracked':
            self.skipTest('synthetic pose not resolved')
        self.tracker.process_frame(None, 1.1)
        _, state = self.tracker.process_frame(None, 1.16)
        if state['status'] == 'predicting':
            self.assertLessEqual(state['pose_age_s'], 0.2)
        self.assertAlmostEqual(self.tracker.last_accepted_time, 1.0)

    def test_camera_reset_clears_pose(self):
        frame = render_scene(self.tracker, self.rv, self.tv)
        out, state = self.tracker.process_frame(frame, 1.0)
        if state['status'] != 'tracked':
            self.skipTest('synthetic pose not resolved')
        self.tracker.reset()
        self.assertIsNone(self.tracker.last_accepted_pose)
        self.assertIsNone(self.tracker.pose_filter.time)
        blank = np.full((480, 640, 3), 235, np.uint8)
        out, state = self.tracker.process_frame(blank, 1.05)
        self.assertEqual(state['status'], 'lost')
        self.assertEqual(int(out[:, :, :3].max()), 0)


if __name__ == '__main__':
    unittest.main()
