"""fulcrum_project.py - projector output driven by fulcrum_track.

Modes (from fulcrum_poses.json "proj_cmd", polled each frame):

  board  - ArUco marker board (aruco_show.build_board).
  dots   - structured-light calibration: one white dot on black at
           cmd["dots"][idx]; idx advances every DWELL_S; the current
           index/position is published to proj_state.json so the
           tracker can correlate camera observations.  When the list
           is exhausted writes mode="dots_done" and holds black.
  colors - per-face assigned colours (FACE_COLORS): each fulcrum
           face's trapezoid is projected through the solved projector
           pinhole (pose polled live, so colours TRACK if the
           projector or the fulcrum moves).  Cameras see the colours
           on the faces and fulcrum_track asserts them.

Run while fulcrum_track.py is running.  ESC on the window quits.
"""
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "projector"))
sys.path.insert(0, os.path.join(HERE))
from camcalib import enum_monitors, fullscreen                 # noqa: E402
from aruco_show import build_board, pick_monitor               # noqa: E402
from fulcrum_track import FACE_COLORS, face_polys              # noqa: E402

POSES = os.path.join(ROOT, "tools", "pipe", "ff",
                    "fulcrum_poses.json")
STATE = os.path.join(ROOT, "tools", "pipe", "ff", "proj_state.json")
CANVAS = (1280, 720)
DWELL_S = 0.45         # per-dot dwell during structured-light calib
DOT_R = 16             # canvas px radius


def load_pose():
    try:
        return json.load(open(POSES))
    except Exception:
        return None


def write_state(d):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(d, fh)
    for _ in range(5):
        try:
            os.replace(tmp, STATE)
            return
        except PermissionError:
            time.sleep(0.02)


def project_px(pts3, K, R, t):
    P = (R @ np.float64(pts3).T).T + t
    z = np.where(np.abs(P[:, 2]) < 1e-6, 1e-6, P[:, 2])
    return np.stack([K[0, 2] + K[0, 0] * P[:, 0] / z,
                     K[1, 2] + K[1, 0] * P[:, 1] / z], axis=1)


def face_canvas(polys, pj):
    """Per-face canvas quad through the solved projector pinhole."""
    K = np.float64([[pj["f"], 0, pj["cx"]],
                    [0, pj["f"], pj["cy"]], [0, 0, 1]])
    R = cv2.Rodrigues(np.float64(pj["rvec"]))[0]
    t = -R @ np.float64(pj["pos_mm"])
    return {name: project_px(poly, K, R, t)
            for name, poly in polys.items()}


def render_faces(quads, margin=0.94):
    img = np.zeros((CANVAS[1], CANVAS[0], 3), np.uint8)
    for name, q in quads.items():
        col = FACE_COLORS.get(name, (80, 80, 80))
        c = q.mean(axis=0)
        q2 = c + (q - c) * margin          # erode: avoid bleeding over
        cv2.fillPoly(img, [np.int32(q2)], col)
    for name, q in quads.items():
        cv2.polylines(img, [np.int32(q)], True, (255, 255, 255), 1)
    return img


def render_dot(uv, idx, total):
    img = np.zeros((CANVAS[1], CANVAS[0], 3), np.uint8)
    cv2.circle(img, (int(round(uv[0])), int(round(uv[1]))),
               DOT_R, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(img, f"{idx + 1}/{total}", (24, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90, 90, 90), 2,
                cv2.LINE_AA)
    return img


def main():
    mons = enum_monitors()
    mon = pick_monitor(mons)
    print("projector monitor:", mon)
    board_img, _, _ = build_board(mon[2], mon[3])
    polys = face_polys()
    fullscreen(board_img, mon, name="fulcrum")
    print("waiting for proj_cmd from fulcrum_track")
    seq, dots, idx, switch_t = -1, [], -1, 0.0
    last_mode = None
    while True:
        data = load_pose() or {}
        cmd = data.get("proj_cmd") or {}
        mode = cmd.get("mode") or \
            ("colors" if data.get("projector") else "board")
        img = None
        if mode == "dots" and cmd.get("dots"):
            if cmd.get("seq") != seq:
                seq, dots, idx = cmd["seq"], cmd["dots"], -1
                switch_t = 0.0
            if time.time() - switch_t >= DWELL_S:
                idx += 1
                switch_t = time.time()
                if idx < len(dots):
                    u, v = dots[idx]
                    write_state({"mode": "dots", "seq": seq,
                                 "idx": idx, "u": u, "v": v,
                                 "t": switch_t})
                else:
                    write_state({"mode": "dots_done", "seq": seq,
                                 "t": switch_t})
                    print(f"seq {seq}: {len(dots)} dots emitted")
            img = (render_dot(dots[idx], idx, len(dots))
                   if 0 <= idx < len(dots)
                   else np.zeros((CANVAS[1], CANVAS[0], 3), np.uint8))
        elif mode == "colors" and (data.get("face_quads") or
                                   data.get("projector")):
            quads = (face_canvas(polys, data["projector"])
                     if data.get("projector") else {})
            quads.update({k: np.float64(v)
                          for k, v in (data.get("face_quads") or
                                       {}).items()})
            img = render_faces(quads,
                               data.get("quad_margin", 0.94))
            if last_mode != "colors":
                write_state({"mode": "colors", "t": time.time()})
        elif mode == "dark":
            # dark canvas while the fulcrum is tracked but uncalibrated -
            # board light washes the side markers out, black keeps them
            # decodable so the camera can re-acquire before a sweep
            img = np.zeros((CANVAS[1], CANVAS[0], 3), np.uint8)
            if last_mode != "dark":
                write_state({"mode": "dark", "t": time.time()})
        else:
            img = board_img
            if last_mode != "board":
                write_state({"mode": "board", "t": time.time()})
        if mode != last_mode:
            print("mode ->", mode)
            last_mode = mode
        fullscreen(img, mon, name="fulcrum")
        if cv2.waitKey(30) & 0xFF == 27:
            break


if __name__ == "__main__":
    main()
