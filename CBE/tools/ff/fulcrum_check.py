"""fulcrum_check.py - review the assembled fulcrum from the webcam.

Grabs one frame from the Logi webcam (index 0, 640x480), detects
DICT_4X4_50 markers, annotates + saves to smoke/fulcrum_webcam.png
(raw saved as fulcrum_webcam_raw.png).  When fulcrum markers (ids
10-14) are found, solves the webcam pose against
fulcrum_spec.json with a non-planar solvePnP and prints the pose +
reprojection error.

Webcam intrinsics: tools/pipe/ff/cheb_view.json (f_px=516, same as
aruco_solve.py).

Run preview_feeds.py paused first (curl 127.0.0.1:8971/pause) - the
feed holds the webcam exclusively.
"""
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE_FF = os.path.join(HERE, "..", "pipe", "ff")
SPEC = os.path.join(PIPE_FF, "fulcrum_spec.json")
CHEB = os.path.join(PIPE_FF, "cheb_view.json")
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "smoke"))
WEBCAM_INDEX = 0
WH = (640, 480)
FULCRUM_IDS = set(range(10, 15))


def detect(gray):
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    par = cv2.aruco.DetectorParameters()
    par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(dic, par)
    c, ids, _ = det.detectMarkers(gray)
    return {} if ids is None else {int(i): c[k].reshape(4, 2)
                                   for k, i in enumerate(ids.flatten())}


def main():
    spec = json.load(open(SPEC))
    world = {int(i): np.float64(m["corners_TL_TR_BR_BL"])
             for i, m in spec["markers"].items()}
    cheb = json.load(open(CHEB))
    f = cheb["cam"]["f_px"]
    K = np.float64([[f, 0, WH[0] / 2], [0, f, WH[1] / 2], [0, 0, 1]])

    cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_MSMF)
    if not cap.isOpened():
        print("webcam: cannot open index", WEBCAM_INDEX)
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WH[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WH[1])
    frame = None
    for _ in range(10):
        ok, fr = cap.read()
        if ok:
            frame = fr
    cap.release()
    if frame is None:
        print("webcam: no frame")
        return

    os.makedirs(OUT, exist_ok=True)
    cv2.imwrite(os.path.join(OUT, "fulcrum_webcam_raw.png"), frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = detect(gray)
    print("markers:", sorted(found))
    for i, c in sorted(found.items()):
        side = np.mean([np.linalg.norm(c[j] - c[(j + 1) % 4])
                        for j in range(4)])
        tag = "FULCRUM" if i in FULCRUM_IDS else "other"
        print(f"  id{i}: {side:.0f}px side  ({tag})")

    vis = frame.copy()
    if found:
        cv2.aruco.drawDetectedMarkers(
            vis, np.array([c.reshape(1, 4, 2) for c in found.values()]),
            np.array([[i] for i in found]))

    ful = {i: c for i, c in found.items() if i in world}
    if ful:
        obj = np.vstack([world[i] for i in ful]).reshape(-1, 1, 3)
        img = np.vstack([ful[i] for i in ful]).reshape(-1, 1, 2)
        ok, rv, tv = cv2.solvePnP(obj, img, K, np.zeros(5),
                                  flags=cv2.SOLVEPNP_SQPNP)
        if ok:
            rv, tv = cv2.solvePnPRefineLM(obj, img, K, np.zeros(5),
                                          rv, tv)
            pred, _ = cv2.projectPoints(obj, rv, tv, K, np.zeros(5))
            e = np.linalg.norm(pred.reshape(-1, 2) - img.reshape(-1, 2),
                               axis=1)
            R = cv2.Rodrigues(rv)[0]
            pos = -R.T @ tv
            print(f"solvePnP ({len(ful)} markers, {len(e)} corners): "
                  f"reproj {e.mean():.2f}px mean / {e.max():.2f}px max")
            print(f"  cam pos in fulcrum frame (mm): "
                  f"{pos.ravel().round(1)}")
            cv2.drawFrameAxes(vis, K, np.zeros(5), rv, tv, 40)
    cv2.imwrite(os.path.join(OUT, "fulcrum_webcam.png"), vis)
    print("saved", os.path.join(OUT, "fulcrum_webcam.png"))


if __name__ == "__main__":
    main()
