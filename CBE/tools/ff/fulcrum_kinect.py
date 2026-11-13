"""fulcrum_kinect.py - live Kinect color preview of the fulcrum pose solve.

Shows the Kinect v2 color feed (1920x1080) with every detected
DICT_4X4_50 marker labelled by id.  Fulcrum markers (ids 10-14) feed a
non-planar solvePnP against fulcrum_spec.json; the fulcrum frame axes
are drawn at the base centre plus a small axis triplet at each detected
fulcrum marker.

NOTE: the Kinect v2 color stream arrives MIRRORED - ArUco candidates are
found but bit-decode fails on the raw frame.  The script detects on the
raw / flipV / flipH variants and keeps the branch whose fulcrum solve
has the lowest reprojection error (wrong mirror corner order fits the
non-planar spec badly).  Pose is solved in the corrected (true-view)
image space with the principal point flipped accordingly.

Keys: ESC/q quit, s save annotated frame to smoke/.
Usage:  python fulcrum_kinect.py [--frames N]   (N>0: headless test run)
"""
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "kinect"))
sys.path.insert(0, os.path.join(ROOT, "tools", "td"))
from kincap import Kin                                    # noqa: E402
from aruco_solve import kinect_pinhole_init              # noqa: E402

SPEC = os.path.join(ROOT, "tools", "pipe", "ff", "fulcrum_spec.json")
OUT = os.path.join(ROOT, "smoke")
AXIS_BIG, AXIS_SMALL = 60.0, 25.0
FLIPS = {"raw": None, "flipV": 0, "flipH": 1}   # cv2.flip codes
CW, CH = 1920, 1080


def detect(gray):
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    par = cv2.aruco.DetectorParameters()
    par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    par.minOtsuStdDev = 2.0
    par.errorCorrectionRate = 0.8
    det = cv2.aruco.ArucoDetector(dic, par)
    c, ids, _ = det.detectMarkers(gray)
    return {} if ids is None else {int(i): c[k].reshape(4, 2)
                                   for k, i in enumerate(ids.flatten())}


def k_for_flip(K, name):
    """Adjust principal point for the applied image flip."""
    K2 = K.copy()
    if name == "flipH":
        K2[0, 2] = (CW - 1) - K2[0, 2]
    elif name == "flipV":
        K2[1, 2] = (CH - 1) - K2[1, 2]
    return K2


def solve_fulcrum(world, found, K):
    """solvePnP over all fulcrum marker corners -> (rv, tv, emean, emax)."""
    ful = {i: c for i, c in found.items() if i in world}
    if not ful:
        return None, ful
    obj = np.vstack([world[i] for i in ful]).reshape(-1, 1, 3)
    img = np.vstack([ful[i] for i in ful]).reshape(-1, 1, 2)
    ok, rv, tv = cv2.solvePnP(obj, img, K, np.zeros(5),
                              flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None, ful
    rv, tv = cv2.solvePnPRefineLM(obj, img, K, np.zeros(5), rv, tv)
    pred, _ = cv2.projectPoints(obj, rv, tv, K, np.zeros(5))
    e = np.linalg.norm(pred.reshape(-1, 2) - img.reshape(-1, 2), axis=1)
    return (rv, tv, float(e.mean()), float(e.max())), ful


def scan_flips(gray, world, K_raw):
    """Detect+solve on every flip variant -> {name: (found, sol)}."""
    out = {}
    for name, code in FLIPS.items():
        g = gray if code is None else cv2.flip(gray, code)
        found = detect(g)
        ful = {i: c for i, c in found.items() if i in world}
        if ful:
            sol, _ = solve_fulcrum(world, found, k_for_flip(K_raw, name))
            out[name] = (found, sol)
    return out


def pick_branch(branches):
    """Best branch: solve exists, lowest reproj error."""
    best = None
    for name, (found, sol) in branches.items():
        if sol and (best is None or sol[2] < best[1][2]):
            best = (name, sol)
    return best


def draw_marker_axes(vis, K, R_fw, t_fw, world, found):
    """Small axis triplet at each fulcrum marker centre."""
    for i, c in found.items():
        if i not in world:
            continue
        W = world[i]
        x = (W[1] - W[0] + W[2] - W[3])
        y = (W[3] - W[0] + W[2] - W[1])
        x /= np.linalg.norm(x)
        y /= np.linalg.norm(y)
        Rm = np.stack([x, y, np.cross(x, y)], axis=1)
        R = R_fw @ Rm
        rv = cv2.Rodrigues(R)[0]
        tv = R_fw @ W.mean(axis=0) + t_fw
        cv2.drawFrameAxes(vis, K, np.zeros(5), rv, tv, AXIS_SMALL)


def main():
    n_frames = 0
    if "--frames" in sys.argv:
        n_frames = int(sys.argv[sys.argv.index("--frames") + 1])
    world = {int(i): np.float64(m["corners_TL_TR_BR_BL"])
             for i, m in json.load(open(SPEC))["markers"].items()}

    k = Kin("color", "depth")
    frame = None
    for _ in range(15):                    # mapper needs live streams
        frame = k.grab("color", timeout=1)
        k.grab("depth", timeout=1)
        if frame is not None:
            break
    K_raw = kinect_pinhole_init(k)
    print(f"kinect pinhole f=({K_raw[0,0]:.1f},{K_raw[1,1]:.1f}) "
          f"c=({K_raw[0,2]:.1f},{K_raw[1,2]:.1f})")

    active = "flipH"                       # kinect v2 is H-mirrored
    miss = 99                              # force flip scan on first frame
    cv2.namedWindow("fulcrum-kinect", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("fulcrum-kinect", 1280, 720)
    t_last = time.time()
    n = 0
    while True:
        raw = k.grab("color", timeout=1)
        if raw is None:
            continue
        n += 1
        t0 = time.time()
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

        g = gray if FLIPS[active] is None else cv2.flip(gray, FLIPS[active])
        found = detect(g)
        ful = {i: c for i, c in found.items() if i in world}
        if ful:
            miss = 0
            sol, _ = solve_fulcrum(world, found, k_for_flip(K_raw, active))
        else:
            miss += 1
            sol = None
            if miss > 5:
                branches = scan_flips(gray, world, K_raw)
                best = pick_branch(branches)
                if best:
                    active, sol = best
                    g = (gray if FLIPS[active] is None
                         else cv2.flip(gray, FLIPS[active]))
                    found = branches[active][0]
                    ful = {i: c for i, c in found.items() if i in world}
                    print(f"flip branch -> {active}")
                    miss = 0

        vis = (raw if FLIPS[active] is None
               else cv2.flip(raw, FLIPS[active])).copy()
        if found:
            cv2.aruco.drawDetectedMarkers(
                vis,
                np.array([c.reshape(1, 4, 2) for c in found.values()]),
                np.array([[i] for i in found]))

        hud = [f"markers {sorted(found)} [{active}]"]
        if sol:
            K = k_for_flip(K_raw, active)
            rv, tv, em, ex = sol
            R_fw = cv2.Rodrigues(rv)[0]
            cv2.drawFrameAxes(vis, K, np.zeros(5), rv, tv, AXIS_BIG)
            draw_marker_axes(vis, K, R_fw, tv.ravel(), world, ful)
            pos = -R_fw.T @ tv.ravel()
            hud.append(f"pnp {em:.2f}/{ex:.2f}px ({len(ful)} markers)")
            hud.append(f"cam@ {[round(v) for v in pos]} mm")
        hud.append(f"{(time.time() - t0) * 1000:.0f}ms")
        for j, t in enumerate(hud):
            cv2.putText(vis, t, (20, 40 + 36 * j),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2,
                        cv2.LINE_AA)
        dt = time.time() - t_last
        t_last = time.time()
        if n % 30 == 0:
            print(f"frame {n}: {1 / max(dt, 1e-3):.0f}fps-ish | "
                  + " | ".join(hud))
        if n_frames:
            cv2.imwrite(os.path.join(OUT, "fulcrum_kinect.png"), vis)
            if n >= n_frames:
                print("saved", os.path.join(OUT, "fulcrum_kinect.png"))
                break
        cv2.imshow("fulcrum-kinect", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("s"):
            cv2.imwrite(os.path.join(OUT, "fulcrum_kinect.png"), vis)
            print("saved", os.path.join(OUT, "fulcrum_kinect.png"))
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
