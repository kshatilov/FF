"""Persist one failed detection frame without publishing calibration data."""

import datetime
import json
import os
import uuid


def save_detection_failure(frame, known_corners, camera_id, surface_id,
                           projector_size, repo_root):
    """Return paths and measured facts; no automatic exposure/pose inference."""
    import cv2

    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
        parameters)
    _, ids, rejected = detector.detectMarkers(frame)
    detected = [] if ids is None else sorted(set(map(int, ids.ravel())))
    known = sorted(set(detected).intersection(known_corners))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    directory = os.path.join(repo_root, "smoke", "hg_failures")
    os.makedirs(directory, exist_ok=True)
    base = os.path.join(directory, "%s_c%d_%s" % (
        timestamp, camera_id, uuid.uuid4().hex[:8]))
    frame_path = base + ".png"
    metadata_path = base + ".json"
    summary = {
        "camera_id": camera_id,
        "surface_id": surface_id,
        "capture_top": "/project1/cam_and_kinect",
        "camera_size": [int(frame.shape[1]), int(frame.shape[0])],
        "projector_size": list(projector_size),
        "detected_ids": detected,
        "known_ids": known,
        "known_count": len(known),
        "expected_count": len(known_corners),
        "rejected_candidates": len(rejected),
        "near_white_fraction": float((gray >= 250).mean()),
        "near_white_definition": "fraction of entire frame with grayscale >= 250/255",
        "frame_path": frame_path,
        "metadata_path": metadata_path,
        "note": (
            "This is the last failed camera frame, not a calibration. "
            "Clipping can hide projected markers, but the full-frame statistic "
            "alone does not identify the projection surface or prove a cause."),
    }
    encoded, image = cv2.imencode(".png", frame)
    if not encoded:
        raise ValueError("could not encode failed camera frame")
    with open(frame_path + ".tmp", "wb") as stream:
        stream.write(image.tobytes())
    os.replace(frame_path + ".tmp", frame_path)
    with open(metadata_path + ".tmp", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(metadata_path + ".tmp", metadata_path)
    return summary
