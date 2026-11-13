"""Camera-relative table color measurements, with TD owning all live images.

This measures the combined table/projector/camera response, not paint reflectance
or colorimetric accuracy. All optimization runs in measured camera code values.
"""

import argparse
import datetime
import hashlib
import itertools
import json
import os
import sys
import time
import uuid

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "hg"))
sys.path.insert(0, os.path.join(_HERE, "..", "td"))

import cv2
import numpy as np

import hg
import hg_geometry as geometry
import hg_wire as wire


ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CALIBRATION = os.path.join(ROOT, "out", "hg", "c0_p0.json")
# Session capture goes through the shared camera selector, which the
# session begin has already set to this measurement's camera. Metadata
# source_top still records the raw calibration camera TOP.
CAM_SWITCH_TOP = "/project1/cam_and_kinect"


def read_calibration(path, camera_id=0):
    with open(path, encoding="utf-8") as stream:
        document = json.load(stream)
    params = geometry.warp_parameters(document)
    if camera_id not in (0, 1) or params["source_id"] != str(camera_id):
        raise ValueError("Color measurements require matching RGB camera 0 or 1.")
    return document, params


def measurement_mask(document):
    """Central 70% of the projector raster, mapped into the raw camera image."""
    cw, ch = document["source"]["resolution"]
    pw, ph = document["destination"]["resolution"]
    points = np.array([[.15 * pw, .15 * ph], [.85 * pw, .15 * ph],
                       [.85 * pw, .85 * ph], [.15 * pw, .85 * ph]])
    inverse = np.linalg.inv(np.asarray(document["H_source_to_destination"]))
    polygon = geometry.transform_points(inverse, points)
    mask = np.zeros((ch, cw), np.uint8)
    cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 1)
    mask = cv2.erode(mask, np.ones((7, 7), np.uint8))
    if int(mask.sum()) < 2000:
        raise ValueError("Insufficient camera coverage for table measurements.")
    return mask.astype(bool), polygon


class Measurement:
    def __init__(self, directory, calibration=DEFAULT_CALIBRATION,
                 camera_id=0, surface_id="Table_01", client=None):
        self.directory = os.path.abspath(directory)
        os.makedirs(self.directory, exist_ok=False)
        self.document, self.params = read_calibration(calibration, camera_id)
        self.camera_id = camera_id
        self.surface_id = surface_id
        self.camera_top = self.params["source_top"]
        self.mask, polygon = measurement_mask(self.document)
        self.client = client or hg.TdClient()
        info = hg._ensure_td(self.client)
        if list(self.params["source_size"]) != hg._td_raster(
                info, self.camera_top.rsplit("/", 1)[-1]):
            raise ValueError("Camera raster changed since geometric calibration.")
        self.owner = uuid.uuid4().hex
        self.started = False
        self.records = []
        self.metadata = {
            "schema": "ff.hg.table_color.measurements",
            "schema_version": 1,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "calibration_path": os.path.abspath(calibration),
            "calibration_sha256": hashlib.sha256(
                json.dumps(self.document, sort_keys=True).encode()).hexdigest(),
            "calibration": self.document,
            "camera_top": self.camera_top,
            "camera_id": camera_id,
            "surface_id": surface_id,
            "roi_camera_polygon": polygon.tolist(),
            "roi_pixel_count": int(self.mask.sum()),
            "assumptions": [
                "Operator reports empty table and fixed camera exposure.",
                "Camera white balance must remain fixed for transferable results.",
                "Camera-relative code values, not calibrated linear radiance.",
                "Projector focus/keystone/mode and room illumination remain fixed.",
            ],
            "records": self.records,
        }
        cv2.imwrite(os.path.join(self.directory, "roi_mask.png"),
                    self.mask.astype(np.uint8) * 255)
        self.publish()

    def publish(self):
        hg._write_json_atomic(os.path.join(self.directory, "measurements.json"),
                              self.metadata)

    def __enter__(self):
        return self

    def __exit__(self, *error):
        if self.started:
            try:
                self.client.rpc({"action": "end", "owner": self.owner})
            except hg.HgError as restore_error:
                # A rejected begin creates no session. Do not replace the
                # original error, but never swallow a failed real restoration.
                if not error[0] or "no active hg session" not in str(restore_error):
                    raise
        self.metadata["restored"] = True
        self.publish()

    def show(self, rgb, label, settle=1.2, frame_count=8):
        rgb = np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)
        width, height = self.params["destination_size"]
        image = np.empty((height, width, 3), np.uint8)
        image[:] = rgb[::-1]
        payload = {"owner": self.owner, "pattern": wire.encode_array(image),
                   "camera_id": self.camera_id, "surface_id": self.surface_id}
        if not self.started:
            # Even a lost acknowledgement must attempt restoration.
            self.started = True
            self.client.rpc({"action": "begin", **payload})
        else:
            self.client.rpc({"action": "update_pattern", **payload})
        time.sleep(settle)
        values, frames = [], []
        for index in range(frame_count):
            frame = self.client.capture_color(CAM_SWITCH_TOP)
            if tuple(frame.shape[:2]) != self.mask.shape:
                raise ValueError("Camera dimensions changed during measurement.")
            frames.append(frame)
            values.append(np.mean(frame[self.mask][:, ::-1], axis=0))
            # Vary the sampling phase to average DLP/rolling-shutter color bands.
            time.sleep((.071, .113, .167, .097)[index % 4])
        pixels = np.concatenate([f[self.mask][:, ::-1] for f in frames])
        record = {
            "index": len(self.records),
            "label": label,
            "projector_rgb": rgb.tolist(),
            "camera_rgb": np.mean(values, axis=0).tolist(),
            "camera_statistic": "temporal mean of spatial ROI means",
            "frame_rgb": np.asarray(values).tolist(),
            "temporal_spread_rgb": np.ptp(values, axis=0).tolist(),
            "clipped_fraction_rgb": (pixels >= 250).mean(axis=0).tolist(),
        }
        filename = "%03d_%s.png" % (record["index"], label)
        cv2.imwrite(os.path.join(self.directory, filename), frames[-1])
        record["camera_frame"] = filename
        self.records.append(record)
        self.publish()
        print(json.dumps(record), flush=True)
        return np.asarray(record["camera_rgb"], float)


def probe(directory):
    with Measurement(directory) as session:
        for value in (0, 32, 64, 96, 128, 160, 192, 224, 255):
            session.show([value] * 3, "gray_%03d" % value)
        for channel in range(3):
            rgb = [0, 0, 0]
            rgb[channel] = 128
            session.show(rgb, "primary_%d_128" % channel)
        session.show([128] * 3, "gray_128_repeat")
        session.show([0] * 3, "black_repeat")


def neutral_drive(camera_rgb, jacobian, current, maximum):
    """Box-constrained local gray balance; never amplify a projector channel."""
    projector = np.eye(3) - np.ones((3, 3)) / 3
    regularization = np.sqrt(.002)
    matrix = np.vstack((projector @ jacobian, regularization * np.eye(3)))
    target = np.concatenate((
        projector @ (jacobian @ current - camera_rgb),
        regularization * np.full(3, maximum)))
    best = None
    for status in itertools.product((-1, 0, 1), repeat=3):
        free = np.flatnonzero(np.asarray(status) == 0)
        fixed = np.flatnonzero(np.asarray(status) != 0)
        drive = np.zeros(3)
        drive[np.asarray(status) == 1] = maximum
        rhs = target - matrix[:, fixed] @ drive[fixed]
        if len(free):
            drive[free] = np.linalg.lstsq(matrix[:, free], rhs, rcond=None)[0]
        if np.any(drive < -1e-6) or np.any(drive > maximum + 1e-6):
            continue
        score = float(np.sum((matrix @ drive - target) ** 2))
        if best is None or score < best[0]:
            best = (score, np.clip(drive, 0, maximum))
    if best is None:
        raise ValueError("No feasible attenuation-only gray balance.")
    return best[1]


def accept_model(model, records, restored):
    """Conservative automatic gate for new camera-relative measurements."""
    if not restored:
        raise ValueError("TD restoration not verified; color profile not published.")
    gains = np.asarray(model["gain_rgb"], float)
    if gains.shape != (3,) or not np.isfinite(gains).all():
        raise ValueError("Nonfinite color gains.")
    if np.any(gains < 0) or np.any(gains > 1):
        raise ValueError("Color correction would amplify a channel.")
    metrics = [model["headroom"], model["jacobian_condition"],
               *model["ambient_start_rgb"], *model["ambient_end_rgb"],
               *model["repeat_reference_drift_rgb"]]
    metrics.extend(value for item in model["validation"]
                   for value in (item["before_channel_spread"],
                                 item["after_channel_spread"]))
    metrics.extend(value for record in records
                   if record["label"].startswith("validation_")
                   for value in record["clipped_fraction_rgb"])
    if not np.isfinite(np.asarray(metrics, float)).all():
        raise ValueError("Nonfinite measurement diagnostics.")
    if not 0 < model["headroom"] <= 1 or model["jacobian_condition"] > 100:
        raise ValueError("Invalid headroom or unstable response model.")
    ambient_drift = float(np.max(np.abs(
        np.asarray(model["ambient_end_rgb"]) - model["ambient_start_rgb"])))
    reference_drift = float(np.max(np.abs(model["repeat_reference_drift_rgb"])))
    if ambient_drift > 2 or reference_drift > 12:
        raise ValueError("Camera/ambient response drifted; fix exposure/WB/lighting.")
    validation = model["validation"]
    if len(validation) != 4 or any(
            item["after_channel_spread"] > item["before_channel_spread"] + 1
            for item in validation):
        raise ValueError("Correction worsened a validation gray; profile rejected.")
    meaningful = [item for item in validation[1:]
                  if item["before_channel_spread"] >= 5]
    improved = [item for item in meaningful
                if item["after_channel_spread"] <= .9 * item["before_channel_spread"]]
    if len(improved) < min(2, len(meaningful)):
        raise ValueError("Insufficient independently measured gray-balance improvement.")
    checked = [record for record in records
               if record["label"].startswith("validation_")]
    if len(checked) != 8 or any(
            max(record["clipped_fraction_rgb"]) >= .01 for record in checked):
        raise ValueError("Validation pixels clipped; profile rejected.")
    model["accepted"] = True
    model["acceptance"] = {
        "decision": "Passed camera-relative gray-balance validation.",
        "ambient_drift_max_codes": ambient_drift,
        "reference_drift_max_codes": reference_drift,
        "white_balance_lock_verified": False,
        "warning": ("Reference drift exceeds 3 codes; fixed white balance "
                    "and stable light are required." if reference_drift > 3 else ""),
    }
    return model


def calibrate(directory, calibration=DEFAULT_CALIBRATION,
              camera_id=0, surface_id="Table_01", client=None):
    """Measure a local RGB Jacobian; validate the resulting gains at four tones."""
    with Measurement(directory, calibration, camera_id, surface_id, client) as session:
        baseline = session.show([0] * 3, "ambient_start", settle=2)
        usable_levels = []
        for level in (32, 64, 96, 128, 160, 192, 224, 255):
            session.show([level] * 3, "headroom_%03d" % level)
            if max(session.records[-1]["clipped_fraction_rgb"]) >= .01:
                break
            usable_levels.append(level)
        if not usable_levels or max(usable_levels) < 64:
            raise ValueError("Insufficient unclipped projector range; reduce exposure.")
        maximum = max(usable_levels)
        reference_level = min(128, maximum - 24)
        step_size = min(24, max(4, reference_level // 5))
        base = np.full(3, float(reference_level))
        reference = session.show(base, "reference_start", settle=2)
        columns = []
        for channel in range(3):
            lower, upper = base.copy(), base.copy()
            lower[channel] -= step_size
            upper[channel] += step_size
            low = session.show(lower, "jacobian_%d_minus" % channel)
            high = session.show(upper, "jacobian_%d_plus" % channel)
            columns.append((high - low) / (2 * step_size))
        jacobian = np.asarray(columns).T
        if not np.isfinite(jacobian).all() or np.linalg.cond(jacobian) > 100:
            raise ValueError("Unstable measured RGB response; do not apply correction.")
        current = base.copy()
        camera = session.show(current, "reference_repeat", settle=2)
        reference_repeat = camera.copy()
        candidates = [(float(np.ptp(camera)), current.copy(), camera.copy())]
        for step in range(3):
            proposed = np.rint(neutral_drive(
                camera, jacobian, current, reference_level))
            if np.max(np.abs(proposed - current)) < 1:
                break
            camera = session.show(proposed, "balance_%d" % step, settle=2)
            current = proposed
            candidates.append((float(np.ptp(camera)), current.copy(), camera.copy()))
        _, best_drive, balanced = min(candidates, key=lambda candidate: candidate[0])
        gains = best_drive / reference_level
        headroom = maximum / 255
        validation = []
        for nominal in (64, 128, 192, 255):
            neutral = np.rint(np.full(3, nominal * headroom))
            corrected = np.rint(neutral * gains)
            before = session.show(neutral, "validation_%d_before" % nominal)
            after = session.show(corrected, "validation_%d_after" % nominal)
            validation.append({
                "nominal": nominal,
                "uncorrected_drive_rgb": neutral.tolist(),
                "corrected_drive_rgb": corrected.tolist(),
                "before_camera_rgb": before.tolist(),
                "after_camera_rgb": after.tolist(),
                "before_channel_spread": float(np.ptp(before)),
                "after_channel_spread": float(np.ptp(after)),
            })
        final_baseline = session.show([0] * 3, "ambient_end", settle=2)
        model = {
            "schema": "ff.hg.table_color",
            "schema_version": 1,
            "method": "global_attenuation_gray_balance",
            "reference": "camera_relative_neutral_rgb",
            "source_id": str(camera_id),
            "source_top": session.camera_top,
            "destination_id": "P0",
            "surface_id": surface_id,
            "source_size": session.params["source_size"],
            "destination_size": session.params["destination_size"],
            "calibration_sha256": session.metadata["calibration_sha256"],
            "measurement_directory": session.directory,
            "gain_rgb": gains.tolist(),
            "headroom": headroom,
            "reference_drive": reference_level,
            "balanced_drive_rgb": best_drive.tolist(),
            "balanced_camera_rgb": balanced.tolist(),
            "jacobian_camera_codes_per_projector_code": jacobian.tolist(),
            "jacobian_condition": float(np.linalg.cond(jacobian)),
            "repeat_reference_drift_rgb": (reference_repeat - reference).tolist(),
            "ambient_start_rgb": baseline.tolist(),
            "ambient_end_rgb": final_baseline.tolist(),
            "validation": validation,
            "limitations": [
                "Global correction of combined table/projector/camera color cast.",
                "Not isolated surface reflectance or colorimetric calibration.",
                "No spatial compensation, ambient subtraction or optical feedback cancellation.",
                "Fixed exposure, white balance, geometry and ambient light required.",
                "Gains and headroom only attenuate; projector black remains black.",
                "Saturated highlights excluded; full white maps to measured safe headroom.",
            ],
        }
    accept_model(model, session.records, session.metadata.get("restored"))
    hg._write_json_atomic(os.path.join(session.directory, "model.json"), model)
    print("MODEL " + json.dumps(model), flush=True)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--probe", metavar="NEW_DIRECTORY")
    mode.add_argument("--calibrate", metavar="NEW_DIRECTORY")
    arguments = parser.parse_args()
    if arguments.probe:
        probe(arguments.probe)
    else:
        calibrate(arguments.calibrate)
