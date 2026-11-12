"""Geometry contract for hg: column vectors, top-left pixel centres, explicit direction."""

import cv2
import numpy as np


def points_array(points):
    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError("Points must be a finite Nx2 array.")
    return value


def image_size(size):
    value = np.asarray(size, dtype=np.float64)
    if (value.shape != (2,) or not np.isfinite(value).all()
            or np.any(value < 1) or np.any(value != np.floor(value))):
        raise ValueError("Image size must be positive integer [width, height].")
    return tuple(int(v) for v in value)


def transform_points(matrix, points):
    points = points_array(points)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Homography must be a finite 3x3 matrix.")
    homogeneous = np.column_stack((points, np.ones(len(points))))
    result = homogeneous @ matrix.T
    scale = result[:, 2]
    if np.any(np.abs(scale) < 1e-12):
        raise ValueError("Point maps to the projective horizon.")
    return result[:, :2] / scale[:, None]


def fit_homography(source, destination):
    """Fit source -> destination; residuals are fit diagnostics, not validation."""
    source, destination = points_array(source), points_array(destination)
    if source.shape != destination.shape or len(source) < 4:
        raise ValueError("At least four paired points are required.")
    for points in (source, destination):
        if len(np.unique(points, axis=0)) < 4:
            raise ValueError("At least four distinct points are required in each image.")
        if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
            raise ValueError("Collinear points cannot define a homography.")
    matrix, mask = cv2.findHomography(source, destination, cv2.RANSAC, 3.0)
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        raise ValueError("Homography estimation failed.")
    if np.linalg.matrix_rank(matrix) != 3:
        raise ValueError("Homography is singular.")
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < 4:
        raise ValueError("Fewer than four inliers.")
    # Frobenius normalization also works when H[2, 2] is zero.
    matrix = matrix / np.linalg.norm(matrix)
    inverse = np.linalg.inv(matrix)
    errors = np.linalg.norm(transform_points(matrix, source) - destination, axis=1)
    return matrix, inverse, {
        "point_count": len(source),
        "inlier_count": int(inliers.sum()),
        "inlier_mask": inliers.tolist(),
        "ransac_threshold_destination_px": 3.0,
        "inlier_mean_destination_px": float(errors[inliers].mean()),
        "inlier_max_destination_px": float(errors[inliers].max()),
        "independent_validation": False,
    }


def uv_to_pixel(size):
    """Bottom-left texture UV -> top-left pixel centres (half-pixel convention)."""
    width, height = image_size(size)
    return np.array([[width, 0, -0.5],
                     [0, -height, height - 0.5],
                     [0, 0, 1]], dtype=np.float64)


def uv_homography(matrix, source_size, destination_size):
    return (np.linalg.inv(uv_to_pixel(destination_size))
            @ np.asarray(matrix, dtype=np.float64) @ uv_to_pixel(source_size))


def resized_source_homography(matrix, original_size, resized_size):
    """Convert H(resized camera -> P) into H(original camera -> P).

    Assumes a full-frame pixel-centre resize, with no crop, flip or lens warp.
    """
    ow, oh = image_size(original_size)
    rw, rh = image_size(resized_size)
    sx, sy = rw / ow, rh / oh
    resize = np.array([[sx, 0, (sx - 1) / 2],
                       [0, sy, (sy - 1) / 2],
                       [0, 0, 1]], dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64) @ resize


def warp_parameters(document):
    """Validate a camera->projector artifact and derive inverse UV shader rows."""
    if (document.get("schema") != "ff.hg" or document.get("schema_version") != 1
            or document.get("kind") != "homography"):
        raise ValueError("Warp requires an ff.hg v1 homography, not a dense map.")
    source, destination = document["source"], document["destination"]
    source_id = source["id"]
    if (source_id not in ("0", "1") or source.get("type") != "camera"
            or destination.get("id") != "P0" or destination.get("type") != "projector"):
        raise ValueError("Warp requires camera 0/1 -> projector P0.")
    source_size = image_size(source["resolution"])
    destination_size = image_size(destination["resolution"])
    expected_top = "/project1/" + ("camLogi" if source_id == "0" else "kinectC")
    if source.get("top_path") != expected_top:
        raise ValueError("Calibration source must be the raw TD camera TOP.")
    if document.get("source_space") != "raw_distorted_pixels":
        raise ValueError("Unsupported calibration pixel space.")
    matrix = np.asarray(document["H_source_to_destination"], dtype=np.float64)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or np.linalg.matrix_rank(matrix) != 3):
        raise ValueError("Warp homography must be a finite, invertible 3x3 matrix.")
    inverse_uv = uv_homography(np.linalg.inv(matrix), destination_size, source_size)
    inverse_uv /= np.max(np.abs(inverse_uv))
    return {
        "source_id": source_id,
        "source_top": expected_top,
        "source_size": list(source_size),
        "destination_size": list(destination_size),
        "inverse_uv_rows": inverse_uv.tolist(),
        "calibration_created_utc": document.get("created_utc"),
    }


def td_pointcloud_depth_mm(rgba):
    """TD bottom-up depth-camera XYZ (metres) -> top-down uint16 axial depth."""
    rgba = np.asarray(rgba)
    if rgba.ndim != 3 or rgba.shape[:2] != (424, 512) or rgba.shape[2] < 3:
        raise ValueError("TD depth point cloud must have shape [424, 512, >=3].")
    xyz = rgba[::-1, :, :3]
    mm = xyz[:, :, 2].astype(np.float64) * 1000
    valid = np.isfinite(xyz).all(axis=2) & (mm > 0) & (mm <= 65535)
    depth = np.zeros((424, 512), dtype=np.uint16)
    depth[valid] = np.rint(mm[valid]).astype(np.uint16)
    return depth


def sanitize_sdk_map(xy, depth, source_camera, destination_size):
    """SDK forward XY lookup; reject invalid coordinates and missing depth."""
    destination_width, destination_height = image_size(destination_size)
    xy = np.asarray(xy, dtype=np.float32).copy()
    depth = np.asarray(depth)
    if xy.ndim != 3 or xy.shape[2] != 2:
        raise ValueError("SDK coordinates must have shape [height, width, 2].")
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("Depth must be a uint16 millimetre image.")
    valid = np.isfinite(xy).all(axis=2)
    valid &= ((xy[:, :, 0] >= 0) & (xy[:, :, 0] < destination_width)
              & (xy[:, :, 1] >= 0) & (xy[:, :, 1] < destination_height))
    if source_camera == 2:
        if xy.shape[:2] != depth.shape:
            raise ValueError("Depth-source map size does not match depth frame.")
        valid &= depth > 0
    elif source_camera == 1:
        if (destination_width, destination_height) != (depth.shape[1], depth.shape[0]):
            raise ValueError("Color-to-depth destination size does not match depth.")
        # Never cast infinity or clip it into an apparently valid edge pixel.
        safe = np.where(valid[:, :, None], xy, 0)
        indices = np.rint(safe).astype(np.int32)
        in_bounds = ((indices[:, :, 0] < destination_width)
                     & (indices[:, :, 1] < destination_height))
        valid &= in_bounds
        indices[~valid] = 0
        valid &= depth[indices[:, :, 1], indices[:, :, 0]] > 0
    else:
        raise ValueError("SDK mapping supports Kinect color (1) and depth (2) only.")
    xy[~valid] = -1
    return xy, valid
