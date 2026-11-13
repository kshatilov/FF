"""Lossless, bounded ndarray messages for the local hg/TouchDesigner bridge."""

import base64
import math
import zlib

import numpy as np


MAX_ARRAY_BYTES = 128 * 1024 * 1024
DTYPES = {"uint8": np.dtype("uint8"), "float32": np.dtype("<f4")}


def encode_array(array):
    array = np.asarray(array)
    if array.dtype == np.uint8:
        dtype = "uint8"
    elif array.dtype.kind == "f" and array.dtype.itemsize == 4:
        dtype = "float32"
    else:
        raise ValueError("Wire arrays must be uint8 or float32.")
    shape = list(array.shape)
    _array_size(shape, dtype)
    raw = np.ascontiguousarray(array, dtype=DTYPES[dtype]).tobytes()
    return {
        "encoding": "zlib-base64",
        "dtype": dtype,
        "shape": shape,
        "data": base64.b64encode(zlib.compress(raw)).decode("ascii"),
    }


def _array_size(shape, dtype):
    if (not isinstance(shape, (list, tuple)) or len(shape) != 3
            or any(type(v) is not int or v <= 0 for v in shape)
            or shape[0] > 8192 or shape[1] > 8192 or shape[2] > 4):
        raise ValueError("Wire shape must be [height, width, channels], channels 1..4.")
    if dtype not in DTYPES:
        raise ValueError("Unsupported wire dtype.")
    size = math.prod(shape) * DTYPES[dtype].itemsize
    if size > MAX_ARRAY_BYTES:
        raise ValueError("Wire array exceeds the byte limit.")
    return size


def decode_array(message):
    if not isinstance(message, dict) or message.get("encoding") != "zlib-base64":
        raise ValueError("Expected a zlib-base64 array message.")
    shape, dtype = message.get("shape"), message.get("dtype")
    expected = _array_size(shape, dtype)
    data = message.get("data")
    if not isinstance(data, str) or len(data) > 2 * MAX_ARRAY_BYTES:
        raise ValueError("Invalid or oversized wire payload.")
    try:
        packed = base64.b64decode(data, validate=True)
        stream = zlib.decompressobj()
        raw = stream.decompress(packed, expected + 1)
    except (ValueError, zlib.error) as error:
        raise ValueError("Malformed compressed array payload.") from error
    if (len(raw) != expected or not stream.eof or stream.unused_data
            or stream.unconsumed_tail):
        raise ValueError("Compressed array size does not match its declared shape.")
    return np.frombuffer(raw, dtype=DTYPES[dtype]).reshape(shape).copy()


def td_color_to_bgr(rgba):
    """TD bottom-up float RGB(A) -> top-down uint8 BGR camera raster."""
    rgba = np.asarray(rgba)
    if rgba.ndim != 3 or rgba.shape[2] < 3:
        raise ValueError("TD color array must have at least three channels.")
    rgb = np.nan_to_num(rgba[::-1, :, :3], nan=0, posinf=1, neginf=0)
    return np.ascontiguousarray(
        np.rint(np.clip(rgb, 0, 1)[:, :, ::-1] * 255).astype(np.uint8))


def bgr_to_td_rgba(bgr):
    """Top-down uint8 BGR pattern -> TD bottom-up float RGBA."""
    bgr = np.asarray(bgr)
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Pattern must be a uint8 BGR image.")
    rgba = np.ones((*bgr.shape[:2], 4), dtype=np.float32)
    rgba[:, :, :3] = bgr[::-1, :, ::-1].astype(np.float32) / 255
    return rgba
