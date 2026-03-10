"""Image-to-body coordinate transformations with zero-copy NumPy/SciPy interop.

All vectors are numpy arrays. Quaternions use [w, x, y, z] convention
and accept scipy.spatial.transform.Rotation directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _core

if TYPE_CHECKING:
    from numpy.typing import NDArray

__version__ = "0.4.0"

# Re-export ImageSize
ImageSize = _core.ImageSize


# ---- Quaternion / Vector helpers ----

def _to_wxyz(q) -> np.ndarray:
    """Convert quaternion-like to contiguous [w, x, y, z] float64 array.

    Accepts numpy array (4,), scipy Rotation, or sequence.
    """
    try:
        from scipy.spatial.transform import Rotation
        if isinstance(q, Rotation):
            xyzw = q.as_quat()
            return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
    except ImportError:
        pass

    arr = np.asarray(q, dtype=np.float64)
    if arr.shape != (4,):
        raise ValueError(f"Quaternion must have shape (4,), got {arr.shape}")
    return np.ascontiguousarray(arr)


def _to_vec3(v) -> np.ndarray:
    """Convert vector-like to contiguous (3,) float64 array."""
    arr = np.asarray(v, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"Vector must have shape (3,), got {arr.shape}")
    return np.ascontiguousarray(arr)


# ============================================================
#  1D pixel-tangent conversions
# ============================================================

def pixel_tan_from_fov(pixel: int, width: int, height: int, fov_rad: float) -> float:
    """Pixel index -> angular tangent via camera FOV (radians)."""
    return _core.pixel_tan_from_fov(pixel, width, height, fov_rad)


def tan_to_pixel_by_fov(pixel_tan: float, width: int, height: int, fov_rad: float) -> int:
    """Tangent -> pixel index via FOV."""
    return _core.tan_to_pixel_by_fov(pixel_tan, width, height, fov_rad)


def pixel_tan_by_pixel_to_tan(pixel: int, width: int, height: int, pixel_to_tan: float) -> float:
    """Pixel index -> tangent via pixel-to-tan factor."""
    return _core.pixel_tan_by_pixel_to_tan(pixel, width, height, pixel_to_tan)


def angle_tan_to_pixel(angle_tan: float, width: int, height: int, pixel_to_tan: float) -> int:
    """Angular tangent -> pixel index."""
    return _core.angle_tan_to_pixel(angle_tan, width, height, pixel_to_tan)


def pixel_tan_by_pixel_to_tan_clipped(
    pixel: int, width: int, height: int, pixel_to_tan: float, threshold: float
) -> float:
    """Pixel -> tangent with dead-zone clipping around center."""
    return _core.pixel_tan_by_pixel_to_tan_clipped(pixel, width, height, pixel_to_tan, threshold)


def tan_to_pixel_by_pixel_to_tan(
    pixel_tan: float, width: int, height: int, pixel_to_tan: float, round_back: bool = False
) -> int:
    """Tangent -> pixel index via pixel-to-tan factor."""
    return _core.tan_to_pixel_by_pixel_to_tan(pixel_tan, width, height, pixel_to_tan, round_back)


def pixel_to_tan_from_fov(width: int, height: int, fov_rad: float) -> float:
    """Compute pixel-to-tan conversion factor from FOV."""
    return _core.pixel_to_tan_from_fov(width, height, fov_rad)


# ============================================================
#  Body space
# ============================================================

def cam_to_body_from_angle(angle_rad: float) -> NDArray[np.float64]:
    """Camera-to-body quaternion [w,x,y,z] from installation angle."""
    return np.asarray(_core.cam_to_body_from_angle(angle_rad))


def tangents_to_ned(w_tan: float, h_tan: float) -> NDArray[np.float64]:
    """Azimuth/elevation tangent pair -> NED direction vector."""
    return np.asarray(_core.tangents_to_ned(w_tan, h_tan))


def ned_to_tangents(ned) -> tuple[float, float]:
    """NED direction -> tangent pair (w_tan, h_tan)."""
    return _core.ned_to_tangents(_to_vec3(ned))


def ned_to_azimuth_elevation(ned) -> tuple[float, float]:
    """NED direction -> (azimuth, elevation) in radians."""
    return _core.ned_to_azimuth_elevation(_to_vec3(ned))


def azimuth_elevation_to_ned(azimuth: float, elevation: float) -> NDArray[np.float64]:
    """Azimuth/elevation (radians) -> NED direction vector."""
    return np.asarray(_core.azimuth_elevation_to_ned(azimuth, elevation))


def warp_image_to_body(w_tan: float, h_tan: float, cam_to_body) -> NDArray[np.float64]:
    """Image tangents -> body-frame direction using camera quaternion."""
    return np.asarray(_core.warp_image_to_body(w_tan, h_tan, _to_wxyz(cam_to_body)))


def warp_body_to_image(dir_body, cam_to_body) -> tuple[float, float]:
    """Body-frame direction -> image tangent pair."""
    return _core.warp_body_to_image(_to_vec3(dir_body), _to_wxyz(cam_to_body))


def pixel_to_ned(
    row: int, col: int, width: int, height: int,
    pixel_to_tan: float, cam_to_body, attitude,
) -> NDArray[np.float64]:
    """Pixel -> NED direction vector.

    cam_to_body, attitude: quaternion [w,x,y,z] or scipy Rotation.
    """
    return np.asarray(_core.pixel_to_ned(
        row, col, width, height, pixel_to_tan,
        _to_wxyz(cam_to_body), _to_wxyz(attitude)))


def ned_to_pixel(
    dir_ned, width: int, height: int,
    pixel_to_tan: float, cam_to_body, attitude,
) -> tuple[int, int]:
    """NED direction -> pixel coordinates (row, col)."""
    return _core.ned_to_pixel(
        _to_vec3(dir_ned), width, height, pixel_to_tan,
        _to_wxyz(cam_to_body), _to_wxyz(attitude))


def pixel_after_rotation(
    row: int, col: int, width: int, height: int,
    pixel_to_tan: float, cam_to_body, q_old, q_new,
    round_back: bool = False,
) -> tuple[int, int]:
    """Pixel position after body rotation change."""
    return _core.pixel_after_rotation(
        row, col, width, height, pixel_to_tan,
        _to_wxyz(cam_to_body), _to_wxyz(q_old), _to_wxyz(q_new), round_back)


def is_pixel_inside_frame(row: int, col: int, width: int, height: int, boundary: float) -> bool:
    """Check if pixel is inside frame with safety margin."""
    return _core.is_pixel_inside_frame(row, col, width, height, boundary)


def is_ned_inside_frame(
    dir_ned, width: int, height: int,
    pixel_to_tan: float, cam_to_body, attitude, boundary: float,
) -> bool:
    """Check if NED direction projects inside frame."""
    return _core.is_ned_inside_frame(
        _to_vec3(dir_ned), width, height, pixel_to_tan,
        _to_wxyz(cam_to_body), _to_wxyz(attitude), boundary)


def pixel_at_elevation(
    row: int, col: int, width: int, height: int,
    pixel_to_tan: float, cam_to_body, attitude,
    desired_elevation: float,
) -> tuple[int, int]:
    """Project pixel to target elevation, preserving azimuth."""
    return _core.pixel_at_elevation(
        row, col, width, height, pixel_to_tan,
        _to_wxyz(cam_to_body), _to_wxyz(attitude), desired_elevation)


def ned_angle_in_pixels(ned1, ned2, pixel_to_tan: float) -> float:
    """Angular separation between NED vectors as pixel distance."""
    return _core.ned_angle_in_pixels(_to_vec3(ned1), _to_vec3(ned2), pixel_to_tan)


# ============================================================
#  Batch (vectorized) — zero-copy numpy arrays
# ============================================================

def pixel_to_ned_batch(
    rows: NDArray[np.uint64], cols: NDArray[np.uint64],
    width: int, height: int, pixel_to_tan: float,
    cam_to_body, attitude,
) -> NDArray[np.float64]:
    """Batch pixels -> NED directions. Returns (N, 3) float64 array.

    Zero-copy: input arrays are accessed directly without copying.
    """
    return np.asarray(_core.pixel_to_ned_batch(
        np.ascontiguousarray(rows, dtype=np.uint64),
        np.ascontiguousarray(cols, dtype=np.uint64),
        width, height, pixel_to_tan,
        _to_wxyz(cam_to_body), _to_wxyz(attitude)))


def ned_to_pixel_batch(
    dirs_ned: NDArray[np.float64],
    width: int, height: int, pixel_to_tan: float,
    cam_to_body, attitude,
) -> NDArray[np.uint64]:
    """Batch NED directions -> pixels. Returns (N, 2) uint64 array.

    Zero-copy: input (N, 3) array is reinterpreted as Vector3* directly.
    """
    return np.asarray(_core.ned_to_pixel_batch(
        np.ascontiguousarray(dirs_ned, dtype=np.float64),
        width, height, pixel_to_tan,
        _to_wxyz(cam_to_body), _to_wxyz(attitude)))


def pixel_after_rotation_batch(
    rows: NDArray[np.uint64], cols: NDArray[np.uint64],
    width: int, height: int, pixel_to_tan: float,
    cam_to_body, q_old, q_new,
    round_back: bool = False,
) -> NDArray[np.uint64]:
    """Batch pixel positions after rotation. Returns (N, 2) uint64 array."""
    return np.asarray(_core.pixel_after_rotation_batch(
        np.ascontiguousarray(rows, dtype=np.uint64),
        np.ascontiguousarray(cols, dtype=np.uint64),
        width, height, pixel_to_tan,
        _to_wxyz(cam_to_body), _to_wxyz(q_old), _to_wxyz(q_new), round_back))


def warp_image_to_body_batch(
    w_tans: NDArray[np.float64], h_tans: NDArray[np.float64],
    cam_to_body,
) -> NDArray[np.float64]:
    """Batch image tangent pairs -> body-frame directions. Returns (N, 3) array."""
    return np.asarray(_core.warp_image_to_body_batch(
        np.ascontiguousarray(w_tans, dtype=np.float64),
        np.ascontiguousarray(h_tans, dtype=np.float64),
        _to_wxyz(cam_to_body)))


__all__ = [
    "__version__",
    "ImageSize",
    "pixel_tan_from_fov",
    "tan_to_pixel_by_fov",
    "pixel_tan_by_pixel_to_tan",
    "angle_tan_to_pixel",
    "pixel_tan_by_pixel_to_tan_clipped",
    "tan_to_pixel_by_pixel_to_tan",
    "pixel_to_tan_from_fov",
    "cam_to_body_from_angle",
    "tangents_to_ned",
    "ned_to_tangents",
    "ned_to_azimuth_elevation",
    "azimuth_elevation_to_ned",
    "warp_image_to_body",
    "warp_body_to_image",
    "pixel_to_ned",
    "ned_to_pixel",
    "pixel_after_rotation",
    "is_pixel_inside_frame",
    "is_ned_inside_frame",
    "pixel_at_elevation",
    "ned_angle_in_pixels",
    "pixel_to_ned_batch",
    "ned_to_pixel_batch",
    "pixel_after_rotation_batch",
    "warp_image_to_body_batch",
]
