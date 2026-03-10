"""Performance benchmark: image-to-body-math (nanobind) vs pure NumPy."""

import time
import math
import numpy as np
import image_to_body_math as p2b


def _banner(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _bench(label: str, fn, *, repeats: int = 5) -> float:
    """Run fn() `repeats` times, return best time in seconds."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    best = min(times)
    return best


# ---- Pure NumPy reference implementations ----

def numpy_tangents_to_ned(w_tan: np.ndarray, h_tan: np.ndarray) -> np.ndarray:
    cos_az = 1.0 / np.sqrt(1.0 + w_tan * w_tan)
    sin_az = w_tan * cos_az
    cos_el = 1.0 / np.sqrt(1.0 + h_tan * h_tan)
    sin_el = h_tan * cos_el
    return np.column_stack([cos_el * cos_az, cos_el * sin_az, -sin_el])


def numpy_quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v (N,3) by quaternion q [w,x,y,z]."""
    w, x, y, z = q
    t = 2.0 * np.cross(np.array([x, y, z]), v)
    return v + w * t + np.cross(np.array([x, y, z]), t)


def numpy_pixel_to_ned(rows, cols, width, height, p2t, cam_q, att_q):
    hw = width / 2.0
    hh = height / 2.0
    w_tan = (rows.astype(np.float64) - hw) * p2t
    h_tan = (cols.astype(np.float64) - hh) * p2t

    # warp_image_to_body (h_tan negated)
    neg_h = -h_tan
    cos_az = 1.0 / np.sqrt(1.0 + w_tan * w_tan)
    sin_az = w_tan * cos_az
    cos_el = 1.0 / np.sqrt(1.0 + neg_h * neg_h)
    sin_el = neg_h * cos_el
    dir_cam = np.column_stack([cos_el * cos_az, cos_el * sin_az, -sin_el])

    dir_body = numpy_quat_rotate(cam_q, dir_cam)
    dir_ned = numpy_quat_rotate(att_q, dir_body)
    return dir_ned


def main():
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    p2t = p2b.pixel_to_tan_from_fov(640, 480, math.radians(90))

    for n in [100, 1_000, 10_000, 100_000]:
        _banner(f"N = {n:,} pixels")

        rows = np.random.randint(0, 640, size=n).astype(np.uint64)
        cols = np.random.randint(0, 480, size=n).astype(np.uint64)

        # ---- p2b batch (nanobind) ----
        t_p2b = _bench("p2b batch", lambda: p2b.pixel_to_ned_batch(
            rows, cols, 640, 480, p2t, identity, identity))

        # ---- pure numpy ----
        t_np = _bench("numpy", lambda: numpy_pixel_to_ned(
            rows, cols, 640, 480, p2t, identity, identity))

        # ---- p2b scalar loop ----
        if n <= 10_000:
            t_scalar = _bench("p2b scalar", lambda: np.array([
                p2b.pixel_to_ned(int(r), int(c), 640, 480, p2t, identity, identity)
                for r, c in zip(rows, cols)
            ]), repeats=1)
        else:
            t_scalar = float('nan')

        speedup = t_np / t_p2b if t_p2b > 0 else float('inf')
        print(f"  p2b batch:  {t_p2b*1e6:10.0f} us")
        print(f"  pure numpy: {t_np*1e6:10.0f} us")
        if not math.isnan(t_scalar):
            print(f"  p2b scalar: {t_scalar*1e6:10.0f} us")
        print(f"  speedup vs numpy: {speedup:.1f}x")

    # ---- Single-call latency ----
    _banner("Single-call latency (scalar)")
    row, col = 320, 240

    fns = {
        "pixel_to_ned":           lambda: p2b.pixel_to_ned(row, col, 640, 480, p2t, identity, identity),
        "ned_to_pixel":           lambda: p2b.ned_to_pixel(np.array([1.0, 0.0, 0.0]), 640, 480, p2t, identity, identity),
        "pixel_after_rotation":   lambda: p2b.pixel_after_rotation(row, col, 640, 480, p2t, identity, identity, identity),
        "tangents_to_ned":        lambda: p2b.tangents_to_ned(0.1, 0.2),
        "cam_to_body_from_angle": lambda: p2b.cam_to_body_from_angle(0.5),
        "pixel_to_tan_from_fov":  lambda: p2b.pixel_to_tan_from_fov(640, 480, 1.57),
        "is_pixel_inside_frame":  lambda: p2b.is_pixel_inside_frame(row, col, 640, 480, 0.1),
    }

    for name, fn in fns.items():
        t = _bench(name, fn, repeats=10000)
        print(f"  {name:30s}  {t*1e9:.0f} ns")


if __name__ == "__main__":
    main()
