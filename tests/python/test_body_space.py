"""End-to-end tests for body space transformations."""

import math

import numpy as np
import pytest

import image_to_body_math as p2b

EPSILON = 1e-4
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


class TestCamToBodyFromAngle:
    def test_zero_is_identity(self):
        q = p2b.cam_to_body_from_angle(0.0)
        np.testing.assert_allclose(q, IDENTITY, atol=EPSILON)

    def test_90deg_tilt(self):
        q = p2b.cam_to_body_from_angle(math.radians(90))
        assert abs(q[0] - math.cos(math.pi / 4)) < EPSILON
        assert abs(q[2] - (-math.sin(math.pi / 4))) < EPSILON


class TestTangentsToNed:
    def test_forward(self):
        ned = p2b.tangents_to_ned(0.0, 0.0)
        np.testing.assert_allclose(ned, [1.0, 0.0, 0.0], atol=EPSILON)

    def test_unit_vector(self):
        ned = p2b.tangents_to_ned(0.5, 0.3)
        assert abs(np.linalg.norm(ned) - 1.0) < EPSILON


class TestNedToTangents:
    def test_forward(self):
        w, h = p2b.ned_to_tangents(np.array([1.0, 0.0, 0.0]))
        assert abs(w) < EPSILON
        assert abs(h) < EPSILON

    def test_round_trip(self):
        for w_in, h_in in [(0.3, 0.2), (-0.1, 0.5), (0.0, 0.0), (1.0, -0.5)]:
            ned = p2b.tangents_to_ned(w_in, h_in)
            w_out, h_out = p2b.ned_to_tangents(ned)
            assert abs(w_in - w_out) < EPSILON
            assert abs(h_in - h_out) < EPSILON


class TestAzimuthElevation:
    def test_round_trip(self):
        for az, el in [(0.3, 0.1), (-0.5, 0.2), (0.0, 0.0)]:
            ned = p2b.azimuth_elevation_to_ned(az, el)
            az_out, el_out = p2b.ned_to_azimuth_elevation(ned)
            assert abs(az - az_out) < EPSILON
            assert abs(el - el_out) < EPSILON

    def test_forward_is_zero(self):
        az, el = p2b.ned_to_azimuth_elevation(np.array([1.0, 0.0, 0.0]))
        assert abs(az) < EPSILON
        assert abs(el) < EPSILON


class TestWarpImageBody:
    def test_identity_cam(self):
        v = p2b.warp_image_to_body(0.0, 0.0, IDENTITY)
        np.testing.assert_allclose(v, [1.0, 0.0, 0.0], atol=EPSILON)

    def test_round_trip(self):
        q = p2b.cam_to_body_from_angle(math.radians(30))
        for w_in, h_in in [(0.2, 0.1), (0.0, 0.0), (-0.3, 0.4)]:
            body = p2b.warp_image_to_body(w_in, h_in, q)
            w_out, h_out = p2b.warp_body_to_image(body, q)
            assert abs(w_in - w_out) < EPSILON
            assert abs(h_in - h_out) < EPSILON


class TestPixelToNed:
    def test_center_is_forward(self):
        p2t = p2b.pixel_to_tan_from_fov(640, 480, math.radians(90))
        ned = p2b.pixel_to_ned(320, 240, 640, 480, p2t, IDENTITY, IDENTITY)
        np.testing.assert_allclose(ned, [1.0, 0.0, 0.0], atol=EPSILON)

    def test_round_trip(self):
        p2t = p2b.pixel_to_tan_from_fov(640, 480, math.radians(90))
        for row, col in [(200, 100), (320, 240), (500, 400)]:
            ned = p2b.pixel_to_ned(row, col, 640, 480, p2t, IDENTITY, IDENTITY)
            r2, c2 = p2b.ned_to_pixel(ned, 640, 480, p2t, IDENTITY, IDENTITY)
            assert abs(r2 - row) <= 1
            assert abs(c2 - col) <= 1


class TestPixelAfterRotation:
    def test_identity_rotation(self):
        p2t = p2b.pixel_to_tan_from_fov(640, 480, math.radians(90))
        row, col = p2b.pixel_after_rotation(200, 150, 640, 480, p2t, IDENTITY, IDENTITY, IDENTITY)
        assert abs(row - 200) <= 1
        assert abs(col - 150) <= 1

    def test_same_rotation_preserves(self):
        p2t = p2b.pixel_to_tan_from_fov(640, 480, math.radians(90))
        q = p2b.cam_to_body_from_angle(math.radians(15))
        row, col = p2b.pixel_after_rotation(300, 200, 640, 480, p2t, IDENTITY, q, q)
        assert abs(row - 300) <= 1
        assert abs(col - 200) <= 1


class TestIsPixelInsideFrame:
    def test_center_inside(self):
        assert p2b.is_pixel_inside_frame(320, 240, 640, 480, 0.0)

    def test_corner_inside_no_margin(self):
        assert p2b.is_pixel_inside_frame(0, 0, 640, 480, 0.0)

    def test_corner_outside_with_margin(self):
        assert not p2b.is_pixel_inside_frame(0, 0, 640, 480, 0.1)


class TestIsNedInsideFrame:
    def test_forward_inside(self):
        p2t = p2b.pixel_to_tan_from_fov(640, 480, math.radians(90))
        assert p2b.is_ned_inside_frame(
            np.array([1.0, 0.0, 0.0]), 640, 480, p2t, IDENTITY, IDENTITY, 0.0)

    def test_backward_outside(self):
        p2t = p2b.pixel_to_tan_from_fov(640, 480, math.radians(90))
        assert not p2b.is_ned_inside_frame(
            np.array([-1.0, 0.0, 0.0]), 640, 480, p2t, IDENTITY, IDENTITY, 0.0)


class TestPixelAtElevation:
    def test_center_at_zero(self):
        p2t = p2b.pixel_to_tan_from_fov(640, 480, math.radians(90))
        row, col = p2b.pixel_at_elevation(320, 240, 640, 480, p2t, IDENTITY, IDENTITY, 0.0)
        assert abs(row - 320) <= 1
        assert abs(col - 240) <= 1


class TestNedAngleInPixels:
    def test_same_direction_zero(self):
        p2t = p2b.pixel_to_tan_from_fov(640, 480, math.radians(90))
        result = p2b.ned_angle_in_pixels(
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            p2t)
        assert abs(result) < EPSILON


class TestScipyInterop:
    def test_rotation_identity(self):
        from scipy.spatial.transform import Rotation
        p2t = p2b.pixel_to_tan_from_fov(640, 480, math.radians(90))
        ned = p2b.pixel_to_ned(320, 240, 640, 480, p2t, IDENTITY, Rotation.identity())
        np.testing.assert_allclose(ned, [1.0, 0.0, 0.0], atol=EPSILON)

    def test_rotation_from_euler(self):
        from scipy.spatial.transform import Rotation
        p2t = p2b.pixel_to_tan_from_fov(640, 480, math.radians(90))
        rot = Rotation.from_euler('z', 45, degrees=True)
        ned = p2b.pixel_to_ned(320, 240, 640, 480, p2t, IDENTITY, rot)
        assert ned[0] > 0  # North component
        assert ned[1] > 0  # East component
        assert abs(ned[2]) < EPSILON

    def test_scipy_cam_to_body(self):
        from scipy.spatial.transform import Rotation
        p2t = p2b.pixel_to_tan_from_fov(640, 480, math.radians(90))
        cam = Rotation.from_euler('y', -30, degrees=True)
        ned = p2b.pixel_to_ned(320, 240, 640, 480, p2t, cam, Rotation.identity())
        assert np.linalg.norm(ned) > 0


class TestBatchOperations:
    def test_pixel_to_ned_batch(self):
        p2t = p2b.pixel_to_tan_from_fov(640, 480, math.radians(90))
        rows = np.array([320, 200, 400], dtype=np.uint64)
        cols = np.array([240, 100, 300], dtype=np.uint64)
        neds = p2b.pixel_to_ned_batch(rows, cols, 640, 480, p2t, IDENTITY, IDENTITY)
        assert neds.shape == (3, 3)
        np.testing.assert_allclose(neds[0], [1.0, 0.0, 0.0], atol=EPSILON)
        for i in range(3):
            assert abs(np.linalg.norm(neds[i]) - 1.0) < EPSILON

    def test_ned_to_pixel_batch(self):
        p2t = p2b.pixel_to_tan_from_fov(640, 480, math.radians(90))
        dirs = np.array([[1.0, 0.0, 0.0], [1.0, 0.1, 0.0]], dtype=np.float64)
        pixels = p2b.ned_to_pixel_batch(dirs, 640, 480, p2t, IDENTITY, IDENTITY)
        assert pixels.shape == (2, 2)
        assert abs(int(pixels[0, 0]) - 320) <= 1
        assert abs(int(pixels[0, 1]) - 240) <= 1

    def test_batch_round_trip(self):
        p2t = p2b.pixel_to_tan_from_fov(640, 480, math.radians(90))
        rows = np.array([100, 200, 300, 400, 500], dtype=np.uint64)
        cols = np.array([50, 100, 200, 300, 400], dtype=np.uint64)

        neds = p2b.pixel_to_ned_batch(rows, cols, 640, 480, p2t, IDENTITY, IDENTITY)
        pixels = p2b.ned_to_pixel_batch(neds, 640, 480, p2t, IDENTITY, IDENTITY)

        for i in range(5):
            assert abs(int(pixels[i, 0]) - int(rows[i])) <= 1
            assert abs(int(pixels[i, 1]) - int(cols[i])) <= 1

    def test_pixel_after_rotation_batch(self):
        p2t = p2b.pixel_to_tan_from_fov(640, 480, math.radians(90))
        rows = np.array([320, 200], dtype=np.uint64)
        cols = np.array([240, 150], dtype=np.uint64)
        result = p2b.pixel_after_rotation_batch(
            rows, cols, 640, 480, p2t, IDENTITY, IDENTITY, IDENTITY)
        assert result.shape == (2, 2)
        assert abs(int(result[0, 0]) - 320) <= 1

    def test_warp_batch(self):
        w = np.array([0.0, 0.1, 0.2], dtype=np.float64)
        h = np.array([0.0, 0.05, 0.1], dtype=np.float64)
        result = p2b.warp_image_to_body_batch(w, h, IDENTITY)
        assert result.shape == (3, 3)
        np.testing.assert_allclose(result[0], [1.0, 0.0, 0.0], atol=EPSILON)

    def test_output_is_contiguous_float64(self):
        p2t = p2b.pixel_to_tan_from_fov(640, 480, math.radians(90))
        rows = np.array([320, 200], dtype=np.uint64)
        cols = np.array([240, 150], dtype=np.uint64)
        neds = p2b.pixel_to_ned_batch(rows, cols, 640, 480, p2t, IDENTITY, IDENTITY)
        assert neds.dtype == np.float64
        assert neds.flags['C_CONTIGUOUS']

    def test_numpy_operations_on_output(self):
        """Output arrays support full numpy operations (zero-copy interop)."""
        p2t = p2b.pixel_to_tan_from_fov(640, 480, math.radians(90))
        ned = p2b.pixel_to_ned(320, 240, 640, 480, p2t, IDENTITY, IDENTITY)

        dot = ned @ np.array([0.0, 0.0, -1.0])
        cross = np.cross(ned, np.array([0.0, 1.0, 0.0]))
        norm = np.linalg.norm(ned)

        assert abs(dot) < EPSILON
        assert abs(norm - 1.0) < EPSILON
        assert cross.shape == (3,)

    def test_batch_matches_scalar(self):
        """Batch operations produce identical results to scalar calls."""
        p2t = p2b.pixel_to_tan_from_fov(640, 480, math.radians(90))
        test_rows = [100, 200, 320, 400, 500]
        test_cols = [50, 100, 240, 350, 420]

        scalar_neds = np.array([
            p2b.pixel_to_ned(r, c, 640, 480, p2t, IDENTITY, IDENTITY)
            for r, c in zip(test_rows, test_cols)
        ])
        batch_neds = p2b.pixel_to_ned_batch(
            np.array(test_rows, dtype=np.uint64),
            np.array(test_cols, dtype=np.uint64),
            640, 480, p2t, IDENTITY, IDENTITY)

        np.testing.assert_allclose(batch_neds, scalar_neds, atol=1e-10)
