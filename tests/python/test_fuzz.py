"""Property-based fuzz testing with hypothesis."""

import math

import numpy as np
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

import image_to_body_math as p2b

# ---- Strategies ----

reasonable_image_dim = st.integers(min_value=10, max_value=4096)
reasonable_fov = st.floats(min_value=0.1, max_value=math.pi - 0.1)
reasonable_angle = st.floats(min_value=-math.pi, max_value=math.pi)
small_float = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
FUZZ = settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow], deadline=None)


@st.composite
def pixel_and_image(draw):
    w = draw(reasonable_image_dim)
    h = draw(reasonable_image_dim)
    row = draw(st.integers(min_value=0, max_value=w))
    col = draw(st.integers(min_value=0, max_value=h))
    return row, col, w, h


@st.composite
def unit_quaternion(draw):
    w = draw(st.floats(min_value=-1, max_value=1))
    x = draw(st.floats(min_value=-1, max_value=1))
    y = draw(st.floats(min_value=-1, max_value=1))
    z = draw(st.floats(min_value=-1, max_value=1))
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    assume(norm > 0.01)
    return np.array([w / norm, x / norm, y / norm, z / norm])


# ============================================================
#  1D round-trip properties
# ============================================================


class TestPixelTanRoundTrip:
    @given(data=pixel_and_image(), fov=reasonable_fov)
    @FUZZ
    def test_by_fov(self, data, fov):
        row, _, w, h = data
        assume(row < w)
        tan_val = p2b.pixel_tan_from_fov(row, w, h, fov)
        recovered = p2b.tan_to_pixel_by_fov(tan_val, w, h, fov)
        assert abs(recovered - row) <= 1

    @given(data=pixel_and_image(), fov=reasonable_fov)
    @FUZZ
    def test_by_p2t(self, data, fov):
        row, _, w, h = data
        assume(row < w)
        p2t = p2b.pixel_to_tan_from_fov(w, h, fov)
        tan_val = p2b.pixel_tan_by_pixel_to_tan(row, w, h, p2t)
        recovered = p2b.tan_to_pixel_by_pixel_to_tan(tan_val, w, h, p2t, round_back=True)
        assert abs(recovered - row) <= 1


# ============================================================
#  Tangent <-> NED round-trips
# ============================================================


class TestTangentNedRoundTrip:
    @given(w_tan=small_float, h_tan=small_float)
    @FUZZ
    def test_round_trip(self, w_tan, h_tan):
        ned = p2b.tangents_to_ned(w_tan, h_tan)
        w_out, h_out = p2b.ned_to_tangents(ned)
        assert abs(w_tan - w_out) < 1e-6
        assert abs(h_tan - h_out) < 1e-6

    @given(w_tan=small_float, h_tan=small_float)
    @FUZZ
    def test_unit_output(self, w_tan, h_tan):
        ned = p2b.tangents_to_ned(w_tan, h_tan)
        assert abs(np.linalg.norm(ned) - 1.0) < 1e-10


# ============================================================
#  Azimuth / elevation
# ============================================================


class TestAzElRoundTrip:
    @given(
        az=reasonable_angle,
        el=st.floats(min_value=-math.pi / 2 + 0.01, max_value=math.pi / 2 - 0.01),
    )
    @FUZZ
    def test_round_trip(self, az, el):
        ned = p2b.azimuth_elevation_to_ned(az, el)
        az_out, el_out = p2b.ned_to_azimuth_elevation(ned)
        assert abs(math.sin(az) - math.sin(az_out)) < 1e-6
        assert abs(math.cos(az) - math.cos(az_out)) < 1e-6
        assert abs(el - el_out) < 1e-6


# ============================================================
#  Warp image <-> body
# ============================================================


class TestWarpRoundTrip:
    @given(w_tan=small_float, h_tan=small_float, angle=reasonable_angle)
    @FUZZ
    def test_round_trip(self, w_tan, h_tan, angle):
        q = p2b.cam_to_body_from_angle(angle)
        body = p2b.warp_image_to_body(w_tan, h_tan, q)
        w_out, h_out = p2b.warp_body_to_image(body, q)
        assert abs(w_tan - w_out) < 1e-4
        assert abs(h_tan - h_out) < 1e-4


# ============================================================
#  Pixel <-> NED (full pipeline)
# ============================================================


class TestPixelNedRoundTrip:
    @given(data=pixel_and_image(), fov=reasonable_fov)
    @FUZZ
    def test_round_trip(self, data, fov):
        row, col, w, h = data
        assume(row > 1 and col > 1 and row < w - 1 and col < h - 1)
        p2t = p2b.pixel_to_tan_from_fov(w, h, fov)
        ned = p2b.pixel_to_ned(row, col, w, h, p2t, IDENTITY, IDENTITY)
        r2, c2 = p2b.ned_to_pixel(ned, w, h, p2t, IDENTITY, IDENTITY)
        assert abs(r2 - row) <= 1
        assert abs(c2 - col) <= 1


# ============================================================
#  Rotation invariance
# ============================================================


class TestRotationInvariance:
    @given(data=pixel_and_image(), fov=reasonable_fov, q=unit_quaternion())
    @FUZZ
    def test_same_rotation_preserves(self, data, fov, q):
        row, col, w, h = data
        assume(row > 5 and col > 5 and row < w - 5 and col < h - 5)
        p2t = p2b.pixel_to_tan_from_fov(w, h, fov)
        r2, c2 = p2b.pixel_after_rotation(row, col, w, h, p2t, IDENTITY, q, q)
        assert abs(r2 - row) <= 2
        assert abs(c2 - col) <= 2


# ============================================================
#  cam_to_body quaternion properties
# ============================================================


class TestCamToBodyProperties:
    @given(angle=reasonable_angle)
    @FUZZ
    def test_unit_quaternion(self, angle):
        q = p2b.cam_to_body_from_angle(angle)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10


# ============================================================
#  Boundary checks
# ============================================================


class TestBoundaryFuzz:
    @given(data=pixel_and_image())
    @FUZZ
    def test_center_always_inside(self, data):
        _, _, w, h = data
        assume(w > 10 and h > 10)
        assert p2b.is_pixel_inside_frame(w // 2, h // 2, w, h, 0.0)

    @given(data=pixel_and_image(), boundary=st.floats(min_value=0.0, max_value=0.49))
    @FUZZ
    def test_center_inside_with_margin(self, data, boundary):
        _, _, w, h = data
        assume(w > 10 and h > 10)
        assert p2b.is_pixel_inside_frame(w // 2, h // 2, w, h, boundary)


# ============================================================
#  Clip threshold
# ============================================================


class TestClipFuzz:
    @given(data=pixel_and_image(), fov=reasonable_fov, threshold=st.floats(min_value=0.01, max_value=1.0))
    @FUZZ
    def test_center_always_clipped(self, data, fov, threshold):
        _, _, w, h = data
        # Use even widths so w//2 is exactly at half_width
        assume(w > 2 and w % 2 == 0)
        p2t = p2b.pixel_to_tan_from_fov(w, h, fov)
        result = p2b.pixel_tan_by_pixel_to_tan_clipped(w // 2, w, h, p2t, threshold)
        assert result == 0.0


# ============================================================
#  Batch consistency
# ============================================================


class TestBatchConsistency:
    @given(
        rows=st.lists(st.integers(min_value=10, max_value=620), min_size=1, max_size=20),
        cols=st.lists(st.integers(min_value=10, max_value=460), min_size=1, max_size=20),
        fov=reasonable_fov,
    )
    @FUZZ
    def test_pixel_to_ned_batch_matches_scalar(self, rows, cols, fov):
        n = min(len(rows), len(cols))
        rows = rows[:n]
        cols = cols[:n]
        w, h = 640, 480
        p2t = p2b.pixel_to_tan_from_fov(w, h, fov)

        scalar = np.array([
            p2b.pixel_to_ned(r, c, w, h, p2t, IDENTITY, IDENTITY)
            for r, c in zip(rows, cols)
        ])
        batch = p2b.pixel_to_ned_batch(
            np.array(rows, dtype=np.uint64),
            np.array(cols, dtype=np.uint64),
            w, h, p2t, IDENTITY, IDENTITY)

        np.testing.assert_allclose(batch, scalar, atol=1e-10)
