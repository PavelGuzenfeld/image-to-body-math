"""End-to-end tests for 1D pixel-tangent conversions."""

import math

import pytest

import image_to_body_math as p2b

EPSILON = 1e-5


class TestPixelTanFromFov:
    def test_center_pixel(self):
        result = p2b.pixel_tan_from_fov(320, 640, 480, math.radians(90))
        assert abs(result) < EPSILON

    def test_right_edge(self):
        result = p2b.pixel_tan_from_fov(640, 640, 480, math.radians(90))
        assert abs(result - 1.0) < EPSILON

    def test_left_edge(self):
        result = p2b.pixel_tan_from_fov(0, 640, 480, math.radians(90))
        assert abs(result - (-1.0)) < EPSILON

    def test_narrow_fov(self):
        fov = math.radians(60)
        result = p2b.pixel_tan_from_fov(640, 640, 480, fov)
        assert abs(result - math.tan(fov / 2)) < EPSILON


class TestTanToPixelByFov:
    def test_zero_gives_center(self):
        result = p2b.tan_to_pixel_by_fov(0.0, 640, 480, math.radians(90))
        assert result == 320

    def test_round_trip(self):
        fov = math.radians(90)
        for pixel in [0, 100, 200, 320, 400, 500, 640]:
            tan_val = p2b.pixel_tan_from_fov(pixel, 640, 480, fov)
            recovered = p2b.tan_to_pixel_by_fov(tan_val, 640, 480, fov)
            assert recovered == pixel


class TestPixelTanByPixelToTan:
    def test_center_gives_zero(self):
        p2t = p2b.pixel_to_tan_from_fov(640, 480, math.radians(90))
        result = p2b.pixel_tan_by_pixel_to_tan(320, 640, 480, p2t)
        assert abs(result) < EPSILON

    def test_off_center(self):
        p2t = 0.001
        result = p2b.pixel_tan_by_pixel_to_tan(400, 640, 480, p2t)
        expected = (400 - 320) * p2t
        assert abs(result - expected) < EPSILON


class TestAngleTanToPixel:
    def test_zero_gives_center(self):
        p2t = p2b.pixel_to_tan_from_fov(640, 480, math.radians(90))
        result = p2b.angle_tan_to_pixel(0.0, 640, 480, p2t)
        assert result == 320


class TestPixelTanClipped:
    def test_within_threshold(self):
        p2t = 0.001
        result = p2b.pixel_tan_by_pixel_to_tan_clipped(325, 640, 480, p2t, 0.1)
        assert result == 0.0

    def test_outside_threshold(self):
        p2t = 0.001
        result = p2b.pixel_tan_by_pixel_to_tan_clipped(500, 640, 480, p2t, 0.01)
        expected = (500 - 320) * p2t
        assert abs(result - expected) < EPSILON


class TestTanToPixelByPixelToTan:
    def test_round_vs_truncate(self):
        p2t = p2b.pixel_to_tan_from_fov(640, 480, math.radians(90))
        tan_val = 0.123
        rounded = p2b.tan_to_pixel_by_pixel_to_tan(tan_val, 640, 480, p2t, round_back=True)
        truncated = p2b.tan_to_pixel_by_pixel_to_tan(tan_val, 640, 480, p2t, round_back=False)
        assert abs(rounded - truncated) <= 1


class TestPixelToTanFromFov:
    def test_90deg_fov(self):
        result = p2b.pixel_to_tan_from_fov(640, 480, math.radians(90))
        expected = math.tan(math.radians(45)) / 320
        assert abs(result - expected) < EPSILON

    def test_different_sizes(self):
        for w in [320, 640, 1280, 1920]:
            p2t = p2b.pixel_to_tan_from_fov(w, 480, math.radians(90))
            assert p2t > 0
