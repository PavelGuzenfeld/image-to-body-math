"""Shared fixtures for Python tests."""

import math

import numpy as np
import pytest


@pytest.fixture
def identity_quat():
    return np.array([1.0, 0.0, 0.0, 0.0])


@pytest.fixture
def standard_image():
    return 640, 480


@pytest.fixture
def standard_fov():
    return math.radians(90)


@pytest.fixture
def standard_p2t():
    import image_to_body_math as p2b
    return p2b.pixel_to_tan_from_fov(640, 480, math.radians(90))
