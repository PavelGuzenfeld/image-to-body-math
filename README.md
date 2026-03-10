# image-to-body-math

Header-only C++23 library for pixel-to-body coordinate conversions, with **zero-copy Python bindings** (NumPy/SciPy).

Converts between image pixel coordinates, tangent-space, body-frame, and NED direction representations using camera parameters and attitude quaternions.

## Features

- **Pixel-to-tangent conversion** via FOV or pixel-to-tangent factor
- **Body-space pipeline** — full pixel-to-NED and NED-to-pixel conversion through camera and attitude quaternions
- **Pixel stabilization** — 6-stage pipeline compensating for body rotation changes
- **Camera installation** — camera-to-body quaternion from installation tilt angle
- **Azimuth/elevation** — conversions between NED vectors and azimuth/elevation angles or tangents
- **Boundary checking** — pixel-in-frame and NED-in-frame tests with absolute or fractional margins
- **Elevation projection** — project a pixel to a target elevation while preserving azimuth
- **NED angle in pixels** — angular separation between two NED vectors expressed as pixel distance
- **Dead-zone clipping** for center-pixel suppression
- **Type-safe angles** via [linalg3d](https://github.com/PavelGuzenfeld/linalg3d) (`Radians` vs `Degrees` at the type level)
- **Type-safe pixel math** via [strong-types](https://github.com/PavelGuzenfeld/strong-types) — `PixelTan`, `PixelToTan`, `NormalizedPixel`, `ClipThreshold` prevent argument mix-ups at compile time
- **constexpr where possible** — pure arithmetic operations evaluate at compile time
- **Python bindings** via [nanobind](https://github.com/wjakob/nanobind) — zero-copy NumPy arrays, scipy.spatial.transform.Rotation interop, vectorized batch operations

## Install

### Python (from PyPI)

```bash
pip install image-to-body-math
```

### Python (from source)

Requires a C++23 compiler (GCC 13+, Clang 17+).

```bash
pip install ".[test]"
```

### C++ (CMake)

Requires CMake 3.25+ and a C++23 compiler. Dependencies (linalg3d, strong-types, gcem, fmt, doctest) are fetched automatically via FetchContent.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

## Python API

All vectors are NumPy arrays. Quaternions use `[w, x, y, z]` order and accept `scipy.spatial.transform.Rotation` directly.

### Quick start

```python
import numpy as np
from scipy.spatial.transform import Rotation
import image_to_body_math as p2b

identity = np.array([1.0, 0.0, 0.0, 0.0])
p2t = p2b.pixel_to_tan_from_fov(640, 480, np.radians(90))

# Pixel → NED direction (returns numpy array)
ned = p2b.pixel_to_ned(320, 240, 640, 480, p2t, identity, identity)
# array([1., 0., 0.])

# scipy Rotation accepted directly
ned = p2b.pixel_to_ned(320, 240, 640, 480, p2t,
    identity, Rotation.from_euler('z', 45, degrees=True))

# NED → pixel
row, col = p2b.ned_to_pixel(ned, 640, 480, p2t, identity, identity)
```

### Batch operations (zero-copy)

```python
# 10,000 pixels → NED directions in one call
rows = np.arange(10000, dtype=np.uint64) % 640
cols = np.arange(10000, dtype=np.uint64) % 480
neds = p2b.pixel_to_ned_batch(rows, cols, 640, 480, p2t, identity, identity)
# neds.shape == (10000, 3), dtype=float64, C-contiguous

# NED directions → pixels (input array reinterpreted as Vector3* — zero copy)
pixels = p2b.ned_to_pixel_batch(neds, 640, 480, p2t, identity, identity)
# pixels.shape == (10000, 2), dtype=uint64

# Full numpy interop — dot products, cross products, norms
down = np.array([0.0, 0.0, -1.0])
elevations = neds @ down  # (10000,) array
```

### All functions

| Function | Description |
|----------|-------------|
| `pixel_tan_from_fov` | Pixel index → angular tangent via FOV |
| `tan_to_pixel_by_fov` | Tangent → pixel index via FOV |
| `pixel_tan_by_pixel_to_tan` | Pixel index → tangent via conversion factor |
| `angle_tan_to_pixel` | Angular tangent → pixel index |
| `pixel_tan_by_pixel_to_tan_clipped` | Pixel → tangent with dead-zone clipping |
| `tan_to_pixel_by_pixel_to_tan` | Tangent → pixel (round or truncate) |
| `pixel_to_tan_from_fov` | Compute conversion factor from FOV |
| `cam_to_body_from_angle` | Camera-to-body quaternion from tilt angle |
| `tangents_to_ned` / `ned_to_tangents` | Tangent pair ↔ NED direction |
| `ned_to_azimuth_elevation` / `azimuth_elevation_to_ned` | NED ↔ azimuth/elevation |
| `warp_image_to_body` / `warp_body_to_image` | Image tangents ↔ body-frame direction |
| `pixel_to_ned` / `ned_to_pixel` | Full pixel ↔ NED pipeline |
| `pixel_after_rotation` | Pixel position after body rotation change |
| `is_pixel_inside_frame` | Boundary check with safety margin |
| `is_ned_inside_frame` | NED visibility check |
| `pixel_at_elevation` | Project pixel to target elevation |
| `ned_angle_in_pixels` | Angular separation as pixel distance |
| `pixel_to_ned_batch` | Batch pixel → NED |
| `ned_to_pixel_batch` | Batch NED → pixel (zero-copy input) |
| `pixel_after_rotation_batch` | Batch rotation compensation |
| `warp_image_to_body_batch` | Batch image → body warp |

## C++ API

### Headers

| Header | Purpose |
|--------|---------|
| `types.hpp` | Strong type definitions, `ImageSize`, `PixelIndex` |
| `math.hpp` | 1D pixel-to-tangent conversions (FOV and pixel-to-tan factor) |
| `body_space.hpp` | 2D image-to-body-to-NED pipeline, rotation stabilization |

### Strong Types

All bare `double` parameters are replaced with tagged strong types:

| Type | Represents |
|------|-----------|
| `PixelTan` | Tangent of a pixel's angular offset from image center |
| `PixelToTan` | Conversion factor: tangent-per-pixel-offset |
| `NormalizedPixel` | Normalized pixel coordinate in [-1, 1] |
| `ClipThreshold` | Dead-zone threshold as fraction of half-width |

Type algebra enforces correct conversions:

```cpp
NormalizedPixel * PixelToTan -> PixelTan   // offset * factor = tangent
PixelTan / PixelToTan -> NormalizedPixel   // inverse
```

Mixing incompatible types is a compile error:

```cpp
PixelTan pt{0.1};
PixelToTan ptt{0.0025};
auto ok = pt / ptt;       // NormalizedPixel
auto bad = pt + ptt;      // compile error -- no tag_sum_result defined
```

### Usage

#### Pixel-to-tangent (1D)

```cpp
#include <image-to-body-math/math.hpp>
using namespace p2b;

auto tan = pixel_tan_from_fov(PixelIndex{15}, ImageSize{20, 10}, Degrees{30}.to_radians());
auto pixel = tan_to_pixel_by_fov(PixelTan{0.1}, ImageSize{640, 480}, Degrees{60}.to_radians());
```

#### Pixel-to-NED (full pipeline)

```cpp
#include <image-to-body-math/body_space.hpp>
using namespace p2b;

const ImageSize size{640, 480};
const auto ptt = pixel_to_tan_from_fov(size, Degrees{60}.to_radians());
const auto cam_q = cam_to_body_from_angle(Degrees{15}.to_radians());
const auto attitude = Quaternion::identity();

// Pixel to NED direction
auto ned = pixel_to_ned(PixelIndex{400}, PixelIndex{300}, size, ptt, cam_q, attitude);

// NED direction back to pixel
auto [row, col] = ned_to_pixel(ned, size, ptt, cam_q, attitude);
```

#### Pixel stabilization (rotation compensation)

```cpp
// Where does pixel (400, 300) end up after the body rotates from q_old to q_new?
auto [new_row, new_col] = pixel_after_rotation(
    PixelIndex{400}, PixelIndex{300}, size, ptt, cam_q, q_old, q_new);
```

#### NED queries

```cpp
// Is a NED direction visible in the current frame?
bool visible = is_ned_inside_frame(ned, size, ptt, cam_q, attitude, 0.1);

// Project pixel to a target elevation (e.g. horizon line at 0 degrees)
auto [r, c] = pixel_at_elevation(
    PixelIndex{400}, PixelIndex{300}, size, ptt, cam_q, attitude, Radians{0.0});

// Angular separation between two NED vectors in pixel units
double px_dist = ned_angle_in_pixels(ned1, ned2, ptt);
```

## Examples

### Image stabilization during flight

Compensate camera image for body rotation between consecutive frames.
Each tracked pixel is re-projected through NED space to find its new position.

```cpp
#include <image-to-body-math/body_space.hpp>
using namespace p2b;

// Camera setup (done once)
const ImageSize frame{1920, 1080};
const auto ptt = pixel_to_tan_from_fov(frame, Degrees{60}.to_radians());
const auto cam_q = cam_to_body_from_angle(Degrees{15}.to_radians()); // 15 deg down-tilt

// Per-frame: get attitude from IMU (scalar-first quaternion: w, x, y, z)
const Quaternion q_prev{0.998, 0.01, 0.02, 0.05};
const Quaternion q_curr{0.997, 0.01, 0.03, 0.06};

// Stabilize a feature point at pixel (960, 400)
auto [new_row, new_col] = pixel_after_rotation(
    PixelIndex{960}, PixelIndex{400}, frame, ptt, cam_q, q_prev, q_curr);
// new_row/new_col = where (960, 400) moved to in the new frame
```

### Target tracking: pixel to world direction

Convert a detected object's bounding box center to a NED direction vector
for geo-referencing or multi-sensor fusion.

```cpp
// Detection at pixel (1200, 600) in a 1920x1080 frame
auto target_ned = pixel_to_ned(
    PixelIndex{1200}, PixelIndex{600}, frame, ptt, cam_q, attitude);

// Get azimuth and elevation of the target
auto [az, el] = ned_to_azimuth_elevation(target_ned);
// az = bearing from north, el = angle above horizon

// Later: project back to pixel when attitude changes
auto [px_row, px_col] = ned_to_pixel(target_ned, frame, ptt, cam_q, new_attitude);
```

### Horizon line detection

Find where the horizon (0 deg elevation) falls in the image.
Useful for sky/ground segmentation or flight display overlays.

```cpp
// Where does the horizon appear at the image center column?
auto [h_row, h_col] = pixel_at_elevation(
    PixelIndex{960}, PixelIndex{540}, frame, ptt, cam_q, attitude, Radians{0.0});
// h_col = vertical pixel position of the horizon at image center

// Check if a tracked target is still visible with 10% margin
bool in_view = is_ned_inside_frame(target_ned, frame, ptt, cam_q, attitude, 0.1);
```

### Measuring angular distance between detections

Compute the angular separation between two detections in pixel units,
useful for clustering or duplicate rejection.

```cpp
auto ned_a = pixel_to_ned(PixelIndex{500}, PixelIndex{300}, frame, ptt, cam_q, attitude);
auto ned_b = pixel_to_ned(PixelIndex{520}, PixelIndex{310}, frame, ptt, cam_q, attitude);

double px_distance = ned_angle_in_pixels(ned_a, ned_b, ptt);
// px_distance ~ 22 pixels (angular separation expressed as pixels)
```

## Performance

`pixel_to_ned` full pipeline (pixel → tangent → body → NED), `640x480` image, identity quaternions. Benchmarked on x86_64 Linux.

### Batch throughput (nanobind vs pure NumPy)

| N pixels | p2b batch | pure NumPy | speedup |
|----------|-----------|------------|---------|
| 100 | 16 us | 180 us | **11x** |
| 1,000 | 69 us | 266 us | **3.9x** |
| 10,000 | 607 us | 1,637 us | **2.7x** |
| 100,000 | 3,749 us | 12,657 us | **3.4x** |

### Single-call latency

| Function | Latency |
|----------|---------|
| `pixel_to_ned` | 2.3 us |
| `ned_to_pixel` | 2.5 us |
| `pixel_after_rotation` | 2.5 us |
| `tangents_to_ned` | 624 ns |
| `cam_to_body_from_angle` | 612 ns |
| `pixel_to_tan_from_fov` | 152 ns |
| `is_pixel_inside_frame` | 157 ns |

Run benchmarks: `python tests/python/bench.py`

## License

MIT
