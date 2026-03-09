# image-to-body-math

Header-only C++23 library for pixel-to-body coordinate conversions. Converts between image pixel coordinates, tangent-space, body-frame, and NED direction representations using camera parameters and attitude quaternions.

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

## Headers

| Header | Purpose |
|--------|---------|
| `types.hpp` | Strong type definitions, `ImageSize`, `PixelIndex` |
| `math.hpp` | 1D pixel-to-tangent conversions (FOV and pixel-to-tan factor) |
| `body_space.hpp` | 2D image-to-body-to-NED pipeline, rotation stabilization |

## Strong Types

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

## Usage

### Pixel-to-tangent (1D)

```cpp
#include <image-to-body-math/math.hpp>
using namespace p2b;

auto tan = pixel_tan_from_fov(PixelIndex{15}, ImageSize{20, 10}, Degrees{30}.to_radians());
auto pixel = tan_to_pixel_by_fov(PixelTan{0.1}, ImageSize{640, 480}, Degrees{60}.to_radians());
```

### Pixel-to-NED (full pipeline)

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

### Pixel stabilization (rotation compensation)

```cpp
// Where does pixel (400, 300) end up after the body rotates from q_old to q_new?
auto [new_row, new_col] = pixel_after_rotation(
    PixelIndex{400}, PixelIndex{300}, size, ptt, cam_q, q_old, q_new);
```

### NED queries

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

## Build

Requires CMake 3.25+ and a C++23 compiler. Dependencies (linalg3d, strong-types, gcem, fmt, doctest) are fetched automatically via FetchContent.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

## License

MIT
