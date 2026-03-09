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

## Build

Requires CMake 3.25+ and a C++23 compiler. Dependencies (linalg3d, strong-types, gcem, fmt, doctest) are fetched automatically via FetchContent.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

## License

MIT
