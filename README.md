# image-to-body-math

Header-only C++23 library for pixel-to-body coordinate conversions. Converts between image pixel coordinates and angular (tangent-space) representations using camera field-of-view parameters.

## Features

- **Pixel-to-angle conversion** via FOV or pixel-to-tangent factor
- **Angle-to-pixel conversion** with optional rounding
- **Dead-zone clipping** for center-pixel suppression
- **Type-safe angles** via [linalg3d](https://github.com/PavelGuzenfeld/linalg3d) (`Radians` vs `Degrees` at the type level)
- **Type-safe pixel math** via [strong-types](https://github.com/PavelGuzenfeld/strong-types) — `PixelTan`, `PixelToTan`, `NormalizedPixel`, `ClipThreshold` prevent argument mix-ups at compile time
- **constexpr where possible** — pure arithmetic operations evaluate at compile time

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
NormalizedPixel * PixelToTan → PixelTan   // offset × factor = tangent
PixelTan / PixelToTan → NormalizedPixel   // inverse
```

Mixing incompatible types is a compile error:

```cpp
PixelTan pt{0.1};
PixelToTan ptt{0.0025};
auto ok = pt / ptt;       // ✓ NormalizedPixel
auto bad = pt + ptt;      // ✗ compile error — no tag_sum_result defined
```

## Usage

```cpp
#include <image-to-body-math/math.hpp>

using namespace p2b;
using namespace linalg3d;

// Convert pixel 15 in a 20-wide image with 30° FOV to an angle
auto angle = pixel_tan_from_fov(PixelIndex{15}, ImageSize{20, 10}, Degrees{30}.to_radians());

// Convert tangent value back to pixel
auto pixel = tan_to_pixel_by_fov(PixelTan{0.1}, ImageSize{640, 480}, Degrees{60}.to_radians());

// Using pixel-to-tan factor
auto tan = pixel_tan_by_pixel_to_tan(PixelIndex{480}, ImageSize{640, 480}, PixelToTan{0.0035});
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
