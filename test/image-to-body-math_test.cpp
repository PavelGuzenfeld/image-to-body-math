#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "image-to-body-math/math.hpp"
#include <doctest/doctest.h>

using namespace p2b;
using namespace linalg3d;

constexpr double EPSILON = 1e-5;

// =========================================================================
// ImageSize
// =========================================================================

TEST_CASE("ImageSize: default constructor")
{
    constexpr ImageSize s{};
    static_assert(s.width == 0);
    static_assert(s.height == 0);
}

TEST_CASE("ImageSize: parameterized")
{
    constexpr ImageSize s{1920, 1080};
    static_assert(s.width == 1920);
    static_assert(s.height == 1080);
    static_assert(s.half_width() == 960.0);
    static_assert(s.half_height() == 540.0);
}

// =========================================================================
// PixelIndex
// =========================================================================

TEST_CASE("PixelIndex: value access")
{
    constexpr PixelIndex p(42);
    static_assert(p.value() == 42);
}

TEST_CASE("PixelIndex: normalized")
{
    constexpr ImageSize s{20, 10};
    constexpr PixelIndex center(10);
    static_assert(center.normalized(s) == 0.0);

    constexpr PixelIndex left(0);
    static_assert(left.normalized(s) == -1.0);

    constexpr PixelIndex right(20);
    static_assert(right.normalized(s) == 1.0);
}

// =========================================================================
// pixel_tan_from_fov
// =========================================================================

TEST_CASE("pixel_tan_from_fov: center pixel gives zero angle")
{
    const auto result = pixel_tan_from_fov(PixelIndex{12}, ImageSize{24, 0}, Degrees{23}.to_radians());
    CHECK(result.tan() == doctest::Approx(0.0).epsilon(EPSILON));
}

TEST_CASE("pixel_tan_from_fov: edge pixel")
{
    const auto result = pixel_tan_from_fov(PixelIndex{20}, ImageSize{20, 0}, Degrees{30}.to_radians());
    CHECK(result.tan() == doctest::Approx(std::tan(PI / 12.0)).epsilon(EPSILON));
}

TEST_CASE("pixel_tan_from_fov: negative edge")
{
    const auto result = pixel_tan_from_fov(PixelIndex{0}, ImageSize{480, 0}, Degrees{50}.to_radians());
    CHECK(result.tan() == doctest::Approx(std::tan(-PI / 7.2)).epsilon(EPSILON));
}

TEST_CASE("pixel_tan_from_fov: precise value")
{
    const auto result = pixel_tan_from_fov(PixelIndex{15}, ImageSize{20, 0}, Degrees{30}.to_radians());
    CHECK(result.tan() == doctest::Approx(std::tan(7.630740212430057 * PI / 180.0)).epsilon(EPSILON));
}

// =========================================================================
// tan_to_pixel_by_fov
// =========================================================================

TEST_CASE("tan_to_pixel_by_fov: center pixel")
{
    const auto result = tan_to_pixel_by_fov(0.0, ImageSize{640, 480}, Degrees{32}.to_radians());
    CHECK(result.value() == 320);
}

TEST_CASE("tan_to_pixel_by_fov: matches legacy")
{
    const double pixel_tan = std::tan(PI / 12.0);
    const auto fov = Degrees{32}.to_radians();
    const auto result = tan_to_pixel_by_fov(pixel_tan, ImageSize{640, 480}, fov);
    // Legacy: round((pixel_tan / tan(fov*pi/360) + 1) * half)
    const double half = 320.0;
    const auto expected = static_cast<uint64_t>(std::round((pixel_tan / std::tan(32.0 * PI / 360.0) + 1.0) * half));
    CHECK(result.value() == expected);
}

// =========================================================================
// pixel_tan_by_pixel_to_tan
// =========================================================================

TEST_CASE("pixel_tan_by_pixel_to_tan: center pixel gives zero")
{
    constexpr auto result = pixel_tan_by_pixel_to_tan(PixelIndex{320}, ImageSize{640, 480}, 0.0025);
    static_assert(result == 0.0);
}

TEST_CASE("pixel_tan_by_pixel_to_tan: off-center")
{
    constexpr auto result = pixel_tan_by_pixel_to_tan(PixelIndex{480}, ImageSize{640, 480}, 0.0035);
    constexpr double expected = (480.0 - 320.0) * 0.0035;
    CHECK(result == doctest::Approx(expected));
}

// =========================================================================
// angle_tan_to_pixel
// =========================================================================

TEST_CASE("angle_tan_to_pixel: zero angle gives center")
{
    const auto result = angle_tan_to_pixel(Radians{0.0}, ImageSize{640, 480}, 0.0025);
    CHECK(result.value() == 320);
}

// =========================================================================
// pixel_tan_by_pixel_to_tan_clipped
// =========================================================================

TEST_CASE("pixel_tan_by_pixel_to_tan_clipped: center within threshold")
{
    constexpr auto result = pixel_tan_by_pixel_to_tan_clipped(PixelIndex{320}, ImageSize{640, 480}, 0.0025, 0.05);
    static_assert(result == 0.0);
}

TEST_CASE("pixel_tan_by_pixel_to_tan_clipped: outside threshold")
{
    constexpr auto result = pixel_tan_by_pixel_to_tan_clipped(PixelIndex{480}, ImageSize{640, 480}, 0.0035, 0.05);
    constexpr double expected = (480.0 - 320.0) * 0.0035;
    CHECK(result == doctest::Approx(expected));
}

// =========================================================================
// tan_to_pixel_by_pixel_to_tan
// =========================================================================

TEST_CASE("tan_to_pixel_by_pixel_to_tan: zero gives center")
{
    const auto result = tan_to_pixel_by_pixel_to_tan(Radians{0.0}, ImageSize{640, 480}, 0.0025, true);
    CHECK(result.value() == 320);
}

TEST_CASE("tan_to_pixel_by_pixel_to_tan: round vs truncate")
{
    const Radians pixel_tan(std::tan(PI / 12.0));
    const ImageSize image_size{640, 480};
    const double pixel_to_tan = 0.0035;

    const auto rounded = tan_to_pixel_by_pixel_to_tan(pixel_tan, image_size, pixel_to_tan, true);
    const auto truncated = tan_to_pixel_by_pixel_to_tan(pixel_tan, image_size, pixel_to_tan, false);

    // Rounded should be >= truncated (for positive values above center)
    CHECK(rounded.value() >= truncated.value());
}
