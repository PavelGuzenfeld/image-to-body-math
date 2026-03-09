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

TEST_CASE("PixelIndex: normalized returns NormalizedPixel")
{
    constexpr ImageSize s{20, 10};
    constexpr PixelIndex center(10);
    static_assert(center.normalized(s).get() == 0.0);

    constexpr PixelIndex left(0);
    static_assert(left.normalized(s).get() == -1.0);

    constexpr PixelIndex right(20);
    static_assert(right.normalized(s).get() == 1.0);
}

TEST_CASE("PixelIndex: offset_from_center")
{
    constexpr ImageSize s{640, 480};
    constexpr PixelIndex center(320);
    static_assert(center.offset_from_center(s) == 0.0);

    constexpr PixelIndex off(480);
    static_assert(off.offset_from_center(s) == 160.0);
}

// =========================================================================
// Strong type safety: compile-time checks
// =========================================================================

TEST_CASE("Strong types: PixelTan and PixelToTan are distinct")
{
    constexpr PixelTan pt{0.5};
    constexpr PixelToTan ptt{0.0025};

    static_assert(!std::is_same_v<PixelTan, PixelToTan>);
    static_assert(!std::is_same_v<PixelTan, ClipThreshold>);
    static_assert(!std::is_same_v<PixelToTan, ClipThreshold>);

    CHECK(pt.get() == 0.5);
    CHECK(ptt.get() == 0.0025);
}

TEST_CASE("Strong types: NormalizedPixel * PixelToTan = PixelTan")
{
    constexpr NormalizedPixel np{0.5};
    constexpr PixelToTan ptt{0.004};
    constexpr auto result = np * ptt;
    static_assert(std::is_same_v<decltype(result), const PixelTan>);
    CHECK(result.get() == doctest::Approx(0.002));
}

TEST_CASE("Strong types: PixelTan / PixelToTan = NormalizedPixel")
{
    constexpr PixelTan pt{0.002};
    constexpr PixelToTan ptt{0.004};
    constexpr auto result = pt / ptt;
    static_assert(std::is_same_v<decltype(result), const NormalizedPixel>);
    CHECK(result.get() == doctest::Approx(0.5));
}

// =========================================================================
// pixel_tan_from_fov
// =========================================================================

TEST_CASE("pixel_tan_from_fov: center pixel gives zero")
{
    const auto result = pixel_tan_from_fov(PixelIndex{12}, ImageSize{24, 0}, Degrees{23}.to_radians());
    CHECK(result.get() == doctest::Approx(0.0).epsilon(EPSILON));
}

TEST_CASE("pixel_tan_from_fov: edge pixel")
{
    const auto result = pixel_tan_from_fov(PixelIndex{20}, ImageSize{20, 0}, Degrees{30}.to_radians());
    CHECK(result.get() == doctest::Approx(std::tan(PI / 12.0)).epsilon(EPSILON));
}

TEST_CASE("pixel_tan_from_fov: negative edge")
{
    const auto result = pixel_tan_from_fov(PixelIndex{0}, ImageSize{480, 0}, Degrees{50}.to_radians());
    CHECK(result.get() == doctest::Approx(std::tan(-PI / 7.2)).epsilon(EPSILON));
}

TEST_CASE("pixel_tan_from_fov: precise value")
{
    const auto result = pixel_tan_from_fov(PixelIndex{15}, ImageSize{20, 0}, Degrees{30}.to_radians());
    CHECK(result.get() == doctest::Approx(std::tan(7.630740212430057 * PI / 180.0)).epsilon(EPSILON));
}

// =========================================================================
// tan_to_pixel_by_fov
// =========================================================================

TEST_CASE("tan_to_pixel_by_fov: center pixel")
{
    const auto result = tan_to_pixel_by_fov(PixelTan{0.0}, ImageSize{640, 480}, Degrees{32}.to_radians());
    CHECK(result.value() == 320);
}

TEST_CASE("tan_to_pixel_by_fov: matches legacy")
{
    const double pixel_tan_val = std::tan(PI / 12.0);
    const auto fov = Degrees{32}.to_radians();
    const auto result = tan_to_pixel_by_fov(PixelTan{pixel_tan_val}, ImageSize{640, 480}, fov);
    const double half = 320.0;
    const auto expected = static_cast<uint64_t>(std::round((pixel_tan_val / std::tan(32.0 * PI / 360.0) + 1.0) * half));
    CHECK(result.value() == expected);
}

// =========================================================================
// pixel_tan_by_pixel_to_tan
// =========================================================================

TEST_CASE("pixel_tan_by_pixel_to_tan: center pixel gives zero")
{
    constexpr auto result = pixel_tan_by_pixel_to_tan(PixelIndex{320}, ImageSize{640, 480}, PixelToTan{0.0025});
    static_assert(result.get() == 0.0);
}

TEST_CASE("pixel_tan_by_pixel_to_tan: off-center")
{
    constexpr auto result = pixel_tan_by_pixel_to_tan(PixelIndex{480}, ImageSize{640, 480}, PixelToTan{0.0035});
    constexpr double expected = (480.0 - 320.0) * 0.0035;
    CHECK(result.get() == doctest::Approx(expected));
}

// =========================================================================
// angle_tan_to_pixel
// =========================================================================

TEST_CASE("angle_tan_to_pixel: zero angle gives center")
{
    const auto result = angle_tan_to_pixel(PixelTan{0.0}, ImageSize{640, 480}, PixelToTan{0.0025});
    CHECK(result.value() == 320);
}

// =========================================================================
// pixel_tan_by_pixel_to_tan_clipped
// =========================================================================

TEST_CASE("pixel_tan_by_pixel_to_tan_clipped: center within threshold")
{
    constexpr auto result = pixel_tan_by_pixel_to_tan_clipped(PixelIndex{320},
                                                              ImageSize{640, 480},
                                                              PixelToTan{0.0025},
                                                              ClipThreshold{0.05});
    static_assert(result.get() == 0.0);
}

TEST_CASE("pixel_tan_by_pixel_to_tan_clipped: outside threshold")
{
    constexpr auto result = pixel_tan_by_pixel_to_tan_clipped(PixelIndex{480},
                                                              ImageSize{640, 480},
                                                              PixelToTan{0.0035},
                                                              ClipThreshold{0.05});
    constexpr double expected = (480.0 - 320.0) * 0.0035;
    CHECK(result.get() == doctest::Approx(expected));
}

// =========================================================================
// tan_to_pixel_by_pixel_to_tan
// =========================================================================

TEST_CASE("tan_to_pixel_by_pixel_to_tan: zero gives center")
{
    const auto result = tan_to_pixel_by_pixel_to_tan(PixelTan{0.0}, ImageSize{640, 480}, PixelToTan{0.0025}, true);
    CHECK(result.value() == 320);
}

TEST_CASE("tan_to_pixel_by_pixel_to_tan: round vs truncate")
{
    const PixelTan pixel_tan{std::tan(PI / 12.0)};
    const ImageSize image_size{640, 480};
    const PixelToTan pixel_to_tan{0.0035};

    const auto rounded = tan_to_pixel_by_pixel_to_tan(pixel_tan, image_size, pixel_to_tan, true);
    const auto truncated = tan_to_pixel_by_pixel_to_tan(pixel_tan, image_size, pixel_to_tan, false);

    CHECK(rounded.value() >= truncated.value());
}
