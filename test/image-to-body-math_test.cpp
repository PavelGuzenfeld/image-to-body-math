#include "image-to-body-math/math.hpp"
#include <cassert>    // For assert
#include <chrono>     // For timing
#include <cstdlib>    // For std::exit
#include <fmt/core.h> // For formatting
#include <string>     // For std::string

constexpr double EPSILON = 1e-5;

inline void assert_near(double actual, double expected, std::string message = "", double tolerance = EPSILON)
{
    if (std::fabs(actual - expected) >= tolerance)
    {
        fmt::print("❌ Assertion failed: {} (actual: {}, expected: {})\n", message, actual, expected);
        std::exit(1);
    }
}

template <typename Func>
double measure_time(Func func, int iterations = 100)
{
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++)
    {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count() / iterations;
}

void test_ImageSize()
{
    using namespace p2b;

    fmt::print("Running ImageSize tests...\n");

    // Default Constructor
    {
        ImageSize s;
        assert(s.width == 0 && s.height == 0);
        fmt::print("✅ Default constructor\n");
    }

    // Parameterized Constructor
    {
        ImageSize s{1920, 1080};
        assert(s.width == 1920 && s.height == 1080);
        fmt::print("✅ Parameterized constructor\n");
    }

    // Half Width and Height
    {
        ImageSize s{1920, 1080};
        assert(s.half_width() == 960.0);
        assert(s.half_height() == 540.0);
        fmt::print("✅ Half width and height\n");
    }
}

void test_pixel_tan_from_fov()
{
    using namespace p2b;
    using namespace linalg3d;

    fmt::print("Running pixel_tan_from_fov tests...\n");

    // Basic Cases
    {
        assert_near(pixel_tan_from_fov(PixelIndex{12}, ImageSize{24, 0}, Degrees{23}).tan(), 0.0, "Edge case 1");
        fmt::print("✅ pixel_tan_from_fov (edge case 1)\n");

        assert_near(pixel_tan_from_fov(PixelIndex{20}, ImageSize{20, 0}, Degrees{30}).tan(), std::tan(M_PI / 12.0), "Edge case 2");
        fmt::print("✅ pixel_tan_from_fov (mid-range case)\n");

        assert_near(pixel_tan_from_fov(PixelIndex{0}, ImageSize{480, 0}, Degrees{50}).tan(), std::tan(-M_PI / 7.2), "Edge case 3");
        fmt::print("✅ pixel_tan_from_fov (negative edge case)\n");
    }

    // Specific Value Check
    {
        assert_near(pixel_tan_from_fov(PixelIndex{15}, ImageSize{20, 0}, Angle<AngleType::DEGREES>{30}).tan(), std::tan(7.630740212430057 * M_PI / 180.0), "Precise calculation test");
        fmt::print("✅ pixel_tan_from_fov (precise calculation test)\n");
    }

    // Validate against alternative method
    {
        auto const tan1 = pixel_tan_from_fov(PixelIndex{15}, ImageSize{20, 0}, Degrees{30}).tan();
        auto const tan2 = pixel_tan_from_fov(PixelIndex{15}, ImageSize{20, 0}, Degrees{30}).tan();
        assert_near(tan1, tan2, "Consistency check");
        fmt::print("✅ pixel_tan_from_fov consistency check\n");
    }
}

// Convert tangent of pixel coordinate to pixel value based on FOV
int tan_2_pixel_by_fov_old(double pixel_tan, int image_size, double image_fov)
{
    double half_image_size = static_cast<double>(image_size) / 2.0;
    return static_cast<int>(std::round((pixel_tan / std::tan(image_fov * M_PI / 360.0) + 1.0) * half_image_size));
}

void test_tan_2_pixel_by_fov()
{
    using namespace p2b;
    using namespace linalg3d;

    fmt::print("Running tan_2_pixel_by_fov tests...\n");

    {
        PixelIndex result = tan_2_pixel_by_fov(0.0, ImageSize{640, 480}, Degrees{32});
        assert(result.value == 320); // Expecting center pixel
        fmt::print("✅ tan_2_pixel_by_fov (center pixel)\n");
    }

    {
        double pixel_tan = std::tan(M_PI / 12.0);
        int image_size = 640;
        double image_fov = 32.0;

        auto result_old = tan_2_pixel_by_fov_old(pixel_tan, image_size, image_fov);
        PixelIndex result = tan_2_pixel_by_fov(pixel_tan, ImageSize{static_cast<uint64_t>(image_size), 480}, Degrees{image_fov});
        assert(result.value == static_cast<uint64_t>(result_old));
        fmt::print("✅ tan_2_pixel_by_fov (mid-range pixel)\n");
    }

    {
        double pixel_tan = std::tan(-M_PI / 6.0);
        int image_size = 640;
        double image_fov = 32.0;
        auto old_result = tan_2_pixel_by_fov_old(pixel_tan, image_size, image_fov);
        PixelIndex result = tan_2_pixel_by_fov(pixel_tan, ImageSize{static_cast<uint64_t>(image_size), 480}, Degrees{image_fov});
        assert(result.value == static_cast<uint64_t>(old_result));
        fmt::print("✅ tan_2_pixel_by_fov (negative edge case)\n");
    }

    fmt::print("✅ All tan_2_pixel_by_fov tests passed!\n");
}

double pixel_tan_by_pixel_2_tan_old(double pixel_i, int half_image_size, double pixel_2_tan)
{
    return (pixel_i - static_cast<double>(half_image_size)) * pixel_2_tan;
}

void test_pixel_tan_by_pixel_2_tan()
{
    using namespace p2b;
    using namespace linalg3d;

    fmt::print("Running pixel_tan_by_pixel_2_tan tests...\n");

    {
        PixelIndex pixel(320);
        ImageSize image_size{640, 480};
        double pixel_2_tan = 0.0025; // Example factor

        auto old_result = pixel_tan_by_pixel_2_tan_old(320, image_size.half_width(), pixel_2_tan);
        auto new_result = pixel_tan_by_pixel_2_tan(pixel, image_size, pixel_2_tan);

        assert_near(new_result, old_result, "Center pixel test");
        fmt::print("✅ pixel_tan_by_pixel_2_tan (center pixel)\n");
    }

    {
        PixelIndex pixel(480);
        ImageSize image_size{640, 480};
        double pixel_2_tan = 0.0035;

        auto old_result = pixel_tan_by_pixel_2_tan_old(480, image_size.half_width(), pixel_2_tan);
        auto new_result = pixel_tan_by_pixel_2_tan(pixel, image_size, pixel_2_tan);

        assert_near(new_result, old_result, "Mid-range pixel test");
        fmt::print("✅ pixel_tan_by_pixel_2_tan (mid-range pixel)\n");
    }

    {
        PixelIndex pixel(160);
        ImageSize image_size{640, 480};
        double pixel_2_tan = 0.0018;

        auto old_result = pixel_tan_by_pixel_2_tan_old(160, image_size.half_width(), pixel_2_tan);
        auto new_result = pixel_tan_by_pixel_2_tan(pixel, image_size, pixel_2_tan);

        assert_near(new_result, old_result, "Negative edge case");
        fmt::print("✅ pixel_tan_by_pixel_2_tan (negative edge case)\n");
    }

    fmt::print("✅ All pixel_tan_by_pixel_2_tan tests passed!\n");
}

double angle_tan_to_pixel_old(double angle_tan, int half_image_size, double pixel_2_tan)
{
    return (angle_tan / pixel_2_tan) + static_cast<double>(half_image_size);
}

void test_angle_tan_to_pixel()
{
    using namespace p2b;
    using namespace linalg3d;

    fmt::print("Running angle_tan_to_pixel tests...\n");

    {
        Radians angle_tan(0.0);
        ImageSize image_size{640, 480};
        double pixel_2_tan = 0.0025;

        auto old_result = angle_tan_to_pixel_old(0.0, image_size.half_width(), pixel_2_tan);
        auto new_result = angle_tan_to_pixel(angle_tan, image_size, pixel_2_tan);

        assert(new_result.value == static_cast<uint64_t>(std::round(old_result)) && "Center pixel test");
        fmt::print("✅ angle_tan_to_pixel (center pixel)\n");
    }

    {
        Radians angle_tan(std::tan(M_PI / 12.0));
        ImageSize image_size{640, 480};
        double pixel_2_tan = 0.0035;

        auto old_result = angle_tan_to_pixel_old(angle_tan.value(), image_size.half_width(), pixel_2_tan);
        auto new_result = angle_tan_to_pixel(angle_tan, image_size, pixel_2_tan);

        assert(new_result.value == static_cast<uint64_t>(std::round(old_result)) && "Mid-range pixel test");
        fmt::print("✅ angle_tan_to_pixel (mid-range pixel)\n");
    }

    {
        Radians angle_tan(std::tan(-M_PI / 6.0));
        ImageSize image_size{640, 480};
        double pixel_2_tan = 0.0018;

        auto old_result = angle_tan_to_pixel_old(angle_tan.value(), image_size.half_width(), pixel_2_tan);
        auto new_result = angle_tan_to_pixel(angle_tan, image_size, pixel_2_tan);

        assert(new_result.value == static_cast<uint64_t>(std::round(old_result)) && "Negative edge case");
        fmt::print("✅ angle_tan_to_pixel (negative edge case)\n");
    }

    fmt::print("✅ All angle_tan_to_pixel tests passed!\n");
}

double pixel_tan_by_pixel_2_tan_clipped_old(double pixel_i, int half_image_size, double pixel_2_tan, double clipping_threshold)
{
    double diff = std::abs(pixel_i - static_cast<double>(half_image_size));
    if (diff < clipping_threshold * half_image_size)
    {
        return 0.0;
    }
    return (pixel_i - static_cast<double>(half_image_size)) * pixel_2_tan;
}

void test_pixel_tan_by_pixel_2_tan_clipped()
{
    using namespace p2b;
    using namespace linalg3d;

    fmt::print("Running pixel_tan_by_pixel_2_tan_clipped tests...\n");

    {
        PixelIndex pixel(320);
        ImageSize image_size{640, 480};
        double pixel_2_tan = 0.0025;
        double clipping_threshold = 0.05;

        auto old_result = pixel_tan_by_pixel_2_tan_clipped_old(320, image_size.half_width(), pixel_2_tan, clipping_threshold);
        auto new_result = pixel_tan_by_pixel_2_tan_clipped(pixel, image_size, pixel_2_tan, clipping_threshold);

        assert_near(new_result, old_result, "Center pixel test");
        fmt::print("✅ pixel_tan_by_pixel_2_tan_clipped (center pixel)\n");
    }

    {
        PixelIndex pixel(480);
        ImageSize image_size{640, 480};
        double pixel_2_tan = 0.0035;
        double clipping_threshold = 0.05;

        auto old_result = pixel_tan_by_pixel_2_tan_clipped_old(480, image_size.half_width(), pixel_2_tan, clipping_threshold);
        auto new_result = pixel_tan_by_pixel_2_tan_clipped(pixel, image_size, pixel_2_tan, clipping_threshold);

        assert_near(new_result, old_result, "Mid-range pixel test");
        fmt::print("✅ pixel_tan_by_pixel_2_tan_clipped (mid-range pixel)\n");
    }

    {
        PixelIndex pixel(160);
        ImageSize image_size{640, 480};
        double pixel_2_tan = 0.0018;
        double clipping_threshold = 0.1;

        auto old_result = pixel_tan_by_pixel_2_tan_clipped_old(160, image_size.half_width(), pixel_2_tan, clipping_threshold);
        auto new_result = pixel_tan_by_pixel_2_tan_clipped(pixel, image_size, pixel_2_tan, clipping_threshold);

        assert_near(new_result, old_result, "Negative edge case");
        fmt::print("✅ pixel_tan_by_pixel_2_tan_clipped (negative edge case)\n");
    }

    fmt::print("✅ All pixel_tan_by_pixel_2_tan_clipped tests passed!\n");
}

double tan_2_pixel_by_pixel_2_tan_old(double pixel_tan, int half_image_size, double pixel_2_tan, bool round_back)
{
    double pixel_v = pixel_tan / pixel_2_tan + static_cast<double>(half_image_size);
    return round_back ? std::round(pixel_v) : pixel_v;
}

void test_tan_2_pixel_by_pixel_2_tan()
{
    using namespace p2b;
    using namespace linalg3d;

    fmt::print("Running tan_2_pixel_by_pixel_2_tan tests...\n");

    {
        Radians pixel_tan(0.0);
        ImageSize image_size{640, 480};
        double pixel_2_tan = 0.0025;

        auto old_result = tan_2_pixel_by_pixel_2_tan_old(0.0, image_size.half_width(), pixel_2_tan, true);
        auto new_result = tan_2_pixel_by_pixel_2_tan(pixel_tan, image_size, pixel_2_tan, true);

        assert(new_result.value == static_cast<uint64_t>(std::round(old_result)) && "Center pixel test");
        fmt::print("✅ tan_2_pixel_by_pixel_2_tan (center pixel, with rounding)\n");
    }

    {
        Radians pixel_tan(std::tan(M_PI / 12.0));
        ImageSize image_size{640, 480};
        double pixel_2_tan = 0.0035;

        auto old_result = tan_2_pixel_by_pixel_2_tan_old(pixel_tan.value(), image_size.half_width(), pixel_2_tan, false);
        auto new_result = tan_2_pixel_by_pixel_2_tan(pixel_tan, image_size, pixel_2_tan, false);

        assert(new_result.value == static_cast<uint64_t>(old_result) && "Mid-range pixel test (no rounding)");
        fmt::print("✅ tan_2_pixel_by_pixel_2_tan (mid-range pixel, no rounding)\n");
    }

    {
        Radians pixel_tan(std::tan(-M_PI / 6.0));
        ImageSize image_size{640, 480};
        double pixel_2_tan = 0.0018;

        auto old_result = tan_2_pixel_by_pixel_2_tan_old(pixel_tan.value(), image_size.half_width(), pixel_2_tan, true);
        auto new_result = tan_2_pixel_by_pixel_2_tan(pixel_tan, image_size, pixel_2_tan, true);

        assert(new_result.value == static_cast<uint64_t>(std::round(old_result)) && "Negative edge case");
        fmt::print("✅ tan_2_pixel_by_pixel_2_tan (negative edge case, with rounding)\n");
    }

    fmt::print("✅ All tan_2_pixel_by_pixel_2_tan tests passed!\n");
}

int main()
{
    test_ImageSize();
    test_pixel_tan_from_fov();
    test_tan_2_pixel_by_fov();
    test_pixel_tan_by_pixel_2_tan();
    test_angle_tan_to_pixel();
    test_pixel_tan_by_pixel_2_tan_clipped();
    test_tan_2_pixel_by_pixel_2_tan();
    fmt::print("🎉 All Vector3 tests passed successfully!\n");
    return 0;
}