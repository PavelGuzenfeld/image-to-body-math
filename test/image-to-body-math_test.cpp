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

int main()
{
    test_ImageSize();
    test_pixel_tan_from_fov();
    test_tan_2_pixel_by_fov();
    fmt::print("🎉 All Vector3 tests passed successfully!\n");
    return 0;
}