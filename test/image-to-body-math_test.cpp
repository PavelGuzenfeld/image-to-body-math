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

void test_performance()
{
    using namespace p2b;
    using namespace linalg3d;

    fmt::print("Running performance tests...\n");

    double time1 = measure_time([]
                                { return pixel_tan_from_fov(PixelIndex(15), ImageSize{20, 0}, Angle<AngleType::DEGREES>(30)); });

    fmt::print("Execution time of pixel_tan_from_fov: {:.4e} s\n", time1);

    double time2 = measure_time([]
                                { return pixel_tan_from_fov(PixelIndex(15), ImageSize{20, 0}, Angle<AngleType::DEGREES>(30)); });

    fmt::print("Execution time of pixel_tan_by_pixel_2_tan: {:.4e} s\n", time2);
}

int main()
{
    test_ImageSize();
    test_pixel_tan_from_fov();
    test_performance();
    fmt::print("🎉 All Vector3 tests passed successfully!\n");
    return 0;
}