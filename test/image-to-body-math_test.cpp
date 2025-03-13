#include "image-to-body-math/types.hpp"
#include <cassert>
#include <fmt/core.h>

constexpr double EPSILON = 1e-5;

inline void assert_near(double actual, double expected, double tolerance = EPSILON)
{
    assert(std::fabs(actual - expected) < tolerance && "Floating-point values are not close enough");
}

void test_size2d()
{
    using namespace p2b;

    fmt::print("Running Size2D tests...\n");

    // Default Constructor
    {
        Size2D s;
        assert(s.width == 0 && s.height == 0);
        fmt::print("✅ Default constructor\n");
    }

    // Parameterized Constructor
    {
        Size2D s{1920, 1080};
        assert(s.width == 1920 && s.height == 1080);
        fmt::print("✅ Parameterized constructor\n");
    }

    // Half Width and Height
    {
        Size2D s{1920, 1080};
        assert(s.half_width() == 960.0);
        assert(s.half_height() == 540.0);
        fmt::print("✅ Half width and height\n");
    }
}

int main()
{
    test_size2d();
    fmt::print("🎉 All Vector3 tests passed successfully!\n");
    return 0;
}