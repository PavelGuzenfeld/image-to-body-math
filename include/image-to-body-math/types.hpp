#pragma once
#include <cmath>
#include <cstdint>

namespace p2b
{

    struct Size2D
    {
        uint64_t width{}, height{};

        [[nodiscard]] constexpr double half_width() const noexcept
        {
            return static_cast<double>(width) / 2.0;
        }

        [[nodiscard]] constexpr double half_height() const noexcept
        {
            return static_cast<double>(height) / 2.0;
        }
    };

} // namespace p2b