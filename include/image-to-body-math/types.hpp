#pragma once
#include <cmath>
#include <cstdint>

namespace p2b
{

    struct ImageSize
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

    class PixelIndex
    {
    public:
        uint64_t value;

        explicit constexpr PixelIndex(uint64_t value) noexcept
            : value(value)
        {
        }

        [[nodiscard]] constexpr double normalized(const ImageSize &size) const noexcept
        {
            return static_cast<double>(value) / size.half_width() - 1.0;
        }
    };

} // namespace p2b