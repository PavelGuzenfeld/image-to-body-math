#pragma once
#include <cstdint>

namespace p2b
{

struct ImageSize
{
    uint64_t width{};
    uint64_t height{};

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
    explicit constexpr PixelIndex(uint64_t v) noexcept : value_{v}
    {
    }

    [[nodiscard]] constexpr uint64_t value() const noexcept
    {
        return value_;
    }

    [[nodiscard]] constexpr double normalized(const ImageSize &size) const noexcept
    {
        return static_cast<double>(value_) / size.half_width() - 1.0;
    }

private:
    uint64_t value_{};
};

} // namespace p2b
