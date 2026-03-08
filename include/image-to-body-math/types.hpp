#pragma once
#include "strong-types/strong.hpp"
#include <cmath>
#include <cstdint>

namespace p2b
{

// ---- Tags ----

struct PixelTanTag;
struct PixelToTanTag;
struct NormalizedPixelTag;
struct ClipThresholdTag;

// ---- Strong types ----

/// Tangent of a pixel's angular offset from image center
using PixelTan = strong_types::Strong<double, PixelTanTag>;

/// Conversion factor: tangent-per-pixel-offset
using PixelToTan = strong_types::Strong<double, PixelToTanTag>;

/// Normalized pixel coordinate in [-1, 1] (0 = center)
using NormalizedPixel = strong_types::Strong<double, NormalizedPixelTag>;

/// Dead-zone threshold as fraction of half-width [0, 1]
using ClipThreshold = strong_types::Strong<double, ClipThresholdTag>;

// ---- Type algebra ----
// NormalizedPixel * PixelToTan → PixelTan  (conceptually: normalized_offset * tan_per_pixel = tangent)
// PixelTan / PixelToTan → NormalizedPixel  (inverse)

} // namespace strong_types

template <>
struct strong_types::tag_sum_result<p2b::PixelTanTag, p2b::PixelTanTag>
{
    using type = p2b::PixelTanTag;
};
template <>
struct strong_types::tag_difference_result<p2b::PixelTanTag, p2b::PixelTanTag>
{
    using type = p2b::PixelTanTag;
};
template <>
struct strong_types::tag_sum_result<p2b::NormalizedPixelTag, p2b::NormalizedPixelTag>
{
    using type = p2b::NormalizedPixelTag;
};
template <>
struct strong_types::tag_difference_result<p2b::NormalizedPixelTag, p2b::NormalizedPixelTag>
{
    using type = p2b::NormalizedPixelTag;
};

template <>
struct strong_types::tag_product_result<p2b::NormalizedPixelTag, p2b::PixelToTanTag>
{
    using type = p2b::PixelTanTag;
};
template <>
struct strong_types::tag_product_result<p2b::PixelToTanTag, p2b::NormalizedPixelTag>
{
    using type = p2b::PixelTanTag;
};
template <>
struct strong_types::tag_quotient_result<p2b::PixelTanTag, p2b::PixelToTanTag>
{
    using type = p2b::NormalizedPixelTag;
};

namespace p2b
{

// ---- ImageSize ----

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

// ---- PixelIndex ----

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

    [[nodiscard]] constexpr NormalizedPixel normalized(const ImageSize &size) const noexcept
    {
        return NormalizedPixel{static_cast<double>(value_) / size.half_width() - 1.0};
    }

    [[nodiscard]] constexpr double offset_from_center(const ImageSize &size) const noexcept
    {
        return static_cast<double>(value_) - size.half_width();
    }

private:
    uint64_t value_{};
};

// ---- Factory helpers ----

[[nodiscard]] inline PixelIndex pixel_from_rounded(double pixel_v) noexcept
{
    return PixelIndex(static_cast<uint64_t>(std::round(pixel_v)));
}

[[nodiscard]] constexpr PixelIndex pixel_from_truncated(double pixel_v) noexcept
{
    return PixelIndex(static_cast<uint64_t>(pixel_v));
}

} // namespace p2b
