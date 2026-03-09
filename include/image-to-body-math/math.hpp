#pragma once
#include "linalg3d/linalg.hpp"
#include "types.hpp"
#include <cmath>

namespace p2b
{
using Radians = linalg3d::Angle<linalg3d::AngleType::RADIANS>;
using Degrees = linalg3d::Angle<linalg3d::AngleType::DEGREES>;

/// Convert a pixel index to its angular tangent using camera FOV.
/// PixelIndex → NormalizedPixel → PixelTan (direct: norm * tan(fov/2))
[[nodiscard]] constexpr PixelTan pixel_tan_from_fov(const PixelIndex &pixel,
                                                    const ImageSize &image_size,
                                                    const Radians &fov) noexcept
{
    const NormalizedPixel norm = pixel.normalized(image_size);
    const double half_fov_tan = linalg3d::ce_tan(fov / 2.0);
    return PixelTan{norm.get() * half_fov_tan};
}

/// Convert a tangent value back to a pixel index using camera FOV.
/// PixelTan → NormalizedPixel → PixelIndex (rounded)
[[nodiscard]] inline PixelIndex tan_to_pixel_by_fov(PixelTan pixel_tan,
                                                    const ImageSize &image_size,
                                                    const Radians &fov) noexcept
{
    const double half_fov_tan = linalg3d::ce_tan(fov / 2.0);
    const double norm = pixel_tan.get() / half_fov_tan;
    return pixel_from_rounded(norm * image_size.half_width() + image_size.half_width());
}

/// Convert a pixel index to its tangent using a pixel-to-tangent factor.
/// PixelIndex → offset_from_center * PixelToTan → PixelTan
[[nodiscard]] constexpr PixelTan pixel_tan_by_pixel_to_tan(const PixelIndex &pixel,
                                                           const ImageSize &image_size,
                                                           PixelToTan pixel_to_tan) noexcept
{
    return PixelTan{pixel.offset_from_center(image_size) * pixel_to_tan.get()};
}

/// Convert an angular tangent to a pixel index using a pixel-to-tangent factor.
/// PixelTan → offset / PixelToTan → PixelIndex (rounded)
[[nodiscard]] inline PixelIndex angle_tan_to_pixel(PixelTan angle_tan,
                                                   const ImageSize &image_size,
                                                   PixelToTan pixel_to_tan) noexcept
{
    return pixel_from_rounded(angle_tan.get() / pixel_to_tan.get() + image_size.half_width());
}

/// Convert a pixel index to its tangent with dead-zone clipping around center.
/// Returns PixelTan{0} if the pixel is within the clipping threshold of center.
[[nodiscard]] constexpr PixelTan pixel_tan_by_pixel_to_tan_clipped(const PixelIndex &pixel,
                                                                   const ImageSize &image_size,
                                                                   PixelToTan pixel_to_tan,
                                                                   ClipThreshold threshold) noexcept
{
    const double offset = pixel.offset_from_center(image_size);
    const double diff = linalg3d::fabs(offset);

    if (diff < threshold.get() * image_size.half_width())
    {
        return PixelTan{0.0};
    }

    return PixelTan{offset * pixel_to_tan.get()};
}

/// Convert a tangent value to a pixel index using a pixel-to-tangent factor.
/// PixelTan → offset / PixelToTan → PixelIndex (rounded or truncated)
[[nodiscard]] inline PixelIndex tan_to_pixel_by_pixel_to_tan(PixelTan pixel_tan,
                                                             const ImageSize &image_size,
                                                             PixelToTan pixel_to_tan,
                                                             bool round_back) noexcept
{
    const double pixel_v = pixel_tan.get() / pixel_to_tan.get() + image_size.half_width();
    return round_back ? pixel_from_rounded(pixel_v) : pixel_from_truncated(pixel_v);
}

} // namespace p2b
