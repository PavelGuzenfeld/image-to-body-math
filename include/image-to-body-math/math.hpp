#pragma once
#include "linalg3d/linalg.hpp"
#include "types.hpp"
#include <cmath>

namespace p2b
{
using Radians = linalg3d::Angle<linalg3d::AngleType::RADIANS>;
using Degrees = linalg3d::Angle<linalg3d::AngleType::DEGREES>;

[[nodiscard]] constexpr Radians pixel_tan_from_fov(const PixelIndex &pixel,
                                                   const ImageSize &image_size,
                                                   const Radians &fov) noexcept
{
    const double norm = pixel.normalized(image_size);
    const double half_fov_tan = linalg3d::ce_tan(fov / 2.0);
    const double pixel_angle = linalg3d::ce_atan2(norm * half_fov_tan, 1.0);
    return Radians(pixel_angle);
}

[[nodiscard]] inline PixelIndex tan_to_pixel_by_fov(double pixel_tan,
                                                    const ImageSize &image_size,
                                                    const Radians &fov) noexcept
{
    const double half_fov_tan = linalg3d::ce_tan(fov / 2.0);
    const double norm = pixel_tan / half_fov_tan;
    const auto pixel_value =
        static_cast<uint64_t>(std::round(norm * image_size.half_width() + image_size.half_width()));
    return PixelIndex(pixel_value);
}

[[nodiscard]] constexpr double pixel_tan_by_pixel_to_tan(const PixelIndex &pixel,
                                                         const ImageSize &image_size,
                                                         double pixel_to_tan) noexcept
{
    return (static_cast<double>(pixel.value()) - image_size.half_width()) * pixel_to_tan;
}

[[nodiscard]] inline PixelIndex angle_tan_to_pixel(const Radians &angle_tan,
                                                   const ImageSize &image_size,
                                                   double pixel_to_tan) noexcept
{
    return PixelIndex(static_cast<uint64_t>(std::round(angle_tan.value() / pixel_to_tan + image_size.half_width())));
}

[[nodiscard]] constexpr double pixel_tan_by_pixel_to_tan_clipped(const PixelIndex &pixel,
                                                                 const ImageSize &image_size,
                                                                 double pixel_to_tan,
                                                                 double clipping_threshold) noexcept
{
    const double diff = linalg3d::fabs(static_cast<double>(pixel.value()) - image_size.half_width());

    if (diff < clipping_threshold * image_size.half_width())
    {
        return 0.0;
    }

    return (static_cast<double>(pixel.value()) - image_size.half_width()) * pixel_to_tan;
}

[[nodiscard]] inline PixelIndex tan_to_pixel_by_pixel_to_tan(const Radians &pixel_tan,
                                                             const ImageSize &image_size,
                                                             double pixel_to_tan,
                                                             bool round_back) noexcept
{
    const double pixel_v = pixel_tan.value() / pixel_to_tan + image_size.half_width();
    const auto pixel_value = round_back ? static_cast<uint64_t>(std::round(pixel_v)) : static_cast<uint64_t>(pixel_v);

    return PixelIndex(pixel_value);
}

} // namespace p2b
