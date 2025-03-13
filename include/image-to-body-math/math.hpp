#pragma once
#include "linalg3d/linalg.hpp"
#include "types.hpp"

namespace p2b
{
    using Radians = linalg3d::Angle<linalg3d::AngleType::RADIANS>;
    using Degrees = linalg3d::Angle<linalg3d::AngleType::DEGREES>;

    [[nodiscard]] constexpr Radians pixel_tan_from_fov(const PixelIndex &pixel, const ImageSize &image_size, const Radians &fov) noexcept
    {
        // Normalize the pixel index to the range [-1, 1]
        double norm = pixel.normalized(image_size);
        // Use half of the field‐of‐view (convert FOV/2 to tangent)
        double half_fov_tan = std::tan(fov / 2.0);
        // Compute the pixel tangent value and then recover the corresponding angle via arctan
        double pixel_angle = std::atan(norm * half_fov_tan);
        return Radians(pixel_angle);
    }

    [[nodiscard]] constexpr PixelIndex tan_2_pixel_by_fov(const double pixel_tan, const ImageSize &image_size, const Radians &fov) noexcept
    {
        double half_fov_tan = std::tan(fov / 2.0);
        double norm = pixel_tan / half_fov_tan;
        uint64_t pixel_value = static_cast<uint64_t>(std::round((norm * image_size.half_width()) + image_size.half_width()));
        return PixelIndex(pixel_value);
    }

    [[nodiscard]] constexpr double pixel_tan_by_pixel_2_tan(const PixelIndex &pixel, const ImageSize &image_size, double pixel_2_tan) noexcept
    {
        return (static_cast<double>(pixel.value) - image_size.half_width()) * pixel_2_tan;
    }

    [[nodiscard]] constexpr PixelIndex angle_tan_to_pixel(const Radians &angle_tan, const ImageSize &image_size, double pixel_2_tan) noexcept
    {
        return PixelIndex(static_cast<uint64_t>(std::round(angle_tan.value() / pixel_2_tan + image_size.half_width())));
    }

    [[nodiscard]] constexpr double pixel_tan_by_pixel_2_tan_clipped(const PixelIndex &pixel, const ImageSize &image_size, double pixel_2_tan, double clipping_threshold) noexcept
    {
        double diff = std::abs(static_cast<double>(pixel.value) - image_size.half_width());

        if (diff < clipping_threshold * image_size.half_width())
        {
            return 0.0;
        }

        return (static_cast<double>(pixel.value) - image_size.half_width()) * pixel_2_tan;
    }

    [[nodiscard]] constexpr PixelIndex tan_2_pixel_by_pixel_2_tan(const Radians &pixel_tan, const ImageSize &image_size, double pixel_2_tan, bool round_back) noexcept
    {
        double pixel_v = pixel_tan.value() / pixel_2_tan + image_size.half_width();
        uint64_t pixel_value = round_back ? static_cast<uint64_t>(std::round(pixel_v))
                                          : static_cast<uint64_t>(pixel_v);

        return PixelIndex(pixel_value);
    }

} // namespace p2b