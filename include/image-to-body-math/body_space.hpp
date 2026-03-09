#pragma once
#include "linalg3d/linalg.hpp"
#include "math.hpp"
#include <utility>

namespace p2b
{

using Vector3 = linalg3d::Vector3;
using Quaternion = linalg3d::Quaternion;

// ---- Camera installation ----

/// Create camera-to-body quaternion from installation angle (rotation about Y-axis).
/// A positive angle tilts the camera downward from the body x-axis.
[[nodiscard]] constexpr Quaternion cam_to_body_from_angle(Radians angle) noexcept
{
    const double half = angle.value() / 2.0;
    const double cos_half = linalg3d::ce_cos(half);
    const double sin_half = linalg3d::ce_sin(half);
    // Rotation of -angle about Y: q = (cos(-a/2), 0, sin(-a/2), 0)
    return Quaternion{cos_half, 0.0, -sin_half, 0.0};
}

// ---- Tangent ↔ NED direction ----

/// Convert azimuth/elevation tangent pair to a NED direction vector.
/// w_tan = tan(azimuth), h_tan = tan(elevation)
[[nodiscard]] inline Vector3 tangents_to_ned(double w_tan, double h_tan) noexcept
{
    const double cos_az = 1.0 / std::sqrt(1.0 + w_tan * w_tan);
    const double sin_az = w_tan * cos_az;

    const double cos_el = 1.0 / std::sqrt(1.0 + h_tan * h_tan);
    const double sin_el = h_tan * cos_el;

    return Vector3{cos_el * cos_az, cos_el * sin_az, -sin_el};
}

/// Convert a NED direction vector to azimuth/elevation tangent pair.
/// Returns {tan(azimuth), tan(elevation)}.
[[nodiscard]] inline std::pair<double, double> ned_to_tangents(const Vector3 &ned) noexcept
{
    const double w_tan = ned.y / ned.x;
    const double h_tan = -ned.z / std::sqrt(ned.x * ned.x + ned.y * ned.y);
    return {w_tan, h_tan};
}

/// Convert a NED direction vector to azimuth and elevation angles (radians).
[[nodiscard]] inline std::pair<Radians, Radians> ned_to_azimuth_elevation(const Vector3 &ned) noexcept
{
    const Vector3 n = ned.normalized();
    const double elevation = std::asin(-n.z);
    const double azimuth = std::atan2(n.y, n.x);
    return {Radians{azimuth}, Radians{elevation}};
}

/// Convert azimuth and elevation angles (radians) to a NED direction vector.
[[nodiscard]] inline Vector3 azimuth_elevation_to_ned(Radians azimuth, Radians elevation) noexcept
{
    const double cos_el = std::cos(elevation.value());
    return Vector3{cos_el * std::cos(azimuth.value()),
                   cos_el * std::sin(azimuth.value()),
                   -std::sin(elevation.value())};
}

// ---- Image ↔ Body frame warping ----

/// Convert pixel tangents (width, height) to a body-frame direction vector
/// using the camera-to-body installation quaternion.
/// Note: h_tan is negated (image Y-axis is down, body Z-axis is down in NED).
[[nodiscard]] constexpr Vector3 warp_image_to_body(double w_tan, double h_tan, const Quaternion &cam_to_body) noexcept
{
    const double cos_az = 1.0 / linalg3d::ce_sqrt(1.0 + w_tan * w_tan);
    const double sin_az = w_tan * cos_az;
    const double neg_h = -h_tan;
    const double cos_el = 1.0 / linalg3d::ce_sqrt(1.0 + neg_h * neg_h);
    const double sin_el = neg_h * cos_el;
    const Vector3 dir_cam{cos_el * cos_az, cos_el * sin_az, -sin_el};
    return cam_to_body * dir_cam;
}

/// Convert a body-frame direction vector back to pixel tangent pair (width, height)
/// using the camera-to-body installation quaternion.
[[nodiscard]] inline std::pair<double, double> warp_body_to_image(const Vector3 &dir_body,
                                                                  const Quaternion &cam_to_body) noexcept
{
    const Vector3 dir_cam = cam_to_body.inverse() * dir_body;
    auto [w_tan, h_tan] = ned_to_tangents(dir_cam);
    return {w_tan, -h_tan};
}

// ---- Full pixel ↔ NED pipelines ----

/// Convert pixel coordinates to a NED direction vector.
/// Pipeline: pixel → tangent → body direction → NED direction
[[nodiscard]] inline Vector3 pixel_to_ned(PixelIndex row,
                                          PixelIndex col,
                                          const ImageSize &image_size,
                                          PixelToTan pixel_to_tan,
                                          const Quaternion &cam_to_body,
                                          const Quaternion &attitude) noexcept
{
    const double x_tan = pixel_tan_by_pixel_to_tan(row, image_size, pixel_to_tan).get();
    const double y_tan = pixel_tan_by_pixel_to_tan(col, image_size, pixel_to_tan).get();

    const Vector3 dir_body = warp_image_to_body(x_tan, y_tan, cam_to_body);
    return attitude * dir_body;
}

/// Convert a NED direction vector to pixel coordinates.
/// Pipeline: NED direction → body direction → tangent → pixel
[[nodiscard]] inline std::pair<PixelIndex, PixelIndex> ned_to_pixel(const Vector3 &dir_ned,
                                                                    const ImageSize &image_size,
                                                                    PixelToTan pixel_to_tan,
                                                                    const Quaternion &cam_to_body,
                                                                    const Quaternion &attitude) noexcept
{
    const Vector3 dir_body = attitude.inverse() * dir_ned;
    auto [x_tan, y_tan] = warp_body_to_image(dir_body, cam_to_body);

    const auto row = tan_to_pixel_by_pixel_to_tan(PixelTan{x_tan}, image_size, pixel_to_tan, false);
    const auto col = tan_to_pixel_by_pixel_to_tan(PixelTan{y_tan}, image_size, pixel_to_tan, false);
    return {row, col};
}

// ---- Pixel stabilization (body rotation compensation) ----

/// Compute a pixel's new position after a body orientation change.
/// 6-stage pipeline: pixel → tan → body → NED (q_old) → body (q_new) → tan → pixel
[[nodiscard]] inline std::pair<PixelIndex, PixelIndex> pixel_after_rotation(PixelIndex row,
                                                                            PixelIndex col,
                                                                            const ImageSize &image_size,
                                                                            PixelToTan pixel_to_tan,
                                                                            const Quaternion &cam_to_body,
                                                                            const Quaternion &q_old,
                                                                            const Quaternion &q_new,
                                                                            bool round_back = false) noexcept
{
    // 1. Pixel → tangent
    const double x_tan = pixel_tan_by_pixel_to_tan(row, image_size, pixel_to_tan).get();
    const double y_tan = pixel_tan_by_pixel_to_tan(col, image_size, pixel_to_tan).get();

    // 2. Tangent → body direction
    const Vector3 dir_body = warp_image_to_body(x_tan, y_tan, cam_to_body);

    // 3. Body → NED (old attitude)
    const Vector3 dir_ned = q_old * dir_body;

    // 4. NED → body (new attitude)
    const Vector3 dir_body_new = q_new.inverse() * dir_ned;

    // 5. Body → tangent
    auto [x_tan_new, y_tan_new] = warp_body_to_image(dir_body_new, cam_to_body);

    // 6. Tangent → pixel
    const auto row_new = tan_to_pixel_by_pixel_to_tan(PixelTan{x_tan_new}, image_size, pixel_to_tan, round_back);
    const auto col_new = tan_to_pixel_by_pixel_to_tan(PixelTan{y_tan_new}, image_size, pixel_to_tan, round_back);
    return {row_new, col_new};
}

// ---- Boundary checking ----

/// Check if a pixel is inside the frame with a safety margin.
/// boundary < 1.0: fraction of image size (e.g., 0.1 = 10% margin)
/// boundary >= 1.0: absolute pixel margin
[[nodiscard]] constexpr bool is_pixel_inside_frame(PixelIndex row,
                                                   PixelIndex col,
                                                   const ImageSize &image_size,
                                                   double boundary) noexcept
{
    double margin_x = boundary;
    double margin_y = boundary;

    if (boundary < 1.0)
    {
        margin_x = boundary * static_cast<double>(image_size.width);
        margin_y = boundary * static_cast<double>(image_size.height);
    }

    const auto r = static_cast<double>(row.value());
    const auto c = static_cast<double>(col.value());
    const auto w = static_cast<double>(image_size.width);
    const auto h = static_cast<double>(image_size.height);

    return r >= margin_x && r <= (w - margin_x) && c >= margin_y && c <= (h - margin_y);
}

/// Compute the PixelToTan factor from camera FOV and image size.
/// pixel_to_tan = tan(fov/2) / half_width
[[nodiscard]] constexpr PixelToTan pixel_to_tan_from_fov(const ImageSize &image_size, const Radians &fov) noexcept
{
    return PixelToTan{linalg3d::ce_tan(fov / 2.0) / image_size.half_width()};
}

} // namespace p2b
