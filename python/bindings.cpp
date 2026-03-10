#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>

#include "image-to-body-math/body_space.hpp"
#include "image-to-body-math/math.hpp"

namespace nb = nanobind;
using namespace nb::literals;

using Vec3In = nb::ndarray<const double, nb::shape<3>, nb::c_contig, nb::device::cpu>;
using QuatIn = nb::ndarray<const double, nb::shape<4>, nb::c_contig, nb::device::cpu>;
using U64_1D = nb::ndarray<const uint64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using F64_1D = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using F64_2D = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

// ---- Zero-copy helpers ----

static p2b::Quaternion to_quat(QuatIn a)
{
    const double *d = a.data();
    return p2b::Quaternion{d[0], d[1], d[2], d[3]};
}

static p2b::Vector3 to_vec3(Vec3In a)
{
    const double *d = a.data();
    return p2b::Vector3{d[0], d[1], d[2]};
}

static auto make_vec3(const p2b::Vector3 &v)
{
    auto *data = new double[3]{v.x, v.y, v.z};
    nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<double *>(p); });
    return nb::ndarray<nb::numpy, double, nb::shape<3>>(data, {3}, owner);
}

static auto make_quat(const p2b::Quaternion &q)
{
    auto *data = new double[4]{q.w, q.x, q.y, q.z};
    nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<double *>(p); });
    return nb::ndarray<nb::numpy, double, nb::shape<4>>(data, {4}, owner);
}

NB_MODULE(_core, m)
{
    m.doc() = "Image-to-body coordinate transformations (C++ core via nanobind)";

    // ---- ImageSize ----
    nb::class_<p2b::ImageSize>(m, "ImageSize")
        .def(nb::init<uint64_t, uint64_t>(), "width"_a, "height"_a)
        .def_rw("width", &p2b::ImageSize::width)
        .def_rw("height", &p2b::ImageSize::height)
        .def("half_width", &p2b::ImageSize::half_width)
        .def("half_height", &p2b::ImageSize::half_height)
        .def("__repr__",
             [](const p2b::ImageSize &s)
             { return "ImageSize(width=" + std::to_string(s.width) + ", height=" + std::to_string(s.height) + ")"; });

    // ============================================================
    //  1D pixel-tangent conversions  (math.hpp)
    // ============================================================

    m.def(
        "pixel_tan_from_fov",
        [](uint64_t pixel, uint64_t w, uint64_t h, double fov)
        { return p2b::pixel_tan_from_fov(p2b::PixelIndex{pixel}, p2b::ImageSize{w, h}, p2b::Radians{fov}).get(); },
        "pixel"_a, "width"_a, "height"_a, "fov_rad"_a,
        "Pixel index -> angular tangent via camera FOV (radians).");

    m.def(
        "tan_to_pixel_by_fov",
        [](double pt, uint64_t w, uint64_t h, double fov) -> uint64_t
        { return p2b::tan_to_pixel_by_fov(p2b::PixelTan{pt}, p2b::ImageSize{w, h}, p2b::Radians{fov}).value(); },
        "pixel_tan"_a, "width"_a, "height"_a, "fov_rad"_a, "Tangent -> pixel index via FOV.");

    m.def(
        "pixel_tan_by_pixel_to_tan",
        [](uint64_t pixel, uint64_t w, uint64_t h, double p2t)
        {
            return p2b::pixel_tan_by_pixel_to_tan(p2b::PixelIndex{pixel}, p2b::ImageSize{w, h}, p2b::PixelToTan{p2t})
                .get();
        },
        "pixel"_a, "width"_a, "height"_a, "pixel_to_tan"_a, "Pixel index -> tangent via pixel-to-tan factor.");

    m.def(
        "angle_tan_to_pixel",
        [](double at, uint64_t w, uint64_t h, double p2t) -> uint64_t
        { return p2b::angle_tan_to_pixel(p2b::PixelTan{at}, p2b::ImageSize{w, h}, p2b::PixelToTan{p2t}).value(); },
        "angle_tan"_a, "width"_a, "height"_a, "pixel_to_tan"_a, "Angular tangent -> pixel index.");

    m.def(
        "pixel_tan_by_pixel_to_tan_clipped",
        [](uint64_t pixel, uint64_t w, uint64_t h, double p2t, double thr)
        {
            return p2b::pixel_tan_by_pixel_to_tan_clipped(p2b::PixelIndex{pixel}, p2b::ImageSize{w, h},
                                                          p2b::PixelToTan{p2t}, p2b::ClipThreshold{thr})
                .get();
        },
        "pixel"_a, "width"_a, "height"_a, "pixel_to_tan"_a, "threshold"_a,
        "Pixel -> tangent with dead-zone clipping around center.");

    m.def(
        "tan_to_pixel_by_pixel_to_tan",
        [](double pt, uint64_t w, uint64_t h, double p2t, bool round_back) -> uint64_t
        {
            return p2b::tan_to_pixel_by_pixel_to_tan(p2b::PixelTan{pt}, p2b::ImageSize{w, h}, p2b::PixelToTan{p2t},
                                                     round_back)
                .value();
        },
        "pixel_tan"_a, "width"_a, "height"_a, "pixel_to_tan"_a, "round_back"_a = false,
        "Tangent -> pixel index via pixel-to-tan factor.");

    m.def(
        "pixel_to_tan_from_fov",
        [](uint64_t w, uint64_t h, double fov)
        { return p2b::pixel_to_tan_from_fov(p2b::ImageSize{w, h}, p2b::Radians{fov}).get(); }, "width"_a, "height"_a,
        "fov_rad"_a, "Compute pixel-to-tan conversion factor from FOV.");

    // ============================================================
    //  Body space  (body_space.hpp)
    // ============================================================

    m.def(
        "cam_to_body_from_angle",
        [](double angle) { return make_quat(p2b::cam_to_body_from_angle(p2b::Radians{angle})); }, "angle_rad"_a,
        "Camera-to-body quaternion [w,x,y,z] from installation angle.");

    m.def(
        "tangents_to_ned", [](double wt, double ht) { return make_vec3(p2b::tangents_to_ned(wt, ht)); }, "w_tan"_a,
        "h_tan"_a, "Azimuth/elevation tangent pair -> NED direction.");

    m.def(
        "ned_to_tangents", [](Vec3In ned) { return p2b::ned_to_tangents(to_vec3(ned)); }, "ned"_a,
        "NED direction -> tangent pair (w_tan, h_tan).");

    m.def(
        "ned_to_azimuth_elevation",
        [](Vec3In ned) -> std::pair<double, double>
        {
            auto [az, el] = p2b::ned_to_azimuth_elevation(to_vec3(ned));
            return {az.value(), el.value()};
        },
        "ned"_a, "NED direction -> (azimuth, elevation) in radians.");

    m.def(
        "azimuth_elevation_to_ned",
        [](double az, double el) { return make_vec3(p2b::azimuth_elevation_to_ned(p2b::Radians{az}, p2b::Radians{el})); },
        "azimuth"_a, "elevation"_a, "Azimuth/elevation (radians) -> NED direction.");

    m.def(
        "warp_image_to_body",
        [](double wt, double ht, QuatIn q) { return make_vec3(p2b::warp_image_to_body(wt, ht, to_quat(q))); },
        "w_tan"_a, "h_tan"_a, "cam_to_body"_a, "Image tangents -> body-frame direction.");

    m.def(
        "warp_body_to_image",
        [](Vec3In dir, QuatIn q) { return p2b::warp_body_to_image(to_vec3(dir), to_quat(q)); }, "dir_body"_a,
        "cam_to_body"_a, "Body-frame direction -> image tangent pair.");

    m.def(
        "pixel_to_ned",
        [](uint64_t row, uint64_t col, uint64_t w, uint64_t h, double p2t, QuatIn cam, QuatIn att)
        {
            return make_vec3(p2b::pixel_to_ned(p2b::PixelIndex{row}, p2b::PixelIndex{col}, p2b::ImageSize{w, h},
                                               p2b::PixelToTan{p2t}, to_quat(cam), to_quat(att)));
        },
        "row"_a, "col"_a, "width"_a, "height"_a, "pixel_to_tan"_a, "cam_to_body"_a, "attitude"_a,
        "Pixel -> NED direction vector.");

    m.def(
        "ned_to_pixel",
        [](Vec3In ned, uint64_t w, uint64_t h, double p2t, QuatIn cam, QuatIn att) -> std::pair<uint64_t, uint64_t>
        {
            auto [r, c] = p2b::ned_to_pixel(to_vec3(ned), p2b::ImageSize{w, h}, p2b::PixelToTan{p2t}, to_quat(cam),
                                             to_quat(att));
            return {r.value(), c.value()};
        },
        "dir_ned"_a, "width"_a, "height"_a, "pixel_to_tan"_a, "cam_to_body"_a, "attitude"_a,
        "NED direction -> pixel coordinates (row, col).");

    m.def(
        "pixel_after_rotation",
        [](uint64_t row, uint64_t col, uint64_t w, uint64_t h, double p2t, QuatIn cam, QuatIn qo, QuatIn qn,
           bool rb) -> std::pair<uint64_t, uint64_t>
        {
            auto [r, c] = p2b::pixel_after_rotation(p2b::PixelIndex{row}, p2b::PixelIndex{col}, p2b::ImageSize{w, h},
                                                    p2b::PixelToTan{p2t}, to_quat(cam), to_quat(qo), to_quat(qn), rb);
            return {r.value(), c.value()};
        },
        "row"_a, "col"_a, "width"_a, "height"_a, "pixel_to_tan"_a, "cam_to_body"_a, "q_old"_a, "q_new"_a,
        "round_back"_a = false, "Pixel position after body rotation change.");

    m.def(
        "is_pixel_inside_frame",
        [](uint64_t row, uint64_t col, uint64_t w, uint64_t h, double boundary)
        { return p2b::is_pixel_inside_frame(p2b::PixelIndex{row}, p2b::PixelIndex{col}, p2b::ImageSize{w, h}, boundary); },
        "row"_a, "col"_a, "width"_a, "height"_a, "boundary"_a, "Check if pixel is inside frame with safety margin.");

    m.def(
        "is_ned_inside_frame",
        [](Vec3In ned, uint64_t w, uint64_t h, double p2t, QuatIn cam, QuatIn att, double boundary)
        {
            return p2b::is_ned_inside_frame(to_vec3(ned), p2b::ImageSize{w, h}, p2b::PixelToTan{p2t}, to_quat(cam),
                                            to_quat(att), boundary);
        },
        "dir_ned"_a, "width"_a, "height"_a, "pixel_to_tan"_a, "cam_to_body"_a, "attitude"_a, "boundary"_a,
        "Check if NED direction projects inside frame.");

    m.def(
        "pixel_at_elevation",
        [](uint64_t row, uint64_t col, uint64_t w, uint64_t h, double p2t, QuatIn cam, QuatIn att,
           double el) -> std::pair<uint64_t, uint64_t>
        {
            auto [r, c] = p2b::pixel_at_elevation(p2b::PixelIndex{row}, p2b::PixelIndex{col}, p2b::ImageSize{w, h},
                                                  p2b::PixelToTan{p2t}, to_quat(cam), to_quat(att), p2b::Radians{el});
            return {r.value(), c.value()};
        },
        "row"_a, "col"_a, "width"_a, "height"_a, "pixel_to_tan"_a, "cam_to_body"_a, "attitude"_a,
        "desired_elevation"_a, "Project pixel to target elevation, preserving azimuth.");

    m.def(
        "ned_angle_in_pixels",
        [](Vec3In n1, Vec3In n2, double p2t)
        { return p2b::ned_angle_in_pixels(to_vec3(n1), to_vec3(n2), p2b::PixelToTan{p2t}); }, "ned1"_a, "ned2"_a,
        "pixel_to_tan"_a, "Angular separation between NED vectors as pixel distance.");

    // ============================================================
    //  Batch (vectorized) — zero-copy input via reinterpret_cast
    // ============================================================

    m.def(
        "pixel_to_ned_batch",
        [](U64_1D rows, U64_1D cols, uint64_t w, uint64_t h, double p2t, QuatIn cam, QuatIn att)
        {
            const size_t n = rows.shape(0);
            if (cols.shape(0) != n)
                throw std::invalid_argument("rows and cols must have same length");

            const auto qc = to_quat(cam);
            const auto qa = to_quat(att);
            const p2b::ImageSize img{w, h};
            const p2b::PixelToTan pt{p2t};
            const uint64_t *r = rows.data();
            const uint64_t *c = cols.data();

            auto *out = new double[n * 3];
            for (size_t i = 0; i < n; ++i)
            {
                const auto ned = p2b::pixel_to_ned(p2b::PixelIndex{r[i]}, p2b::PixelIndex{c[i]}, img, pt, qc, qa);
                out[i * 3] = ned.x;
                out[i * 3 + 1] = ned.y;
                out[i * 3 + 2] = ned.z;
            }

            nb::capsule owner(out, [](void *p) noexcept { delete[] static_cast<double *>(p); });
            size_t shape[2] = {n, 3};
            return nb::ndarray<nb::numpy, double>(out, 2, shape, owner);
        },
        "rows"_a, "cols"_a, "width"_a, "height"_a, "pixel_to_tan"_a, "cam_to_body"_a, "attitude"_a,
        "Batch pixels -> NED directions. Returns (N,3) array.");

    m.def(
        "ned_to_pixel_batch",
        [](F64_2D dirs, uint64_t w, uint64_t h, double p2t, QuatIn cam, QuatIn att)
        {
            const size_t n = dirs.shape(0);
            if (dirs.shape(1) != 3)
                throw std::invalid_argument("dirs_ned must have shape (N, 3)");

            const auto qc = to_quat(cam);
            const auto qa = to_quat(att);
            const p2b::ImageSize img{w, h};
            const p2b::PixelToTan pt{p2t};

            // Zero-copy: Vector3 is {double x, y, z} — identical layout to double[3]
            const auto *vecs = reinterpret_cast<const p2b::Vector3 *>(dirs.data());

            auto *out = new uint64_t[n * 2];
            for (size_t i = 0; i < n; ++i)
            {
                auto [row, col] = p2b::ned_to_pixel(vecs[i], img, pt, qc, qa);
                out[i * 2] = row.value();
                out[i * 2 + 1] = col.value();
            }

            nb::capsule owner(out, [](void *p) noexcept { delete[] static_cast<uint64_t *>(p); });
            size_t shape[2] = {n, 2};
            return nb::ndarray<nb::numpy, uint64_t>(out, 2, shape, owner);
        },
        "dirs_ned"_a, "width"_a, "height"_a, "pixel_to_tan"_a, "cam_to_body"_a, "attitude"_a,
        "Batch NED directions -> pixels. Returns (N,2) uint64 array.");

    m.def(
        "pixel_after_rotation_batch",
        [](U64_1D rows, U64_1D cols, uint64_t w, uint64_t h, double p2t, QuatIn cam, QuatIn qo, QuatIn qn, bool rb)
        {
            const size_t n = rows.shape(0);
            if (cols.shape(0) != n)
                throw std::invalid_argument("rows and cols must have same length");

            const auto qc = to_quat(cam);
            const auto q_old = to_quat(qo);
            const auto q_new = to_quat(qn);
            const p2b::ImageSize img{w, h};
            const p2b::PixelToTan pt{p2t};
            const uint64_t *r = rows.data();
            const uint64_t *c = cols.data();

            auto *out = new uint64_t[n * 2];
            for (size_t i = 0; i < n; ++i)
            {
                auto [rn, cn] = p2b::pixel_after_rotation(p2b::PixelIndex{r[i]}, p2b::PixelIndex{c[i]}, img, pt, qc,
                                                          q_old, q_new, rb);
                out[i * 2] = rn.value();
                out[i * 2 + 1] = cn.value();
            }

            nb::capsule owner(out, [](void *p) noexcept { delete[] static_cast<uint64_t *>(p); });
            size_t shape[2] = {n, 2};
            return nb::ndarray<nb::numpy, uint64_t>(out, 2, shape, owner);
        },
        "rows"_a, "cols"_a, "width"_a, "height"_a, "pixel_to_tan"_a, "cam_to_body"_a, "q_old"_a, "q_new"_a,
        "round_back"_a = false, "Batch pixel positions after rotation. Returns (N,2) uint64 array.");

    m.def(
        "warp_image_to_body_batch",
        [](F64_1D wt, F64_1D ht, QuatIn cam)
        {
            const size_t n = wt.shape(0);
            if (ht.shape(0) != n)
                throw std::invalid_argument("w_tans and h_tans must have same length");

            const auto qc = to_quat(cam);
            const double *w = wt.data();
            const double *h = ht.data();

            auto *out = new double[n * 3];
            for (size_t i = 0; i < n; ++i)
            {
                const auto v = p2b::warp_image_to_body(w[i], h[i], qc);
                out[i * 3] = v.x;
                out[i * 3 + 1] = v.y;
                out[i * 3 + 2] = v.z;
            }

            nb::capsule owner(out, [](void *p) noexcept { delete[] static_cast<double *>(p); });
            size_t shape[2] = {n, 3};
            return nb::ndarray<nb::numpy, double>(out, 2, shape, owner);
        },
        "w_tans"_a, "h_tans"_a, "cam_to_body"_a,
        "Batch image tangent pairs -> body-frame directions. Returns (N,3) array.");
}
