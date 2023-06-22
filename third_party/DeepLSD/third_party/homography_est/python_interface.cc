#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/stl_bind.h>
#include "hest.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<int>);

PYBIND11_MODULE(homography_est, m) {
    py::bind_vector<std::vector<int>>(m, "VectorInt", py::buffer_protocol());
    py::implicitly_convertible<py::list, std::vector<int>>();

    m.doc() = "Homography estimation from line segments.";
    py::class_<hest::LineSegment>(m, "LineSegment")
        .def(py::init<const Eigen::Vector2d &, const Eigen::Vector2d &>());
    m.def("ransac_line_homography", &hest::ransacLineHomography,
          "Estimate a homography between two sets of line segments.");
    m.def("ransac_point_homography", &hest::ransacPointHomography,
          "Estimate a homography from corresponding points x1=H*x2.");
    m.def("ransac_point_line_homography", &hest::ransacPointLineHomography,
          "Estimate a homography from point and line segments. (x1=H*x2)");
}