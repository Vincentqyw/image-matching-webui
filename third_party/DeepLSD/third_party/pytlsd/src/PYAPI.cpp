#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "lsd.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
namespace py = pybind11;

void check_img_format(const py::buffer_info& correct_info, const py::buffer_info& info, std::string name=""){
  std::stringstream ss;
  if (info.format != correct_info.format) {
    ss << "Error: " << name << " array has format \"" << info.format
       << "\" but the format should be \"" << correct_info.format << "\"";
    throw py::type_error(ss.str());
  }
  if (info.shape.size() != correct_info.shape.size()) {
    ss << "Error: " << name << " array has " << info.shape.size()
       << "dimensions but should have " << correct_info.shape.size();
    throw py::type_error(ss.str());
  }

  for(int i =0 ; i < info.shape.size() ; i++){
    if (info.shape[i] != correct_info.shape[i]) {
      ss << "Error: " << name << " array has " << info.shape[i] << " elements in dimension " << info.shape.size()
         << " but should have " << correct_info.shape[i];
      throw py::type_error(ss.str());
    }
  }
}

//  return LineSegmentDetection(n_out, img, X, Y, scale, sigma_scale, quant,
//                              ang_th, log_eps, density_th, n_bins,
//                              reg_img, reg_x, reg_y);

// Passing in a generic array
// Passing in an array of doubles
py::array_t<float> run_lsd(const py::array_t<double>& img,
                           double scale=0.8,
                           double sigma_scale=0.6,
                           double density_th=0.0, /* Minimal density of region points in rectangle. */
                           const py::array_t<double>& gradnorm = py::array_t<double>(),
                           const py::array_t<double>& gradangle = py::array_t<double>(),
                           bool grad_nfa = false) {
  double quant = 2.0;       /* Bound to the quantization error on the
                                gradient norm.                                */
  double ang_th = 22.5;     /* Gradient angle tolerance in degrees.           */
  // double log_eps = 0.0;     /* Detection threshold: -log10(NFA) > log_eps     */
  int n_bins = 1024;        /* Number of bins in pseudo-ordering of gradient
                               modulus.                                       */
  double log_eps = 0;

  py::buffer_info info = img.request();
  if (info.format != "d" && info.format != "B" ) {
    throw py::type_error("Error: The provided numpy array has the wrong type");
  }

  double *modgrad_ptr{};
  double *angles_ptr{};
  if (gradnorm.size() != 0 ) {
    py::buffer_info gradnorm_info = gradnorm.request();
    check_img_format(info, gradnorm_info, "Gradnorm");
    modgrad_ptr = static_cast<double *>(gradnorm_info.ptr);
  }

  if (gradangle.size() != 0) {
    py::buffer_info gradangle_info = gradangle.request();
    check_img_format(info, gradangle_info, "Gradangle");
    angles_ptr = static_cast<double *>(gradangle_info.ptr);
  }

  if (info.shape.size() != 2) {
    throw py::type_error("Error: You should provide a 2 dimensional array.");
  }

  double *imagePtr = static_cast<double *>(info.ptr);

  // LSD call. Returns [x1,y1,x2,y2,width,p,-log10(NFA)] for each segment
  int N;
  double *out = LineSegmentDetection(
    &N, imagePtr, info.shape[1], info.shape[0], scale, sigma_scale, quant,
    ang_th, log_eps, density_th, n_bins, grad_nfa, modgrad_ptr, angles_ptr);
  // std::cout << "Detected " << N << " LSD Segments" << std::endl;

  py::array_t<float> segments({N, 5});
  for (int i = 0; i < N; i++) {
    segments[py::make_tuple(i, 0)] = out[7 * i + 0];
    segments[py::make_tuple(i, 1)] = out[7 * i + 1];
    segments[py::make_tuple(i, 2)] = out[7 * i + 2];
    segments[py::make_tuple(i, 3)] = out[7 * i + 3];
    segments[py::make_tuple(i, 4)] = out[7 * i + 5];
    // p:           out[7 * i + 4]);
    // -log10(NFA): out[7 * i + 5]);
  }
  free((void *) out);
  return segments;
}


PYBIND11_MODULE(pytlsd, m) {
    m.doc() = R"pbdoc(
        Python transparent bindings for LSD (Line Segment Detector)
        -----------------------

        .. currentmodule:: pytlsd

        .. autosummary::
           :toctree: _generate

           lsd
    )pbdoc";

    m.def("lsd", &run_lsd, R"pbdoc(
        Computes Line Segment Detection (LSD) in the image.
    )pbdoc",
          py::arg("img"),
          py::arg("scale") = 0.8,
          py::arg("sigma_scale") = 0.6,
          py::arg("density_th") = 0.0,
          py::arg("gradnorm") = py::array(),
          py::arg("gradangle") = py::array(),
          py::arg("grad_nfa") = false);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
