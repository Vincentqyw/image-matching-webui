#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <utils.h>
#include <EDLineDetector.h>
#include <multiscale/MultiOctaveSegmentDetector.h>
#include <LineBandDescriptor.h>
#include <PairwiseLineMatching.h>
#include "multiscale/MultiScaleMatching.h"

# define ASSERT_ALW(x) \
  if(!(x)) throw std::logic_error(std::string("") + __FILE__ + ":" + std::to_string(__LINE__) + " Error: " + #x);

# define ASSERT_ALW_NEAR(a, b, th) ASSERT_ALW(std::abs((a) - (b)) < th)
# define ASSERT_ALW_FLT(a, b) ASSERT_ALW_NEAR(a, b, 0.001)

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
using namespace eth;
namespace py = pybind11;

typedef std::vector<std::vector<std::pair<int, py::array_t<float>>>> PyScaleLines;

inline py::array_t<float> keyline_to_pyarr(const std::vector<float> &s) {
  py::array_t<float> result(s.size());
  for (int i = 0; i < s.size(); i++) result[py::make_tuple(i)] = s[i];
  return result;
}

inline Segment pyarr_to_segment(const py::array_t<float> &pys) {
//  ASSERT_ALW(pys.size() == 4 || pys.size() == 5);
  ASSERT_ALW(pys.size() >= 4);
  return Segment(pys.at(0), pys.at(1), pys.at(2), pys.at(3));
}

inline Segments pyarr_to_segments(const py::array_t<float> &pys) {
//  ASSERT_ALW(pys.size() == 4 || pys.size() == 5);
  ASSERT_ALW(pys.ndim() == 2);
  ASSERT_ALW(pys.shape(1) >= 4);
  Segments result(pys.shape(0));
  for (int i = 0; i < result.size(); i++) {
    result[i] = {pys.at(i, 0), pys.at(i, 1), pys.at(i, 2), pys.at(i, 3)};
  }
  return result;
}

inline ScaleLines scalelines_to_cpp(const py::list &multiscale_segs) {
  PyScaleLines cpp_multiscale_segs = py::cast<PyScaleLines>(multiscale_segs);
  ScaleLines cppMultiscaleSegs(multiscale_segs.size());
  for (int i = 0; i < multiscale_segs.size(); i++) {
    for (const std::pair<int, py::array_t<float>> &kv : cpp_multiscale_segs[i]) {
      ASSERT_ALW(kv.second.size() >= 4);
      cv::line_descriptor::KeyLine kl = keyline_from_seg(pyarr_to_segment(kv.second));
      kl.pt = {0, 0};
      kl.octave = kv.first;
      float s = std::pow(M_SQRT2, kl.octave);
      kl.startPointX *= s;
      kl.startPointY *= s;
      kl.endPointX *= s;
      kl.endPointY *= s;
      kl.lineLength *= s;
      kl.response = kv.second.at(4);
      kl.numOfPixels = kv.second.at(5);
      kl.size = 0;
      cppMultiscaleSegs[i].push_back(kl);
    }
  }
  return cppMultiscaleSegs;
}

inline PyScaleLines scalelines_to_py(const ScaleLines &scaleLines) {
  PyScaleLines result;
  for (const LinesVec &lineVec : scaleLines) {
    result.emplace_back();
    for (const cv::line_descriptor::KeyLine &octaveSeg : lineVec) {
      py::array_t<float> pyseg = keyline_to_pyarr({octaveSeg.sPointInOctaveX,
                                                   octaveSeg.sPointInOctaveY,
                                                   octaveSeg.ePointInOctaveX,
                                                   octaveSeg.ePointInOctaveY,
                                                   octaveSeg.response,
                                                   (float) octaveSeg.numOfPixels
                                                  });
      result.back().emplace_back(octaveSeg.octave, pyseg);
    }
  }
  return result;
}

inline std::vector<std::vector<py::array_t<float>>> descrs_to_py(const std::vector<std::vector<cv::Mat>> &descriptors) {
  std::vector<std::vector<py::array_t<float>>> result(descriptors.size());
  for (int i = 0; i < descriptors.size(); i++) {
    for (cv::Mat d : descriptors[i]) {
      assert(d.rows == 1);
      py::array_t<float> pydescr(d.cols);
      std::memcpy(pydescr.mutable_data(), d.ptr<float>(), d.cols * sizeof(float));
      result[i].push_back(pydescr);
    }
  }
  return result;
}

inline std::vector<std::vector<cv::Mat>> descrs_to_cpp(const std::vector<std::vector<py::array_t<
    float>>> &descriptors) {
  std::vector<std::vector<cv::Mat>> result(descriptors.size());
  for (int i = 0; i < descriptors.size(); i++) {
    for (const py::array_t<float> &d : descriptors[i]) {
      assert(d.ndim() == 1);
      cv::Mat cppdescr(1, d.shape(0), CV_32F, (uchar *) d.data());
      result[i].push_back(cppdescr);
    }
  }
  return result;
}

inline std::vector<py::array_t<float>> descrs_to_py(const std::vector<cv::Mat> &descriptors) {
  std::vector<py::array_t<float>> result(descriptors.size());
  for (int i = 0; i < descriptors.size(); i++) {
    cv::Mat d = descriptors[i];
    result[i] = py::array_t<float>(d.cols);
    std::memcpy(result[i].mutable_data(), d.ptr<float>(), d.cols * sizeof(float));
  }
  return result;
}

py::array_t<float> run_edlines_single_scale(const py::array &img) {

  py::buffer_info info = img.request();
  if (info.format != "B") {
    throw py::type_error("Error: The provided numpy array has the wrong type");
  }

  if (info.shape.size() != 2) {
    throw py::type_error("Error: You should provide a 2 dimensional array.");
  }

  auto *imagePtr = static_cast<uint8_t *>(info.ptr);
  cv::Mat image(info.shape[0], info.shape[1], CV_8UC1, imagePtr);

  // LSD call. Returns [x1,y1,x2,y2,width,p,-log10(NFA)] for each segment
  EDLineDetector edline;
  SalientSegments salientSegs = edline.detectSalient(image);
  // std::cout << "Detected " << salientSegs.size() << " Edlines Segments" << std::endl;

  py::array_t<float> segments({(int) salientSegs.size(), 5});
  for (int i = 0; i < salientSegs.size(); i++) {
    segments[py::make_tuple(i, 0)] = salientSegs[i].segment[0];
    segments[py::make_tuple(i, 1)] = salientSegs[i].segment[1];
    segments[py::make_tuple(i, 2)] = salientSegs[i].segment[2];
    segments[py::make_tuple(i, 3)] = salientSegs[i].segment[3];
    segments[py::make_tuple(i, 4)] = salientSegs[i].salience;
  }
  return segments;
}

py::list run_edlines_multiscale(const py::array &img) {

  py::buffer_info info = img.request();
  if (info.format != "B") {
    throw py::type_error("Error: The provided numpy array has the wrong type");
  }

  if (info.shape.size() != 2) {
    throw py::type_error("Error: You should provide a 2 dimensional array.");
  }

  auto *imagePtr = static_cast<uint8_t *>(info.ptr);
  cv::Mat image(info.shape[0], info.shape[1], CV_8UC1, imagePtr);

  eth::MultiOctaveSegmentDetector detector(std::make_shared<eth::EDLineDetector>());
  eth::ScaleLines scaleLines = detector.octaveKeyLines(image);
  return py::cast(scalelines_to_py(scaleLines));
}

py::list run_merge_multiscale_segs(const std::vector<py::array_t<float>> &multiscale_segments) {

  std::vector<Segments> octaveSegments(multiscale_segments.size());
  std::vector<std::vector<float>> saliencies(multiscale_segments.size());
  std::vector<std::vector<size_t>> nPixels(multiscale_segments.size());
  for (int i = 0; i < multiscale_segments.size(); i++) {
    ASSERT_ALW(multiscale_segments[i].ndim() == 2);
    ASSERT_ALW(multiscale_segments[i].shape(1) == 5);
    const auto &segments = multiscale_segments[i];
    octaveSegments[i] = pyarr_to_segments(segments);
    for (int j = 0; j < segments.shape(0); j++) {
      saliencies[i].push_back(segments.at(j, 4));
      Segment s = octaveSegments[i][j];
      int n = std::ceil(std::max(std::abs(s[2] - s[0]), std::abs(s[3] - s[1])));
      nPixels[i].push_back(n);
    }
  }

  ScaleLines scaleLines = MultiOctaveSegmentDetector::mergeOctaveLines(octaveSegments,
                                                                       saliencies,
                                                                       nPixels);
  return py::cast(scalelines_to_py(scaleLines));
}

py::array_t<float> run_lbd_single_scale(const py::array &img,
                                        const py::array_t<float> &segments,
                                        int numOfBands = 9,
                                        int widthOfBand = 7) {
  ASSERT_ALW(numOfBands > 0);
  ASSERT_ALW(widthOfBand > 0);

  py::buffer_info info = img.request();
  if (info.format != "B") {
    throw py::type_error("Error: The provided numpy array has the wrong type");
  }

  if (info.shape.size() != 2) {
    throw py::type_error("Error: You should provide a 2 dimensional array.");
  }

  auto *imagePtr = static_cast<uint8_t *>(info.ptr);
  cv::Mat image(info.shape[0], info.shape[1], CV_8UC1, imagePtr);

  Segments cppSegs = pyarr_to_segments(segments);
  ScaleLines cppMultiscaleSegs(cppSegs.size());

  for (int i = 0; i < cppSegs.size(); i++) {
    cv::line_descriptor::KeyLine kl = keyline_from_seg(cppSegs[i]);
    kl.pt = {0, 0};
    kl.octave = 0;
    kl.response = segments.shape(1) > 4 ? segments.at(i, 4) : 0;
    kl.numOfPixels = segments.shape(1) > 5 ?
                     segments.at(i, 5) :
                     std::ceil(std::max(std::abs(kl.endPointX - kl.startPointX),
                                        std::abs(kl.endPointY - kl.startPointY)));
    kl.size = 0;
    cppMultiscaleSegs[i].push_back(kl);
  }

  eth::LineBandDescriptor lineDesc(numOfBands, widthOfBand);
  std::vector<std::vector<cv::Mat>> descriptors;
  lineDesc.compute(image, cppMultiscaleSegs, descriptors);

  std::vector<cv::Mat> flattened(descriptors.size());
  for (int i = 0; i < descriptors.size(); i++) {
    ASSERT_ALW(descriptors[i].size() == 1);
    flattened[i] = descriptors[i][0];
  }

  return py::cast(descrs_to_py(flattened));
}

py::list run_lbd_multiscale(const py::array &img,
                            const py::list &multiscale_segs,
                            int numOfBands = 9,
                            int widthOfBand = 7) {
  ASSERT_ALW(numOfBands > 0);
  ASSERT_ALW(widthOfBand > 0);

  py::buffer_info info = img.request();
  if (info.format != "B") {
    throw py::type_error("Error: The provided numpy array has the wrong type");
  }

  if (info.shape.size() != 2) {
    throw py::type_error("Error: You should provide a 2 dimensional array.");
  }

  auto *imagePtr = static_cast<uint8_t *>(info.ptr);
  cv::Mat image(info.shape[0], info.shape[1], CV_8UC1, imagePtr);

  ScaleLines cppMultiscaleSegs = scalelines_to_cpp(multiscale_segs);

  eth::LineBandDescriptor lineDesc(numOfBands, widthOfBand);
  std::vector<std::vector<cv::Mat>> descriptors;
  lineDesc.compute(image, cppMultiscaleSegs, descriptors);
  return py::cast(descrs_to_py(descriptors));
}

py::list run_lbd_multiscale_pyr(const std::vector<py::array> &pyr,
                                const py::list &multiscale_segs,
                                int numOfBands = 9,
                                int widthOfBand = 7) {
  ASSERT_ALW(numOfBands > 0);
  ASSERT_ALW(widthOfBand > 0);

  if (pyr.empty()) {
    throw py::type_error("Error: You should provide at least one image in the pyramid.");
  }

  std::vector<cv::Mat> cppPyr;
  for (const py::array &pyr_img : pyr) {
    py::buffer_info info = pyr_img.request();
    if (info.format != "B") {
      throw py::type_error("Error: The provided numpy array has the wrong type");
    }

    if (info.shape.size() != 2) {
      throw py::type_error("Error: You should provide a 2 dimensional array.");
    }

    auto *imagePtr = static_cast<uint8_t *>(info.ptr);
    cppPyr.emplace_back(info.shape[0], info.shape[1], CV_8UC1, imagePtr);
  }

  ScaleLines cppMultiscaleSegs = scalelines_to_cpp(multiscale_segs);

  eth::LineBandDescriptor lineDesc(numOfBands, widthOfBand);
  std::vector<std::vector<cv::Mat>> descriptors;

  auto gradientExtractor = std::make_shared<eth::StateOctaveKeyLineDetector>(nullptr);
  auto pyramidInfo = std::make_shared<eth::MultiOctaveSegmentDetector>(gradientExtractor);
  pyramidInfo->octaveKeyLines(cppPyr);

  lineDesc.compute(cppPyr[0], cppMultiscaleSegs, descriptors, pyramidInfo);
  return py::cast(descrs_to_py(descriptors));
}

py::list run_lbd_matching_multiscale(const py::list &segs1, const py::list &segs2,
                                     const py::list &descrs1, const py::list &descrs2) {
  std::vector<std::pair<uint32_t, uint32_t>> matches;

  PairwiseLineMatching lineMatch;
  ScaleLines cppMultiscaleSegs1, cppMultiscaleSeg2;

  ScaleLines cppSegs1 = scalelines_to_cpp(segs1), cppSegs2 = scalelines_to_cpp(segs2);
  const auto &tmp1 = py::cast<std::vector<std::vector<py::array_t<float>>> >(descrs1);
  const auto &tmp2 = py::cast<std::vector<std::vector<py::array_t<float>>> >(descrs2);
  std::vector<std::vector<cv::Mat>> cppDescrs1 = descrs_to_cpp(tmp1),
      cppDescrs2 = descrs_to_cpp(tmp2);

  cv::Mat_<double> desDisMat = eth::MultiScaleMatching::bruteForceMatching(
      cppDescrs1, cppDescrs2, cv::NORM_L2);

  lineMatch.matchLines(cppSegs1, cppSegs2, cppDescrs1, cppDescrs2, matches);

  std::vector<std::tuple<uint32_t, uint32_t, double>> results(matches.size());
  for (int i = 0; i < matches.size(); i++) {
    results[i] = {matches[i].first, matches[i].second, -desDisMat(matches[i].first, matches[i].second)};
  }

  return py::cast(results);
}

PYBIND11_MODULE(pytlbd, m) {
  m.doc() = R"pbdoc(
        Python transparent bindings for LBD (Line Band Descriptor)
        -----------------------

        .. currentmodule:: pytlbd

        .. autosummary::
           :toctree: _generate

           lsd
    )pbdoc";

  m.def("edlines_single_scale", &run_edlines_single_scale, R"pbdoc(
        Computes Lilian Zhang implementation of EDLines without multi-scale behaviour.
    )pbdoc");

  m.def("edlines_multiscale", &run_edlines_multiscale, R"pbdoc(
        Computes Lilian Zhang implementation of EDLines in multiple scales.
    )pbdoc");

  m.def("merge_multiscale_segs", &run_merge_multiscale_segs, R"pbdoc(
        Groups the same segment detected in several scales.
    )pbdoc");

  m.def("lbd_single_scale", &run_lbd_single_scale, R"pbdoc(
        Computes Line Band Descriptor (LBD) in the detected segments.
    )pbdoc",
        py::arg("img"),
        py::arg("segs"),
        py::arg("numOfBands") = 9,
        py::arg("widthOfBand") = 7);

  m.def("lbd_multiscale", &run_lbd_multiscale, R"pbdoc(
        Computes Line Band Descriptor (LBD) in the detected multi-scale segments.
        The descriptor contains 8m floating point values, where m is the number of bands.
        For each band, the first 4 values are the mean gradients and the last 4 are the s.t.d. of the gradients.
    )pbdoc",
        py::arg("img"),
        py::arg("multiscale_segs"),
        py::arg("numOfBands") = 9,
        py::arg("widthOfBand") = 7
  );

  m.def("lbd_multiscale_pyr", &run_lbd_multiscale_pyr, R"pbdoc(
        Computes Line Band Descriptor (LBD) in the detected multi-scale segments.
        The descriptor contains 8m floating point values, where m is the number of bands.
        For each band, the first 4 values are the mean gradients and the last 4 are the s.t.d. of the gradients.
    )pbdoc",
        py::arg("img"),
        py::arg("multiscale_segs"),
        py::arg("numOfBands") = 9,
        py::arg("widthOfBand") = 7
  );

  m.def("lbd_matching_multiscale", &run_lbd_matching_multiscale, R"pbdoc(
        Computes the matching between two sets of multi-scale segments.
    )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
