/**
 * @copyright 2020 Xoan Iago Suarez Canosa. All rights reserved.
 * Constact: iago.suarez.canosa@alumnos.eth.es
 * Software developed in the PhD: Augmented Reality for Urban Environments
 */
#ifndef LINES_MULTISCALE_MULTIOCTAVESEGMENTDETECTOR_H_
#define LINES_MULTISCALE_MULTIOCTAVESEGMENTDETECTOR_H_

#include <BresenhamAlgorithm.h>
#include "multiscale/OctaveKeyLineDetector.h"
#include "SalientSegmentsDetector.h"

namespace eth {

/**
 * This class stores the detection results in several octaves of an image. It can
 * be used to convert an SegmentsDetector object without state in a stated object.
 * If innerDetector is nullptr, the class just performs image gradient computation.
 */
class StateOctaveKeyLineDetector : public OctaveKeyLineDetector {
 public:
  explicit StateOctaveKeyLineDetector(const SegmentsDetectorPtr &innerDetector) : innerDetector(innerDetector) {
  }

  Segments detect(const cv::Mat &image) override {
    clear();
    auto salienceDetector = std::dynamic_pointer_cast<SalientSegmentsDetector>(innerDetector);
    if (salienceDetector) {
      SalientSegments salientSegments = salienceDetector->detectSalient(image);
      lineEndpoints.reserve(salientSegments.size());
      lineSalience.reserve(salientSegments.size());
      for (SalientSegment &ss : salientSegments) {
        lineEndpoints.push_back(ss.segment);
        lineSalience.push_back(ss.salience);
      }
    } else {
      if (innerDetector != nullptr) {
        lineEndpoints = innerDetector->detect(image);
        lineSalience = std::vector<float>(lineEndpoints.size(), 1.0f);
      }
    }

    octaveImg = image;
    cv::Sobel(image, dxImg, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
    cv::Sobel(image, dyImg, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);
    imageSize = image.size();

    lineEquations.reserve(lineEndpoints.size());
    lineDirection.reserve(lineEndpoints.size());

    cv::Vec3d eq;
    for (const Segment &s : lineEndpoints) {
      eq = math::segEquation(s);
      lineEquations.push_back(eq);
      std::vector<Pixel> pixels = bresenham(s[0], s[1], s[2], s[3]);
      lineDirection.push_back(generateSegmentDirection(dxImg, dyImg, pixels, eq));
    }
    return lineEndpoints;
  }

  static float generateSegmentDirection(const cv::Mat &dxImg,
                                        const cv::Mat &dyImg,
                                        const std::vector<Pixel> &pixels,
                                        const cv::Vec3d &lineEquation) {
    // First compute the direction of line, make sure that the dark side always be the left side of a line.
    int meanGradientX = 0, meanGradientY = 0;
    int x, y;
    int margin = 2;
    for (int i = margin; i < int(pixels.size()) - margin; i++) {
      UPM_DBG_ASSERT_VECTOR_IDX(pixels, i);
      UPM_DBG_ASSERT(dxImg.size() == dyImg.size());
      x = pixels[i].x, y = pixels[i].y;
      if (y < 0 || y >= dxImg.rows) continue;
      if (x < 0 || x >= dxImg.cols) continue;
      meanGradientX += dxImg.at<short>(y, x);
      meanGradientY += dyImg.at<short>(y, x);
    }
    //TODO This can be done in a more elegant way.
    double dx = fabs(lineEquation[1]);
    double dy = fabs(lineEquation[0]);
    if (meanGradientX == 0 && meanGradientY == 0) {
      //not possible, if happens, it must be a wrong line,
      return 0;
    }
    if (meanGradientX > 0 && meanGradientY >= 0) {
      //first quadrant, and positive direction of X axis.
      return atan2(-dy, dx);
    }
    if (meanGradientX <= 0 && meanGradientY > 0) {
      //second quadrant, and positive direction of Y axis.
      return atan2(dy, dx);
    }
    if (meanGradientX < 0 && meanGradientY <= 0) {
      //third quadrant, and negative direction of X axis.
      return atan2(dy, -dx);
    }
    if (meanGradientX >= 0 && meanGradientY < 0) {
      //fourth quadrant, and negative direction of Y axis.
      return atan2(-dy, -dx);
    }
    return 0;
  }

  std::string getName() const override {
    if (!innerDetector) {
      return "GradientsExtractor";
    }
    return std::string("Stated") + innerDetector->getName();
  };

  const cv::Mat &getOctaveImg() const override { return octaveImg; }
  const cv::Mat &getDxImg() const override { return dxImg; }
  const cv::Mat &getDyImg() const override { return dyImg; }
  inline cv::Size getImgSize() const override { return imageSize; }
  inline const std::vector<float> &getSegmentsDirection() const override { return lineDirection; }
  inline const std::vector<float> &getSegmentsSalience() const override { return lineSalience; }
  inline const std::vector<cv::Vec3d> &getLineEquations() const override { return lineEquations; }
  inline SegmentsDetectorPtr clone() const override {
    SegmentsDetectorPtr inner = innerDetector ? innerDetector->clone() : nullptr;
    return std::make_shared<StateOctaveKeyLineDetector>(inner);
  };
  bool doesSmooth() const override { return innerDetector != nullptr; }
  SegmentsDetectorPtr getInnerDetector() const { return innerDetector; }

  void clear() {
    dxImg = cv::Mat();
    dyImg = cv::Mat();
    lineEndpoints.clear();
    lineDirection.clear();
    lineSalience.clear();
    lineEquations.clear();
    imageSize = {};
  }

 protected:
  cv::Mat octaveImg, dxImg, dyImg;
  SegmentsDetectorPtr innerDetector;
  //store the line direction
  std::vector<float> lineDirection;
  std::vector<float> lineSalience;
  std::vector<cv::Vec3d> lineEquations;
  cv::Size imageSize;
};

class MultiOctaveSegmentDetector {
 public:
  MultiOctaveSegmentDetector(eth::SegmentsDetectorPtr detector, int ksize = 5, int numOfOctaves = 5);

  std::vector<std::vector<cv::line_descriptor::KeyLine>> octaveKeyLines(const cv::Mat &image);

  std::vector<std::vector<cv::line_descriptor::KeyLine>> octaveKeyLines(const std::vector<cv::Mat> &pyramid);


  static ScaleLines mergeOctaveLines(const std::vector<Segments> &octaveSegments,
                                     const std::vector<std::vector<float>> &saliencies,
                                     const std::vector<std::vector<size_t>> &nPixels);

  inline const std::vector<eth::OctaveKeyLineDetectorPtr> &getDetectors() const { return octaveSegDetectors; }

  inline eth::OctaveKeyLineDetectorPtr getDetector(size_t octave) const {
    assert(octave < octaveSegDetectors.size());
    return octaveSegDetectors[octave];
  }

  /**
 * Computes the pyramid of gaussian images that are needed to detect the lines.
 * @param initialImage The level 0 image in the pyramid
 * @param factor The scale factor between octaves.
 * @return A list of images.
 */
  std::vector<cv::Mat> buildGaussianPyramid(const cv::Mat &initialImage, float factor = M_SQRT2);

  inline void setSmoothOctaveImg(bool smooth) { smoothOctaveImg = smooth; }

 private:
  std::vector<eth::OctaveKeyLineDetectorPtr> octaveSegDetectors;

  // The size of Gaussian kernel: ksize X ksize, default value is 5
  int ksize;
  // The number of image octave
  unsigned int numOfOctaves;
  // Whether to smooth or not the image before processing it
  bool smoothOctaveImg;
};

typedef SMART_PTR(MultiOctaveSegmentDetector) MultiOctaveSegmentDetectorPtr;
}

#endif  // LINES_MULTISCALE_MULTIOCTAVESEGMENTDETECTOR_H_
