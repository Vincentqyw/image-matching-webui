/**
 * @copyright 2018 Xoan Iago Suarez Canosa. All rights reserved.
 * Constact: iago.suarez.canosa@alumnos.eth.es
 * Software developed in the PhD: Augmented Reality for Urban Environments
 */
#ifndef LINES_SEGMENTSDETECTOR_H_
#define LINES_SEGMENTSDETECTOR_H_

#include <string>
#include "utils.h"

namespace eth {

class SegmentsDetector {
 public:
  /**
   * @brief Detects the 2D segments that are present in the image.
   * @param image The input CV_8UC3 or CV_8UC1 image.
   * @param segments The output segments
   */
  virtual Segments detect(const cv::Mat &image) = 0;  // NOLINT(runtime/references)

  /**
   * @brief Returns the name of the detector
   * @return
   */
  virtual std::string getName() const = 0;

  virtual SMART_PTR(SegmentsDetector) clone() const = 0;
};

typedef SMART_PTR(SegmentsDetector) SegmentsDetectorPtr;
}  // namespace eth
#endif  // LINES_SEGMENTSDETECTOR_H_
