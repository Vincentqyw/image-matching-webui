/**
 * @copyright 2019 Xoan Iago Suarez Canosa. All rights reserved.
 * Constact: iago.suarez.canosa@alumnos.eth.es
 * Software developed in the PhD: Augmented Reality for Urban Environments
 */
#ifndef LINE_EXPERIMENTS_INCLUDE_LINES_SALIENTSEGMENTSDETECTOR_H_
#define LINE_EXPERIMENTS_INCLUDE_LINES_SALIENTSEGMENTSDETECTOR_H_

#include <algorithm>
#include "SegmentsDetector.h"

namespace eth {

class SalientSegmentsDetector : public SegmentsDetector {
 public:
  /**
   * @brief Detects the 2D segments weighted by its importance that are present in the image.
   * @param image The input CV_8UC3 or CV_8UC1 image.
   * @param segments The output segments
   */
  virtual SalientSegments detectSalient(const cv::Mat &image) {
    // Detect segments regularly
    Segments segs = detect(image);

    // Set the salience as the square root of the segment length
    SalientSegments result(segs.size());
    for (int i = 0; i < segs.size(); i++) {
      result[i].segment = segs[i];
      result[i].salience = std::sqrt(math::segLength(segs[i]));
    }

    return result;
  }
};

typedef SMART_PTR(SalientSegmentsDetector) SalientSegmentsDetectorPtr;
}  // namespace eth
#endif //LINE_EXPERIMENTS_INCLUDE_LINES_SALIENTSEGMENTSDETECTOR_H_
