/**
 * @copyright 2020 Xoan Iago Suarez Canosa. All rights reserved.
 * Constact: iago.suarez.canosa@alumnos.eth.es
 * Software developed in the PhD: Augmented Reality for Urban Environments
 */
#ifndef LINE_EXPERIMENTS_INCLUDE_LINES_OCTAVEKEYLINEDETECTOR_H_

#include <utility>
#include <utils.h>
//#include "utils/math/CommonMaths.h"
#include "SegmentsDetector.h"
#include "SalientSegmentsDetector.h"

namespace eth {

class OctaveKeyLineDetector : public SalientSegmentsDetector {
 public:

  virtual bool doesSmooth() const { return true; }

  virtual const cv::Mat &getOctaveImg() const {
    UPM_NOT_IMPLEMENTED_EXCEPTION;
  }
  virtual const cv::Mat &getDxImg() const {
    UPM_NOT_IMPLEMENTED_EXCEPTION;
  }
  virtual const cv::Mat &getDyImg() const {
    UPM_NOT_IMPLEMENTED_EXCEPTION;
  }
  virtual cv::Size getImgSize() const {
    UPM_NOT_IMPLEMENTED_EXCEPTION;
  }

  virtual const std::vector<float> &getSegmentsDirection() const {
    UPM_NOT_IMPLEMENTED_EXCEPTION;
  }

  virtual const std::vector<float> &getSegmentsSalience() const {
    UPM_NOT_IMPLEMENTED_EXCEPTION;
  }

  const Segments &getDetectedSegments() const {
    return lineEndpoints;
  }

  virtual const std::vector<cv::Vec3d> &getLineEquations() const {
    UPM_NOT_IMPLEMENTED_EXCEPTION;
  }

  virtual size_t getNumberOfPixels(size_t segmentId) const {
    assert(segmentId < lineEndpoints.size());
    return math::segLength(lineEndpoints[segmentId]);
  }

 protected:
  //store the line endpoints, [x1,y1,x2,y3]
  Segments lineEndpoints;
};

typedef std::shared_ptr<OctaveKeyLineDetector> OctaveKeyLineDetectorPtr;

}
#define LINE_EXPERIMENTS_INCLUDE_LINES_OCTAVEKEYLINEDETECTOR_H_

#endif //LINE_EXPERIMENTS_INCLUDE_LINES_OCTAVEKEYLINEDETECTOR_H_
