#ifndef PYTLBD_SRC_INCLUDE_UTILS_H_
#define PYTLBD_SRC_INCLUDE_UTILS_H_

#include <iostream>
#include <memory>
#include <opencv2/line_descriptor.hpp>

#define UPM_DBG_ASSERT assert
#define UPM_DBG_ASSERT_VECTOR_IDX(x, y)
#define UPM_NOT_IMPLEMENTED_EXCEPTION \
    std::cerr << "Error: Function Not Implemented" << std::endl; \
    throw std::logic_error("Error: Function Not Implemented");

#define SMART_PTR(x) std::shared_ptr<x>
#define UPM_ABS(x) std::abs(x)

namespace eth {
typedef cv::Vec4f Segment;
typedef std::vector<Segment> Segments;
// Specifies a vector of lines.
typedef std::vector<cv::line_descriptor::KeyLine> LinesVec;
// Each element in ScaleLines is a vector of lines which corresponds the same line detected in different octave images.
typedef std::vector<LinesVec> ScaleLines;
typedef cv::Point2i Pixel;

/**
 * This struct represents a segment weighted by its importance in the image
 */
struct SalientSegment {
  Segment segment;
  double salience;

  SalientSegment() = default;
  SalientSegment(const Segment &segment, double salience) : segment(segment), salience(salience) {}

  inline bool operator<(const SalientSegment &rhs) const {
    if (salience == rhs.salience) {
      float dx1 = segment[0] - segment[2];
      float dx2 = rhs.segment[0] - rhs.segment[2];
      float dy1 = segment[1] - segment[3];
      float dy2 = rhs.segment[1] - rhs.segment[3];
      return std::sqrt(dx1 * dx1 + dy1 * dy1) > std::sqrt(dx2 * dx2 + dy2 * dy2);
    } else {
      return salience > rhs.salience;
    }
  }

  friend std::ostream &operator<<(std::ostream &os, const SalientSegment &segment) {
    os << "segment: " << segment.segment << " salience: " << segment.salience;
    return os;
  }
};

typedef std::vector<SalientSegment> SalientSegments;

namespace math{
/**
 * @brief Calculates the length of a line segment
 * @param s The input segment
 * @return The length of the segment
 */
inline float
segLength(const Segment &s) {
  // The optimal way to do that, is to compute first the differences
  const float dx = s[0] - s[2];
  const float dy = s[1] - s[3];
  // And after that, do the square root, avoiding the use of double
  return std::sqrt(dx * dx + dy * dy);
}


inline cv::Vec3d
segEquation(const Segment &s) {
  cv::Vec3d eq = cv::Vec3d(s[0], s[1], 1).cross(cv::Vec3d(s[2], s[3], 1));
  eq /= std::sqrt(eq[0] * eq[0] + eq[1] * eq[1]);
  return eq;
}
}

inline cv::line_descriptor::KeyLine keyline_from_seg(const cv::Vec4f &seg) {
  cv::line_descriptor::KeyLine kl;
  kl.startPointX = seg[0];
  kl.startPointY = seg[1];
  kl.endPointX = seg[2];
  kl.endPointY = seg[3];
  kl.sPointInOctaveX = seg[0];
  kl.sPointInOctaveY = seg[1];
  kl.ePointInOctaveX = seg[2];
  kl.ePointInOctaveY = seg[3];

  float dx = kl.endPointX - kl.startPointX, dy = kl.endPointY - kl.startPointY;
  kl.angle = atan2(dy, dx);
  kl.class_id = 0;
  kl.octave = 0;
  kl.pt = {0.5f * (kl.endPointX + kl.startPointX), 0.5f * (kl.endPointY + kl.startPointY)};
  kl.size = 0;
  kl.lineLength = sqrt(dx * dx + dy * dy);
  kl.numOfPixels = std::max(std::abs(dx), std::abs(dy));
  return kl;
}

}
#endif //PYTLBD_SRC_INCLUDE_UTILS_H_
