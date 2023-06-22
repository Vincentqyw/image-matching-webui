/*
 * LineBandDescriptor.hh
 *
 *  Created on: Dec 12, 2011
 *      Author: lz
 */

#ifndef LINEDESCRIPTOR_HH_
#define LINEDESCRIPTOR_HH_

#include <map>

#include "multiscale/MultiOctaveSegmentDetector.h"
#include "multiscale/OctaveKeyLineDetector.h"

namespace eth {

/* This class is used to generate the line descriptors from multi-scale images  */
class LineBandDescriptor {
 public:
  enum {
    NearestNeighbor = 0,  // The nearest neighbor is taken as matching
    NNDR = 1  // Nearest/next ratio
  };

  LineBandDescriptor();

  LineBandDescriptor(unsigned int numOfBand, unsigned int widthOfBand);

  int compute(const cv::Mat &image,
              eth::ScaleLines &keyLines,
              std::vector<std::vector<cv::Mat>> &descriptors,
              eth::MultiOctaveSegmentDetectorPtr multiOctaveDetector = nullptr);

  int matchLineByDescriptor(eth::ScaleLines &keyLinesLeft,
                            eth::ScaleLines &keyLinesRight,
                            std::vector<std::vector<cv::Mat>> &descriptorsLeft,
                            std::vector<std::vector<cv::Mat>> &descriptorsRight,
                            std::vector<short> &matchLeft,
                            std::vector<short> &matchRight,
                            int criteria = NNDR);

 private:
  float LowestThreshold;//global threshold for line descriptor distance, default is 0.35
  float NNDRThreshold;//the NNDR threshold for line descriptor distance, default is 0.6

  unsigned int numOfBand_;  // the number of band used to compute line descriptor
  unsigned int widthOfBand_;  // the width of band;
  // The local gaussian coefficient apply to the orthogonal line direction within each band
  std::vector<float> gaussCoefL_;
  // The global gaussian coefficient apply to each Row within line support region
  std::vector<float> gaussCoefG_;
};

}  // namespace eth
#endif /* LINEDESCRIPTOR_HH_ */
