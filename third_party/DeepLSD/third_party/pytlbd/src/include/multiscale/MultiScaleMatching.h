/**
 * @copyright 2020 Xoan Iago Suarez Canosa. All rights reserved.
 * Constact: iago.suarez.canosa@alumnos.eth.es
 * Software developed in the PhD: Augmented Reality for Urban Environments
 */
#ifndef LINES_MULTISCALE_MULTISCALEMATCHING_H_
#define LINES_MULTISCALE_MULTISCALEMATCHING_H_
#include <opencv2/opencv.hpp>

namespace eth {
class MultiScaleMatching {
 public:
  /**
 * Computes the matching distance between the descriptors. When matching two sets of LineVecs extracted from
 * an image pairs, the distances between all descriptors of a reference LineVec and a test LineVec are
 * evaluated, and the minimal descriptor distance is used to measure the LineVec appearance similarity s.
 * @param descriptorsLeft Descriptors in left image. A vector where each element
 * contains the descriptor for a multi-scale segment.
 * @param descriptorsRight Descriptors in rihgt image. A vector where each element
 * contains the descriptor for a multi-scale segment.
 * @param normType The type of norm to use.
 * @return A distance matrix of shape (descriptorsLeft, descriptorsRight)
 */
  static cv::Mat_<double> bruteForceMatching(std::vector<std::vector<cv::Mat>> &descriptorsLeft,
                                             std::vector<std::vector<cv::Mat>> &descriptorsRight,
                                             int normType = cv::NORM_L2);

  static std::vector<cv::DMatch> matchesFromDistMatrix(cv::Mat_<double> &desDisMat,
                                                       int kMatches = 1,
                                                       float maxDist = 10000);
};
}

#endif //LINES_MULTISCALE_MULTISCALEMATCHING_H_
