/*
 * PairwiseLineMatching.hh
 *
 *  Created on: 2011-8-19
 *      Author: lz
 */

#ifndef PAIRWISELINEMATCHING_HH_
#define PAIRWISELINEMATCHING_HH_
#include <map>
#include "LineBandDescriptor.h"

namespace eth {
//each node in the graph is a possible line matching pair in the left and right image
struct CompareL {
  bool operator()(const double &lhs, const double &rhs) const { return lhs > rhs; }
};
typedef std::multimap<double, unsigned int, CompareL> EigenMAP;

class PairwiseLineMatching {
 public:
  PairwiseLineMatching(double DescriptorDifThreshold = 0.35, int norm_type = cv::NORM_L2);

  /**
   * Construct the relational graph be- tween two groups of LineVecs and to establish the matching
   * results from this graph. Before that, some pre-processes are intro- duced first to reduce the
   * dimension of the graph matching problem by excluding the clear non-matches.
   * @param linesInLeft The KeyLines in the left image
   * @param linesInRight The KeyLines in the right image
   * @param descriptorsLeft The descriptors of the KeyLines in the left image
   * @param descriptorsRight The descriptors of the KeyLines in the right image
   * @param matchResult The matching results
   */
  void matchLines(ScaleLines &linesInLeft,
                  ScaleLines &linesInRight,
                  std::vector<std::vector<cv::Mat>> &descriptorsLeft,
                  std::vector<std::vector<cv::Mat>> &descriptorsRight,
                  std::vector<std::pair<uint32_t, uint32_t>> &matchResult);

 private:
  /** Compute the approximate global rotation angle between image pair(i.e. the left and right images).
   * As shown in Bin Fan's work "Robust line matching through line-point invariants", this approximate
   * global rotation angle can greatly prune the spurious line correspondences. This is the idea of their
   * fast matching version. Nevertheless, the approaches to estimate the approximate global rotation angle
   * are different. Their is based on the rotation information included in the matched point feature(such as SIFT)
   * while ours is computed from angle histograms of lines in images. Our approach also detect whether there is an
   * appropriate global rotation angle between image pair.
   * step 1: Get the angle histograms of detected lines in the left and right images, respectively;
   * step 2: Search the shift angle between two histograms to minimize their difference. Take this shift angle as
   *         approximate global rotation angle between image pair.
   * input:  detected lines in the left and right images
   * return: the global rotation angle
   */
  double globalRotationOfImagePair(eth::ScaleLines &linesInLeft, eth::ScaleLines &linesInRight);
  /** Build the symmetric non-negative adjacency matrix M, whose nodes are the potential assignments a = (i_l, j_r)
   * and whose weights on edges measure the agreements between pairs of potential assignments. That is where the pairwise
   * constraints are applied(c.f. A spectral technique for correspondence problems using pairwise constraints, M.Leordeanu).
   */
  void buildAdjacencyMatrix(eth::ScaleLines &linesInLeft,
                            eth::ScaleLines &linesInRight,
                            std::vector<std::vector<cv::Mat>> &descriptorsLeft,
                            std::vector<std::vector<cv::Mat>> &descriptorsRight);
  /**
   * Get the final matching from the principal eigenvector.
   */
  void matchingResultFromPrincipalEigenvector(eth::ScaleLines &linesInLeft,
                                              eth::ScaleLines &linesInRight,
                                              std::vector<std::pair<uint32_t, uint32_t>> &matchResult);

  struct Node {
    unsigned int leftLineID; // The index of line in the left image
    unsigned int rightLineID; // The index of line in the right image
  };

  double globalRotationAngle;//the approximate global rotation angle between image pairs

  /* construct a map to store the principal eigenvector and its index.
   * each pair in the map is in this form (eigenvalue, index);
   * Note that, we use eigenvalue as key in the map and index as their value.
   * This is because the map need be sorted by the eigenvalue rather than index
   * for our purpose.
   */
  EigenMAP eigenMap;
  std::vector<Node> nodesList; // Save all the possible matched line pairs
  double minOfEigenVec; // The acceptable minimal value in the principal eigen vector;
  double DescriptorDifThreshold; //0.35, or o.5 are good values for LBD
  int norm_type;  // The type of norm to apply in the descriptor {cv::NORM_L2, cv::NORM_HAMMING}

};
}  // namespace eth
#endif /* PAIRWISELINEMATCHING_HH_ */
