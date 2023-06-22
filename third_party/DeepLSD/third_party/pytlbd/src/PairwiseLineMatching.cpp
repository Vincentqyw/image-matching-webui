/*
 * PairwiseLineMatching.cpp
 *
 *  Created on: 2011-8-19
 *      Author: lz
 */

#include "multiscale/MultiScaleMatching.h"
#include "PairwiseLineMatching.h"

#define TIME_START
#define TIME_STOP

#ifdef WITH_ARPACK
#include <arlsmat.h>
#include <arlssym.h>
#endif

// The maximum number of nodes that we can support
#define MAX_SUPPORTED_NODES 40000
#define Inf       1e10 //Infinity
//the resolution scale of theta histogram, used when compute angle histogram of lines in the image
#define ResolutionScale  20  //10 degree
#define RadsResolutionScale  0.3490658503988659 //10 degree
/*The following two thresholds are used to decide whether the estimated global rotation is acceptable.
 *Some image pairs don't have a stable global rotation angle, e.g. the image pair of the wide baseline
 *non planar scene. */
#define AcceptableAngleHistogramDifference 0.49
#define AcceptableLengthVectorDifference   0.4
// A matching between histogram orientations is considered relevant if the probability
// of its distance is smaller than  StatisticalRelevanceTh
#define StatisticalRelevanceTh   0.02


/*The following four thresholds are used to decide whether a line in the left and a line in the right
 *are the possible matched line pair. If they are, then their differences should be smaller than these
 *thresholds.*/
#define LengthDifThreshold                 4
#define AngleDifferenceThreshold           0.7854//45degree
/*The following four threshold are used to decide whether the two possible matched line pair agree with each other.
 *They reflect the similarity of pairwise geometric information.
 */
#define RelativeAngleDifferenceThreshold   0.7854//45degree
#define IntersectionRationDifThreshold     1
#define ProjectionRationDifThreshold       1


//this is used when get matching result from principal eigen vector
#define WeightOfMeanEigenVec               0.1

namespace eth {
PairwiseLineMatching::PairwiseLineMatching(double DescriptorDifThreshold, int norm_type) :
    DescriptorDifThreshold(DescriptorDifThreshold), norm_type(norm_type) {
}

void PairwiseLineMatching::matchLines(eth::ScaleLines &linesInLeft,
                                      eth::ScaleLines &linesInRight,
                                      std::vector<std::vector<cv::Mat>> &descriptorsLeft,
                                      std::vector<std::vector<cv::Mat>> &descriptorsRight,
                                      std::vector<std::pair<uint32_t, uint32_t>> &matchResult) {
#ifndef NDEBUG
  // Check that all the segments have been described
  assert(linesInLeft.size() == descriptorsLeft.size());
  assert(linesInRight.size() == descriptorsRight.size());
  for (int i = 0; i < descriptorsLeft.size(); i++) {
    assert(linesInLeft[i].size() == descriptorsLeft[i].size());
    for (int j = 0; j < descriptorsLeft[i].size(); j++) {
      assert(!descriptorsLeft[i][j].empty());
    }
  }
  for (int i = 0; i < linesInRight.size(); i++) {
    assert(linesInRight[i].size() == descriptorsRight[i].size());
    for (int j = 0; j < descriptorsRight[i].size(); j++) {
      assert(!descriptorsRight[i][j].empty());
    }
  }
#endif

  //compute the global rotation angle of image pair;
  TIME_START("3.1 ESTIMATE GLOBAL ROTATION");
  globalRotationAngle = globalRotationOfImagePair(linesInLeft, linesInRight);

  TIME_STOP("3.1 ESTIMATE GLOBAL ROTATION");
  TIME_START("3.2 BUILD ADJACENCY MATRIX");
  buildAdjacencyMatrix(linesInLeft, linesInRight, descriptorsLeft, descriptorsRight);

  TIME_STOP("3.2 BUILD ADJACENCY MATRIX");
  TIME_START("3.3 GENERATE MATCHING FROM EIGENVECTOR");
  matchingResultFromPrincipalEigenvector(linesInLeft, linesInRight, matchResult);

  TIME_STOP("3.3 GENERATE MATCHING FROM EIGENVECTOR");
}

//template<int dim>
//cv::Mat plotHistogram(cv::Vec<double, dim> hist) {
//  int hist_w = 512, hist_h = 400;
//  int bin_w = hist_w / (float) dim;
//  cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));
//  for (int i = 0; i < dim; i++) {
//    line(histImage,
//         cv::Point(15 + bin_w * i, hist_h),
//         cv::Point(15 + bin_w * i, hist_h - hist[i] * 400),
//         cv::Scalar(0, 0, 0), bin_w / 2, 8, 0);
//  }
//  return histImage;
//}

inline double normalCDF(double value, double mean, double std) {
//  return 0.5 * std::erfc(-((value - mean) / std) * M_SQRT1_2);
  double result = 0.5 * std::erfc(-((value - mean) / std) * M_SQRT1_2);
  return result;
}

template<int dim>
inline std::pair<double, double> meanStdev(const cv::Vec<double, dim> &vec) {
  double mean = 0, stdev = 0;
  for (int i = 0; i < dim; i++) mean += vec[i];
  mean = mean / dim;

  for (int i = 0; i < dim; i++) stdev += (vec[i] - mean) * (vec[i] - mean);
  stdev = std::sqrt(stdev / dim);
  return {mean, stdev};
}

double PairwiseLineMatching::globalRotationOfImagePair(eth::ScaleLines &linesInLeft, eth::ScaleLines &linesInRight) {
  constexpr double TwoPI = 2 * M_PI;
  // TwoPI / RadsResolutionScale
  constexpr uint32_t dim = TwoPI / RadsResolutionScale; //number of the bins of histogram
  constexpr double angleShift = RadsResolutionScale / 2;//make sure zero is the middle of the interval

  //step 1: compute the angle histogram of lines in the left and right images
  uint32_t index;//index in the histogram
  double direction;
  double rotationAngle = TwoPI;

  cv::Vec<double, dim> angleHistLeft;
  cv::Vec<double, dim> angleHistRight;
  cv::Vec<double, dim> lengthLeft;//lengthLeft[i] store the total line length of all the lines in the ith angle bin.
  cv::Vec<double, dim> lengthRight;
  angleHistLeft = 0;
  angleHistRight = 0;
  lengthLeft = 0;
  lengthRight = 0;

  for (auto &lineVec : linesInLeft) {
    // Compute the line direction in degrees
    direction = lineVec[0].angle + M_PI + angleShift;
    direction = direction < TwoPI ? direction : (direction - TwoPI);
    // Calculate the index where it lies
    index = floor(direction / RadsResolutionScale);
    angleHistLeft[index]++;
    lengthLeft[index] += lineVec[0].lineLength;
  }
  for (auto &lineVec : linesInRight) {
    // Compute the line direction in degrees
    direction = lineVec[0].angle + M_PI + angleShift;
    direction = direction < TwoPI ? direction : (direction - TwoPI);
    // Calculate the index where it lies
    index = floor(direction / RadsResolutionScale);
    angleHistRight[index]++;
    lengthRight[index] += lineVec[0].lineLength;
  }

  angleHistLeft = (1 / cv::norm(angleHistLeft)) * angleHistLeft;
  angleHistRight = (1 / cv::norm(angleHistRight)) * angleHistRight;
  lengthLeft = (1 / cv::norm(lengthLeft)) * lengthLeft;
  lengthRight = (1 / cv::norm(lengthRight)) * lengthRight;

  //step 2: find shift to decide the approximate global rotation
  cv::Vec<double, dim> difVec; //the difference vector between left histogram and shifted right histogram
  cv::Vec<double, dim> shiftDifVec;
  double minDif = 10; //the minimal angle histogram difference
  double secondMinDif = 10; //the second minimal histogram difference
  size_t minShift; //the shift of right angle histogram when minimal difference achieved

  cv::Vec<double, dim> lengthDifVec;//the length difference vector between left and right
  cv::Vec<double, dim> shiftLengthDifVec;
  double minLenDif = 10;//the minimal length difference
  double secondMinLenDif = 10;//the second minimal length difference
  size_t minLenShift;//the shift of right length vector when minimal length difference achieved

  double normOfVec;
  for (size_t shift = 0; shift < dim; shift++) {
    for (size_t j = 0; j < dim; j++) {
      index = j + shift;
      index = index < dim ? index : (index - dim);
      difVec[j] = std::abs(angleHistLeft[j] - angleHistRight[index]);
      lengthDifVec[j] = std::abs(lengthLeft[j] - lengthRight[index]);
    }
    //find the minShift and secondMinShift for angle histogram
    normOfVec = cv::norm(difVec);
    shiftDifVec[shift] = normOfVec;
    if (normOfVec < secondMinDif) {
      if (normOfVec < minDif) {
        secondMinDif = minDif;
        minDif = normOfVec;
        minShift = shift;
      } else {
        secondMinDif = normOfVec;
      }
    }
    // Find the minLenShift and secondMinLenShift of length vector
    normOfVec = cv::norm(lengthDifVec);
    shiftLengthDifVec[shift] = normOfVec;
    if (normOfVec < secondMinLenDif) {
      if (normOfVec < minLenDif) {
        secondMinLenDif = minLenDif;
        minLenDif = normOfVec;
        minLenShift = shift;
      } else {
        secondMinLenDif = normOfVec;
      }
    }
  }

//  cv::imshow("angleHistLeft", plotHistogram(angleHistLeft));
//  cv::imshow("angleHistRight", plotHistogram(angleHistRight));
//  cv::imshow("lengthLeft", plotHistogram(lengthLeft));
//  cv::imshow("lengthRight", plotHistogram(lengthRight));
//  cv::imshow("Angle Diff", plotHistogram(shiftDifVec * 0.2));
//  cv::imshow("Length Diff", plotHistogram(shiftLengthDifVec * 0.2));

  auto anglesStats = meanStdev(shiftDifVec);
  double angleProb = normalCDF(minDif, anglesStats.first, anglesStats.second);
  auto lengthsStats = meanStdev(shiftLengthDifVec);
  double lengthProb = normalCDF(minLenDif, lengthsStats.first, lengthsStats.second);

//  std::cout << "Length min value: " << minLenDif << ", mean: " << lengthsStats.first
//            << ", Angle std: " << lengthsStats.second << ", prob: " << lengthProb << std::endl;
//  std::cout << "Angle min value: " << minDif << ", mean: " << anglesStats.first
//            << ", Angle std: " << anglesStats.second << ", prob: " << angleProb << std::endl;

//  cv::waitKey();

  bool statisticallyRelevant = angleProb < StatisticalRelevanceTh || lengthProb < StatisticalRelevanceTh;
//  std::cout << "--> Statistically relevant: " << (statisticallyRelevant ? "TRUE" : "FALSE") << std::endl;
//  std::cout << "--> Same optimal rot: " << (minLenShift == minShift ? "TRUE" : "FALSE") << std::endl;

  //first check whether there exist an approximate global rotation angle between image pair
  if (minDif < AcceptableAngleHistogramDifference &&
      minLenDif < AcceptableLengthVectorDifference &&
      minLenShift == minShift &&
      statisticallyRelevant) {
    rotationAngle = minShift * ResolutionScale;
    if (rotationAngle > 90 && 360 - rotationAngle > 90) {
      //In most case we believe the rotation angle between two image pairs should belong to [-Pi/2, Pi/2]
      rotationAngle = rotationAngle - 180;
    }
    rotationAngle = rotationAngle * M_PI / 180;
  }

  return rotationAngle;
}

inline void segmentEquation(const cv::line_descriptor::KeyLine &kl, double &a, double &b, double &c) {
  a = kl.endPointY - kl.startPointY;  // disY
  b = kl.startPointX - kl.endPointX;  // -disX
  c = (0 - b * kl.startPointY) - a * kl.startPointX;  // disX*sy - disY*sx
}

/**
 * @brief project the end points of kl onto the line2 (a, b, c) and compute their distances to line2
 * @param kl
 * @param a
 * @param b
 * @param c
 * @param length
 * @return
 */
inline double endpointsToLineDist(const cv::line_descriptor::KeyLine &kl, double a, double b, double c, double length) {
  double disS = fabs(a * kl.startPointX + b * kl.startPointY + c) / length;
  double disE = fabs(a * kl.endPointX + b * kl.endPointY + c) / length;
  return (disS + disE) / kl.lineLength;
}

void PairwiseLineMatching::buildAdjacencyMatrix(eth::ScaleLines &linesInLeft,
                                                eth::ScaleLines &linesInRight,
                                                std::vector<std::vector<cv::Mat>> &descriptorsLeft,
                                                std::vector<std::vector<cv::Mat>> &descriptorsRight) {
  assert(!descriptorsLeft.empty() && !descriptorsLeft[0].empty());
  assert(descriptorsLeft[0][0].type() == CV_32FC1 && descriptorsLeft[0][0].rows == 1);
  assert(!descriptorsRight.empty() && !descriptorsRight[0].empty());
  assert(descriptorsRight[0][0].type() == CV_32FC1 && descriptorsRight[0][0].rows == 1);

  /* first step, find nodes which are possible correspondent lines in the left and right images according to
   * their direction, gray value  and gradient magnitude. */
  TIME_START("3.2.1 COMPUTE DESCRIPTOR DIST-MATRIX");
  cv::Mat_<double> desDisMat = eth::MultiScaleMatching::bruteForceMatching(
      descriptorsLeft, descriptorsRight, norm_type);
  TIME_STOP("3.2.1 COMPUTE DESCRIPTOR DIST-MATRIX");

  // Store the descriptor distance of lines in left and right images.
  TIME_START("3.2.2 DISTANCE-RADIUS AND LENGTH FILTERING");
  nodesList.clear();
  double TwoPI = 2 * M_PI;
  size_t numLineLeft = linesInLeft.size(), numLineRight = linesInRight.size();
  double angleDif, lengthDif;
  for (size_t i = 0; i < numLineLeft; i++) {
    for (size_t j = 0; j < numLineRight; j++) {
      if (desDisMat(i, j) > DescriptorDifThreshold) {
        continue;//the descriptor difference is too large;
      }

      if (globalRotationAngle < TwoPI) {
        // There exist a global rotation angle between two image
        lengthDif = fabs(linesInLeft[i][0].lineLength - linesInRight[j][0].lineLength)
            / MIN(linesInLeft[i][0].lineLength, linesInRight[j][0].lineLength);
        if (lengthDif > LengthDifThreshold) {
          continue;  // The length difference is too large;
        }

        angleDif = fabs(linesInLeft[i][0].angle + globalRotationAngle - linesInRight[j][0].angle);
        if (fabs(TwoPI - angleDif) > AngleDifferenceThreshold && angleDif > AngleDifferenceThreshold) {
          continue;  // The angle difference is too large;
        }

      } else {
        // There doesn't exist a global rotation angle between two image, so the angle difference test is canceled.
        lengthDif = fabs(linesInLeft[i][0].lineLength - linesInRight[j][0].lineLength)
            / MIN(linesInLeft[i][0].lineLength, linesInRight[j][0].lineLength);
        if (lengthDif > LengthDifThreshold) {
          continue;  // The length difference is too large;
        }
      }
      // Line i in left image and line j in right image pass the test, (i,j) is a possible matched line pair.
      Node node;
      node.leftLineID = i;
      node.rightLineID = j;
      nodesList.push_back(node);
    } // end inner loop
  }

  if (nodesList.size() > MAX_SUPPORTED_NODES) {
    std::cerr << "Too many nodes in the graph. The process will probably break due the lack of memory so"
                 " lets reduce the number to " << MAX_SUPPORTED_NODES << std::endl;
    nodesList.resize(MAX_SUPPORTED_NODES);
  }

  TIME_STOP("3.2.2 DISTANCE-RADIUS AND LENGTH FILTERING");
  TIME_START("3.2.3 GENERATE SPARSE SVD PROBLEM");

  /* Second step, build the adjacency matrix which reflect the geometric constraints between nodes.
   * The matrix is stored in the Compressed Sparse Column(CSC) format. */
  eigenMap.clear();
  size_t dim = nodesList.size();// Dimension of the problem.
  int nnz = 0;// Number of nonzero elements in adjacenceMat.
  /*adjacenceVec only store the lower part of the adjacency matrix which is a symmetric matrix.
   *                    | 0  1  0  2  0 |
   *                    | 1  0  3  0  1 |
   *eg:  adjMatrix =    | 0  3  0  2  0 |
   *                    | 2  0  2  0  3 |
   *                    | 0  1  0  3  0 |
   *     adjacenceVec = [0,1,0,2,0,0,3,0,1,0,2,0,0,3,0]
   */
  //	Matrix<double> testMat(dim,dim);
  //	testMat.SetZero();

#ifdef  WITH_ARPACK
  cv::Mat_<double> adjacenceVec = cv::Mat_<double>::zeros(1, dim * (dim + 1) / 2);
  //std::cout << "Computing LBD matching with ARPACK adjacenceVec size: " << adjacenceVec.size() << std::endl;
#define SET_NODE_SIMILARITY(i, j, similarity) adjacenceVec((2 * dim - (j) - 1) * (j) / 2 + (i)) = (similarity)
#else
  cv::Mat_<double> A = cv::Mat_<double>::zeros(dim, dim);
  //std::cout << "Computing LBD matching with OpenCV. A size: " << A.size() << std::endl;
#define SET_NODE_SIMILARITY(i, j, similarity) A((i), (j)) = (similarity); A((j), (i)) = (similarity);
#endif

  /*In order to save computational time, the following variables are used to store
   *the pairwise geometric information which has been computed and will be reused many times
   *latter. The reduction of computational time is at the expenses of memory consumption.
   */
  //flag to show whether the ith pair of left image has already been computed.
  cv::Mat_<uint8_t> bComputedLeft = cv::Mat_<uint8_t>::zeros(numLineLeft, numLineLeft);
  //the ratio of intersection point and the line in the left pair
  cv::Mat_<double> intersecRatioLeft(numLineLeft, numLineLeft);
  //the point to line distance divided by the projected length of line in the left pair.
  cv::Mat_<double> projRatioLeft(numLineLeft, numLineLeft);

  // flag to show whether the ith pair of right image has already been computed.
  cv::Mat_<uint8_t> bComputedRight = cv::Mat_<uint8_t>::zeros(numLineRight, numLineRight);
  // the ratio of intersection point and the line in the right pair
  cv::Mat_<double> intersecRatioRight(numLineRight, numLineRight);
  // the point to line distance divided by the projected length of line in the right pair.
  cv::Mat_<double> projRatioRight(numLineRight, numLineRight);

  size_t idLeft1, idLeft2;//the id of lines in the left pair
  size_t idRight1, idRight2;//the id of lines in the right pair
  double relativeAngleLeft, relativeAngleRight;//the relative angle of each line pair

  double iRatio1L, iRatio1R, iRatio2L, iRatio2R;
  double pRatio1L, pRatio1R, pRatio2L, pRatio2R;

  double relativeAngleDif, iRatioDif, pRatioDif;

  double interSectionPointX, interSectionPointY;
  double a1, a2, b1, b2, c1, c2;//line1: a1 x + b1 y + c1 =0; line2: a2 x + b2 y + c2=0
  double a1b2_a2b1;//a1b2-a2b1
  double length1, length2, len;
  double disX, disY;
  double similarity;
  for (size_t j = 0; j < dim; j++) {//column
    idLeft1 = nodesList[j].leftLineID;
    idRight1 = nodesList[j].rightLineID;
    for (size_t i = j + 1; i < dim; i++) {//row
      idLeft2 = nodesList[i].leftLineID;
      idRight2 = nodesList[i].rightLineID;
      if ((idLeft1 == idLeft2) || (idRight1 == idRight2)) {
        continue;//not satisfy the one to one match condition
      }
      //first compute the relative angle between left pair and right pair.
      relativeAngleLeft = linesInLeft[idLeft1][0].angle - linesInLeft[idLeft2][0].angle;
      relativeAngleLeft = (relativeAngleLeft < M_PI) ? relativeAngleLeft : (relativeAngleLeft - TwoPI);
      relativeAngleLeft = (relativeAngleLeft > (-M_PI)) ? relativeAngleLeft : (relativeAngleLeft + TwoPI);
      relativeAngleRight = linesInRight[idRight1][0].angle - linesInRight[idRight2][0].angle;
      relativeAngleRight = (relativeAngleRight < M_PI) ? relativeAngleRight : (relativeAngleRight - TwoPI);
      relativeAngleRight = (relativeAngleRight > (-M_PI)) ? relativeAngleRight : (relativeAngleRight + TwoPI);
      relativeAngleDif = fabs(relativeAngleLeft - relativeAngleRight);
      if ((TwoPI - relativeAngleDif) > RelativeAngleDifferenceThreshold
          && relativeAngleDif > RelativeAngleDifferenceThreshold) {
        continue;//the relative angle difference is too large;
      } else if ((TwoPI - relativeAngleDif) < RelativeAngleDifferenceThreshold) {
        relativeAngleDif = TwoPI - relativeAngleDif;
      }

      //at last, check the intersect point ratio and point to line distance ratio
      //check whether the geometric information of pairs (idLeft1,idLeft2) and (idRight1,idRight2) have already been computed.
      if (!bComputedLeft(idLeft1, idLeft2)) {//have not been computed yet
        /*compute the intersection point of segment i and j.
         *a1x + b1y + c1 = 0 and a2x + b2y + c2 = 0.
         *x = (c2b1 - c1b2)/(a1b2 - a2b1) and
         *y = (c1a2 - c2a1)/(a1b2 - a2b1)*/
        segmentEquation(linesInLeft[idLeft1][0], a1, b1, c1);
        length1 = linesInLeft[idLeft1][0].lineLength;
        segmentEquation(linesInLeft[idLeft2][0], a2, b2, c2);
        length2 = linesInLeft[idLeft2][0].lineLength;

        a1b2_a2b1 = a1 * b2 - a2 * b1;
        if (fabs(a1b2_a2b1) < 0.001) {//two lines are almost parallel
          iRatio1L = Inf;
          iRatio2L = Inf;
        } else {
          interSectionPointX = (c2 * b1 - c1 * b2) / a1b2_a2b1;
          interSectionPointY = (c1 * a2 - c2 * a1) / a1b2_a2b1;
          //r1 = (s1I*s1e1)/(|s1e1|*|s1e1|)
          disX = interSectionPointX - linesInLeft[idLeft1][0].startPointX;
          disY = interSectionPointY - linesInLeft[idLeft1][0].startPointY;
          len = disY * a1 - disX * b1;
          iRatio1L = len / (length1 * length1);
          //r2 = (s2I*s2e2)/(|s2e2|*|s2e2|)
          disX = interSectionPointX - linesInLeft[idLeft2][0].startPointX;
          disY = interSectionPointY - linesInLeft[idLeft2][0].startPointY;
          len = disY * a2 - disX * b2;
          iRatio2L = len / (length2 * length2);
        }
        intersecRatioLeft(idLeft1, idLeft2) = iRatio1L;
        intersecRatioLeft(idLeft2, idLeft1) = iRatio2L;//line order changed

        /*project the end points of line1 onto line2 and compute their distances to line2 */
        projRatioLeft(idLeft1, idLeft2) = endpointsToLineDist(linesInLeft[idLeft1][0], a2, b2, c2, length2);
        /*project the end points of line2 onto line1 and compute their distances to line1 */
        projRatioLeft(idLeft2, idLeft1) = endpointsToLineDist(linesInLeft[idLeft2][0], a1, b1, c1, length1);

        //mark them as computed
        bComputedLeft(idLeft1, idLeft2) = true;
        bComputedLeft(idLeft2, idLeft1) = true;
      }

      if (!bComputedRight(idRight1, idRight2)) {//have not been computed yet
        segmentEquation(linesInRight[idRight1][0], a1, b1, c1);
        length1 = linesInRight[idRight1][0].lineLength;

        segmentEquation(linesInRight[idRight2][0], a2, b2, c2);
        length2 = linesInRight[idRight2][0].lineLength;

        a1b2_a2b1 = a1 * b2 - a2 * b1;
        if (fabs(a1b2_a2b1) < 0.001) {//two lines are almost parallel
          iRatio1R = Inf;
          iRatio2R = Inf;
        } else {
          interSectionPointX = (c2 * b1 - c1 * b2) / a1b2_a2b1;
          interSectionPointY = (c1 * a2 - c2 * a1) / a1b2_a2b1;
          //r1 = (s1I*s1e1)/(|s1e1|*|s1e1|)
          disX = interSectionPointX - linesInRight[idRight1][0].startPointX;
          disY = interSectionPointY - linesInRight[idRight1][0].startPointY;
          len = disY * a1 - disX * b1;//because b1=-disX
          iRatio1R = len / (length1 * length1);
          //r2 = (s2I*s2e2)/(|s2e2|*|s2e2|)
          disX = interSectionPointX - linesInRight[idRight2][0].startPointX;
          disY = interSectionPointY - linesInRight[idRight2][0].startPointY;
          len = disY * a2 - disX * b2;//because b2=-disX
          iRatio2R = len / (length2 * length2);
        }
        intersecRatioRight(idRight1, idRight2) = iRatio1R;
        intersecRatioRight(idRight2, idRight1) = iRatio2R;//line order changed

        /*project the end points of line1 onto line2 and compute their distances to line2 */
        projRatioRight(idRight1, idRight2) = endpointsToLineDist(linesInRight[idRight1][0], a2, b2, c2, length2);
        /*project the end points of line2 onto line1 and compute their distances to line1; */
        projRatioRight(idRight2, idRight1) = endpointsToLineDist(linesInRight[idRight2][0], a1, b1, c1, length1);

        //mark them as computed
        bComputedRight(idRight1, idRight2) = true;
        bComputedRight(idRight2, idRight1) = true;
      }

      // Read information from matrix
      iRatio1L = intersecRatioLeft(idLeft1, idLeft2);
      iRatio2L = intersecRatioLeft(idLeft2, idLeft1);
      pRatio1L = projRatioLeft(idLeft1, idLeft2);
      pRatio2L = projRatioLeft(idLeft2, idLeft1);
      iRatio1R = intersecRatioRight(idRight1, idRight2);
      iRatio2R = intersecRatioRight(idRight2, idRight1);
      pRatio1R = projRatioRight(idRight1, idRight2);
      pRatio2R = projRatioRight(idRight2, idRight1);

      // Geometric checks to discard non-valid matches
      pRatioDif = MIN(fabs(pRatio1L - pRatio1R), fabs(pRatio2L - pRatio2R));
      if (pRatioDif > ProjectionRationDifThreshold) {
        continue;//the projection length ratio difference is too large;
      }
      if ((iRatio1L == Inf) || (iRatio2L == Inf) || (iRatio1R == Inf) || (iRatio2R == Inf)) {
        //don't consider the intersection length ratio
        similarity = 4 - desDisMat(idLeft1, idRight1) / DescriptorDifThreshold
            - desDisMat(idLeft2, idRight2) / DescriptorDifThreshold
            - pRatioDif / ProjectionRationDifThreshold
            - relativeAngleDif / RelativeAngleDifferenceThreshold;
        SET_NODE_SIMILARITY(i, j, similarity);
        nnz++;
      } else {
        iRatioDif = MIN(fabs(iRatio1L - iRatio1R), fabs(iRatio2L - iRatio2R));
        if (iRatioDif > IntersectionRationDifThreshold) {
          continue;//the intersection length ratio difference is too large;
        }
        //now compute the similarity score between two line pairs.
        similarity = 5 - desDisMat(idLeft1, idRight1) / DescriptorDifThreshold
            - desDisMat(idLeft2, idRight2) / DescriptorDifThreshold
            - iRatioDif / IntersectionRationDifThreshold - pRatioDif / ProjectionRationDifThreshold
            - relativeAngleDif / RelativeAngleDifferenceThreshold;
        SET_NODE_SIMILARITY(i, j, similarity);
        nnz++;
      }
    }
  }
  TIME_STOP("3.2.3 GENERATE SPARSE SVD PROBLEM");
  TIME_START("3.2.4 SOLVE SPARSE SVD PROBLEM");

#undef SET_NODE_SIMILARITY
#ifdef WITH_ARPACK
  /*Third step, solve the principal eigenvector of the adjacency matrix using Arpack lib */

  // pointer to an array that stores the nonzero elements of Adjacency matrix.
  std::vector<double> adjacenceMat(nnz, 0);
  // pointer to an array that stores the row indices of the non-zeros in adjacenceMat.
  std::vector<int> irow(nnz);
  // pointer to an array of pointers to the beginning of each column of adjacenceMat.
  std::vector<int> pcol(dim + 1);
  int idOfNNZ = 0;//the order of none zero element
  pcol[0] = 0;
  size_t tempValue;
  for (size_t j = 0; j < dim; j++) {//column
    for (size_t i = j; i < dim; i++) {//row
      tempValue = (2 * dim - j - 1) * j / 2 + i;
      if (adjacenceVec(tempValue) != 0) {
        adjacenceMat[idOfNNZ] = adjacenceVec(tempValue);
        irow[idOfNNZ] = i;
        idOfNNZ++;
      }
    }
    pcol[j + 1] = idOfNNZ;
  }

  ARluSymMatrix<double> arMatrix(dim, nnz, adjacenceMat.data(), irow.data(), pcol.data());
  // Defining what we need: the first eigenvector of arMatrix with largest magnitude.
  ARluSymStdEig<double> dprob(2, arMatrix, "LM");
  // Finding eigenvalues and eigenvectors.
  dprob.FindEigenvectors();

  double meanEigenVec = 0;
  if (dprob.EigenvectorsFound()) {
    double value;
    for (size_t j = 0; j < dim; j++) {
      value = fabs(dprob.Eigenvector(1, j));
      meanEigenVec += value;
      eigenMap.insert(std::make_pair(value, j));
    }
  }

#else
  cv::Mat eigenVals, eigenVecs;
  cv::eigen(A, eigenVals, eigenVecs);
  assert(eigenVals.type() == CV_64FC1);
  assert(eigenVecs.type() == CV_64FC1);

  double meanEigenVec = 0;
  if (eigenVals.at<double>(0, 0) > 1e-5) {
    for (size_t j = 0; j < dim; j++) {
      double value = fabs(eigenVecs.at<double>(0, j));
      meanEigenVec += value;
      eigenMap.insert(std::make_pair(value, j));
    }
  }
#endif

  minOfEigenVec = WeightOfMeanEigenVec * meanEigenVec / dim;
  TIME_STOP("3.2.4 SOLVE SPARSE SVD PROBLEM");

}

void PairwiseLineMatching::matchingResultFromPrincipalEigenvector(
    ScaleLines &linesInLeft,
    ScaleLines &linesInRight,
    std::vector<std::pair<uint32_t, uint32_t>> &matchResult) {
  double TwoPI = 2 * M_PI;
  std::vector<std::pair<uint32_t, uint32_t>> matchRet1;
  std::vector<unsigned int> matchRet2;
  double matchScore1 = 0;
  EigenMAP::iterator iter;
  size_t id, idLeft2, idRight2;
  double sideValueL, sideValueR;
  double pointX, pointY;
  double relativeAngleLeft, relativeAngleRight;//the relative angle of each line pair
  double relativeAngleDif;

  matchResult.clear();
  if (eigenMap.empty()) {
    return;
  }

  /*first try, start from the top element in eigenmap */
  while (true) {
    iter = eigenMap.begin();
    //if the top element in the map has small value, then there is no need to continue find more matching line pairs;
    if (iter->first < minOfEigenVec) {
      break;
    }
    id = iter->second;
    UPM_DBG_ASSERT_VECTOR_IDX(nodesList, id);
    size_t idLeft1 = nodesList[id].leftLineID;
    size_t idRight1 = nodesList[id].rightLineID;
    matchRet1.emplace_back(idLeft1, idRight1);
    matchScore1 += iter->first;
    eigenMap.erase(iter++);
    //remove all potential assignments in conflict with top matched line pair
    double xe_xsLeft = linesInLeft[idLeft1][0].endPointX - linesInLeft[idLeft1][0].startPointX;
    double ye_ysLeft = linesInLeft[idLeft1][0].endPointY - linesInLeft[idLeft1][0].startPointY;
    double xe_xsRight = linesInRight[idRight1][0].endPointX - linesInRight[idRight1][0].startPointX;
    double ye_ysRight = linesInRight[idRight1][0].endPointY - linesInRight[idRight1][0].startPointY;
    double coefLeft = sqrt(xe_xsLeft * xe_xsLeft + ye_ysLeft * ye_ysLeft);
    double coefRight = sqrt(xe_xsRight * xe_xsRight + ye_ysRight * ye_ysRight);
    for (; iter->first >= minOfEigenVec;) {
      id = iter->second;
      idLeft2 = nodesList[id].leftLineID;
      idRight2 = nodesList[id].rightLineID;
      //check one to one match condition
      if ((idLeft1 == idLeft2) || (idRight1 == idRight2)) {
        eigenMap.erase(iter++);
        continue;//not satisfy the one to one match condition
      }
      //check sidedness constraint, the middle point of line2 should lie on the same side of line1.
      //sideValue = (y-ys)*(xe-xs)-(x-xs)*(ye-ys);
      pointX = 0.5 * (linesInLeft[idLeft2][0].startPointX + linesInLeft[idLeft2][0].endPointX);
      pointY = 0.5 * (linesInLeft[idLeft2][0].startPointY + linesInLeft[idLeft2][0].endPointY);
      sideValueL = (pointY - linesInLeft[idLeft1][0].startPointY) * xe_xsLeft
          - (pointX - linesInLeft[idLeft1][0].startPointX) * ye_ysLeft;
      sideValueL = sideValueL / coefLeft;
      pointX = 0.5 * (linesInRight[idRight2][0].startPointX + linesInRight[idRight2][0].endPointX);
      pointY = 0.5 * (linesInRight[idRight2][0].startPointY + linesInRight[idRight2][0].endPointY);
      sideValueR = (pointY - linesInRight[idRight1][0].startPointY) * xe_xsRight
          - (pointX - linesInRight[idRight1][0].startPointX) * ye_ysRight;
      sideValueR = sideValueR / coefRight;
      if (sideValueL * sideValueR < 0 && fabs(sideValueL) > 5
          && fabs(sideValueR) > 5) {//have the different sign, conflict happens.
        eigenMap.erase(iter++);
        continue;
      }
      //check relative angle difference
      relativeAngleLeft = linesInLeft[idLeft1][0].angle - linesInLeft[idLeft2][0].angle;
      relativeAngleLeft = (relativeAngleLeft < M_PI) ? relativeAngleLeft : (relativeAngleLeft - TwoPI);
      relativeAngleLeft = (relativeAngleLeft > (-M_PI)) ? relativeAngleLeft : (relativeAngleLeft + TwoPI);
      relativeAngleRight = linesInRight[idRight1][0].angle - linesInRight[idRight2][0].angle;
      relativeAngleRight = (relativeAngleRight < M_PI) ? relativeAngleRight : (relativeAngleRight - TwoPI);
      relativeAngleRight = (relativeAngleRight > (-M_PI)) ? relativeAngleRight : (relativeAngleRight + TwoPI);
      relativeAngleDif = fabs(relativeAngleLeft - relativeAngleRight);
      if ((TwoPI - relativeAngleDif) > RelativeAngleDifferenceThreshold
          && relativeAngleDif > RelativeAngleDifferenceThreshold) {
        eigenMap.erase(iter++);
        continue;//the relative angle difference is too large;
      }
      iter++;
    }
  }//end while(stillLoop)
  matchResult = matchRet1;
//  std::cout << "matchRet1.size" << matchRet1.size() << ", minOfEigenVec= " << minOfEigenVec << std::endl;
}
}  // namespace eth
