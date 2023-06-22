#include "multiscale/MultiScaleMatching.h"

namespace eth {

inline double hellinger(float mean1, float std1, float mean2, float std2) {
  return std::sqrt(1 - std::sqrt((2 * std1 * std2) / (std1 * std1 + std2 * std2))
      * std::exp(-0.25 * ((mean1 - mean2) * (mean1 - mean2)) / (std1 * std1 + std2 * std2)));
}

cv::Mat_<double> MultiScaleMatching::bruteForceMatching(std::vector<std::vector<cv::Mat>> &descriptorsLeft,
                                                        std::vector<std::vector<cv::Mat>> &descriptorsRight,
                                                        int normType) {
//  double bandDist, bandDist1, bandDist2, bandDist3, bandDist4;
  size_t numLineLeft = descriptorsLeft.size(), numLineRight = descriptorsRight.size();

  // Store the descriptor distance of lines in left and right images.
  cv::Mat_<double> desDisMat(numLineLeft, numLineRight);

  // first compute descriptor distances by BF Matching
  float minDis, dis;
  size_t sameLineSize, sameLineSizeR;
  for (int idL = 0; idL < numLineLeft; idL++) {
    sameLineSize = descriptorsLeft[idL].size();
    for (int idR = 0; idR < numLineRight; idR++) {
      minDis = FLT_MAX;
      sameLineSizeR = descriptorsRight[idR].size();
      for (short lineIDInSameLines = 0; lineIDInSameLines < sameLineSize; lineIDInSameLines++) {
        for (short lineIDInSameLinesR = 0; lineIDInSameLinesR < sameLineSizeR; lineIDInSameLinesR++) {
          // L2 norm of the descriptor
          if (normType == cv::NORM_HAMMING) {
            cv::Mat xorOut;
            cv::bitwise_xor(descriptorsLeft[idL][lineIDInSameLines], descriptorsRight[idR][lineIDInSameLinesR], xorOut);
            dis = cv::norm(xorOut, normType);
          } else {
//            dis = 0;
//            if (descriptorsLeft[idL][lineIDInSameLines].cols == 72) {
//              float * dL = descriptorsLeft[idL][lineIDInSameLines].ptr<float>();
//              float * dR = descriptorsRight[idR][lineIDInSameLinesR].ptr<float>();
//              for (int i = 0; i < 72; i+=8) {
//                bandDist1 = hellinger(dR[i + 0], dR[i + 4], dL[i + 0], dL[i + 4]);
//                bandDist2 = hellinger(dR[i + 1], dR[i + 5], dL[i + 1], dL[i + 5]);
//                bandDist3 = hellinger(dR[i + 2], dR[i + 6], dL[i + 2], dL[i + 6]);
//                bandDist4 = hellinger(dR[i + 3], dR[i + 7], dL[i + 3], dL[i + 7]);
//                bandDist = bandDist1 + bandDist2 + bandDist3 + bandDist4;
//                dis += bandDist;
//              }
//              dis /= 0.425f * 72;
//            }
            dis = cv::norm(descriptorsLeft[idL][lineIDInSameLines] - descriptorsRight[idR][lineIDInSameLinesR],
                           normType);
          }
          if (dis < minDis) {
            minDis = dis;
          }
        }
      }  //end for(short lineIDInSameLines = 0; lineIDInSameLines<sameLineSize; lineIDInSameLines++)
      // Store the smaller distance between the lineVec's
      desDisMat(idL, idR) = minDis;
    }  // end for(int idR=0; idR<rightSize; idR++)
  }  // end for(int idL=0; idL<leftSize; idL++)

  return desDisMat;
}

std::vector<cv::DMatch> MultiScaleMatching::matchesFromDistMatrix(cv::Mat_<double> &desDisMat,
                                                                  int kMatches,
                                                                  float maxDist) {
  // desDisMat should have shape (trainFeatures, queryFeatures), (leftFeatures, rightFeatures)
  std::vector<cv::DMatch> matches;
  double minDistance, _;
  cv::Point minLoc, minLoc2;
  while (true) {
    cv::minMaxLoc(desDisMat, &minDistance, &_, &minLoc);
    if (minDistance > maxDist) {
      break;//the descriptor difference is too large;
    }

    desDisMat(minLoc) = FLT_MAX;
    matches.emplace_back(minLoc.x, minLoc.y, minDistance);
    cv::Mat row = desDisMat.row(minLoc.y);
    for (int i = 0; i < (kMatches - 1); i++) {
      cv::minMaxLoc(row, &minDistance, &_, &minLoc2);
      if (minDistance > maxDist) {
        break;//the descriptor difference is too large;
      }
      desDisMat(minLoc2) = FLT_MAX;
      matches.emplace_back(minLoc2.x, minLoc2.y, minDistance);
    }
    desDisMat.row(minLoc.y).setTo(FLT_MAX);
  }
  return matches;
}
}
