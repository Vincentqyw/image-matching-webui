/*
 * LineBandDescriptor.cpp
 *
 *  Created on: Dec 12, 2011
 *      Author: lz
 */

#include <opencv2/opencv.hpp>
#include <opencv2/line_descriptor.hpp>
#include "LineBandDescriptor.h"

#define SalienceScale 0.9//0.9

//#define DEBUGLinesInOctaveImages
typedef cv::line_descriptor::KeyLine KeyLine;
namespace eth {

LineBandDescriptor::LineBandDescriptor() : numOfBand_(9), widthOfBand_(7) {

  gaussCoefL_.resize(widthOfBand_ * 3);
  double u = (widthOfBand_ * 3 - 1) / 2;
  double sigma = (widthOfBand_ * 2 + 1) / 2;// (widthOfBand_*2+1)/2;
  double invsigma2 = -1 / (2 * sigma * sigma);
  double dis;
  for (int i = 0; i < widthOfBand_ * 3; i++) {
    dis = i - u;
    gaussCoefL_[i] = exp(dis * dis * invsigma2);
  }
  gaussCoefG_.resize(numOfBand_ * widthOfBand_);
  u = (numOfBand_ * widthOfBand_ - 1) / 2;
  sigma = u;
  invsigma2 = -1 / (2 * sigma * sigma);
  for (int i = 0; i < numOfBand_ * widthOfBand_; i++) {
    dis = i - u;
    gaussCoefG_[i] = exp(dis * dis * invsigma2);
  }

  // 2 is used to show recall ratio;
  // 0.2 is used to show scale space results,
  // 0.35 is used when verify geometric constraints.
  LowestThreshold = 0.3;
  NNDRThreshold = 0.6;
}

LineBandDescriptor::LineBandDescriptor(unsigned int numOfBand,
                                       unsigned int widthOfBand) {
  numOfBand_ = numOfBand;
  widthOfBand_ = widthOfBand;
  gaussCoefL_.resize(widthOfBand_ * 3);
  double u = (widthOfBand_ * 3 - 1) / 2;
  double sigma = (widthOfBand_ * 2 + 1) / 2;// (widthOfBand_*2+1)/2;
  double invsigma2 = -1 / (2 * sigma * sigma);
  double dis;
  for (int i = 0; i < widthOfBand_ * 3; i++) {
    dis = i - u;
    gaussCoefL_[i] = exp(dis * dis * invsigma2);
  }
  gaussCoefG_.resize(numOfBand_ * widthOfBand_);
  u = (numOfBand_ * widthOfBand_ - 1) / 2;
  sigma = u;
  invsigma2 = -1 / (2 * sigma * sigma);
  for (int i = 0; i < numOfBand_ * widthOfBand_; i++) {
    dis = i - u;
    gaussCoefG_[i] = exp(dis * dis * invsigma2);
  }
  LowestThreshold = 0.35;
  NNDRThreshold = 0.2;
}

/*The definitions of line descriptor,mean values of {g_dL>0},{g_dL<0},{g_dO>0},{g_dO<0} of each row in band
 *and std values of sum{g_dL>0},sum{g_dL<0},sum{g_dO>0},sum{g_dO<0} of each row in band.
 * With overlap region. */
int LineBandDescriptor::compute(const cv::Mat &image,
                                std::vector<std::vector<KeyLine>> &keyLines,
                                std::vector<std::vector<cv::Mat>> &descriptors,
                                eth::MultiOctaveSegmentDetectorPtr multiOctaveDetector) {
  if (keyLines.empty()) {
    std::cerr << "Input keylines array is empty. Doing nothing." << std::endl;
    return 0;
  }

  if (image.type() != CV_8UC1) {
    std::cerr << "Error: The image should have type CV_8UC1" << std::endl;
    return -1;
  }

  if (multiOctaveDetector == nullptr) {
    auto gradientExtractor = std::make_shared<eth::StateOctaveKeyLineDetector>(nullptr);
    multiOctaveDetector = std::make_shared<eth::MultiOctaveSegmentDetector>(gradientExtractor);
  }
  assert(!multiOctaveDetector->getDetectors().empty());
  bool computePyramid = keyLines.empty();
  computePyramid |= multiOctaveDetector->getDetectors().empty();
  if (!computePyramid) {
    for (int o = 0; o < multiOctaveDetector->getDetectors().size(); o++) {
      cv::Size tmp = multiOctaveDetector->getDetector(o)->getImgSize();
      computePyramid |=  (tmp.width <= 0) || (tmp.height <= 0);
    }
  }

  if (computePyramid) {
    if (keyLines.empty()) {
      keyLines = multiOctaveDetector->octaveKeyLines(image);
    } else {
      multiOctaveDetector->octaveKeyLines(image);
    }
  }

  // Resize the output descriptors structures
  descriptors.resize(keyLines.size());
  for (int i = 0; i < keyLines.size(); i++) descriptors[i].resize(keyLines[i].size());

  constexpr float eps = 1e-5f;
  //the default length of the band is the line length.
  short numOfFinalLine = keyLines.size();
  cv::Vec2f dL; // Line direction cos(dir), sin(dir)
  cv::Vec2f dO; // The clockwise orthogonal vector of line direction.
  short heightOfLSP = widthOfBand_ * numOfBand_;//the height of line support region;
  //each band, we compute the m( pgdL, ngdL,  pgdO, ngdO) and std( pgdL, ngdL,  pgdO, ngdO);
  short descriptorSize = numOfBand_ * 8;
  float pgdLRowSum;//the summation of {g_dL |g_dL>0 } for each row of the region;
  float ngdLRowSum;//the summation of {g_dL |g_dL<0 } for each row of the region;
  float pgdL2RowSum;//the summation of {g_dL^2 |g_dL>0 } for each row of the region;
  float ngdL2RowSum;//the summation of {g_dL^2 |g_dL<0 } for each row of the region;
  float pgdORowSum;//the summation of {g_dO |g_dO>0 } for each row of the region;
  float ngdORowSum;//the summation of {g_dO |g_dO<0 } for each row of the region;
  float pgdO2RowSum;//the summation of {g_dO^2 |g_dO>0 } for each row of the region;
  float ngdO2RowSum;//the summation of {g_dO^2 |g_dO<0 } for each row of the region;

  std::vector<float> pgdLBandSum(numOfBand_, 0);//the summation of {g_dL |g_dL>0 } for each band of the region;
  std::vector<float> ngdLBandSum(numOfBand_, 0);//the summation of {g_dL |g_dL<0 } for each band of the region;
  std::vector<float> pgdL2BandSum(numOfBand_, 0);//the summation of {g_dL^2 |g_dL>0 } for each band of the region;
  std::vector<float> ngdL2BandSum(numOfBand_, 0);//the summation of {g_dL^2 |g_dL<0 } for each band of the region;
  std::vector<float> pgdOBandSum(numOfBand_, 0);//the summation of {g_dO |g_dO>0 } for each band of the region;
  std::vector<float> ngdOBandSum(numOfBand_, 0);//the summation of {g_dO |g_dO<0 } for each band of the region;
  std::vector<float> pgdO2BandSum(numOfBand_, 0);//the summation of {g_dO^2 |g_dO>0 } for each band of the region;
  std::vector<float> ngdO2BandSum(numOfBand_, 0);//the summation of {g_dO^2 |g_dO<0 } for each band of the region;

  short lengthOfLSP; //the length of line support region, varies with lines
  short halfHeight = (heightOfLSP - 1) / 2;
  short halfWidth;
  short bandID;
  float coefInGaussion;
  float lineMiddlePointX, lineMiddlePointY;
  float sCorX, sCorY, sCorX0, sCorY0;
  short tempCor, xCor, yCor;//pixel coordinates in image plane
  short dx, dy;
  float gDL;//store the gradient projection of pixels in support region along dL vector
  float gDO;//store the gradient projection of pixels in support region along dO vector
  short imageWidth, imageHeight, realWidth;
  const short *pdxImg, *pdyImg;
  float *desVec;
  short sameLineSize;
  short octaveCount;
  KeyLine *pSingleLine;
  cv::Size imgSize;
  for (short lineIDInScaleVec = 0; lineIDInScaleVec < numOfFinalLine; lineIDInScaleVec++) {
    sameLineSize = keyLines[lineIDInScaleVec].size();
    for (short lineIDInSameLine = 0; lineIDInSameLine < sameLineSize; lineIDInSameLine++) {
      pSingleLine = &(keyLines[lineIDInScaleVec][lineIDInSameLine]);
      octaveCount = pSingleLine->octave;
      imgSize = multiOctaveDetector->getDetector(octaveCount)->getDxImg().size();
      pdxImg = multiOctaveDetector->getDetector(octaveCount)->getDxImg().ptr<short>();
      pdyImg = multiOctaveDetector->getDetector(octaveCount)->getDyImg().ptr<short>();
      realWidth = multiOctaveDetector->getDetector(octaveCount)->getImgSize().width;
      imageWidth = realWidth - 1;
      imageHeight = multiOctaveDetector->getDetector(octaveCount)->getImgSize().height - 1;
      //initialization
      std::fill(pgdLBandSum.begin(), pgdLBandSum.end(), 0);
      std::fill(ngdLBandSum.begin(), ngdLBandSum.end(), 0);
      std::fill(pgdL2BandSum.begin(), pgdL2BandSum.end(), 0);
      std::fill(ngdL2BandSum.begin(), ngdL2BandSum.end(), 0);
      std::fill(pgdOBandSum.begin(), pgdOBandSum.end(), 0);
      std::fill(ngdOBandSum.begin(), ngdOBandSum.end(), 0);
      std::fill(pgdO2BandSum.begin(), pgdO2BandSum.end(), 0);
      std::fill(ngdO2BandSum.begin(), ngdO2BandSum.end(), 0);
      lengthOfLSP = keyLines[lineIDInScaleVec][lineIDInSameLine].numOfPixels;
      halfWidth = (lengthOfLSP - 1) / 2;
      lineMiddlePointX = 0.5 * (pSingleLine->sPointInOctaveX + pSingleLine->ePointInOctaveX);
      lineMiddlePointY = 0.5 * (pSingleLine->sPointInOctaveY + pSingleLine->ePointInOctaveY);
      /*1.rotate the local coordinate system to the line direction
       *2.compute the gradient projection of pixels in line support region*/
      dL[0] = cos(pSingleLine->angle);
      dL[1] = sin(pSingleLine->angle);
      dO[0] = -dL[1];
      dO[1] = dL[0];
      sCorX0 = -dL[0] * halfWidth + dL[1] * halfHeight + lineMiddlePointX;//hID =0; wID = 0;
      sCorY0 = -dL[1] * halfWidth - dL[0] * halfHeight + lineMiddlePointY;
      for (short hID = 0; hID < heightOfLSP; hID++) {
        //initialization
        sCorX = sCorX0;
        sCorY = sCorY0;

        pgdLRowSum = 0;
        ngdLRowSum = 0;
        pgdORowSum = 0;
        ngdORowSum = 0;

        for (short wID = 0; wID < lengthOfLSP; wID++) {
          tempCor = round(sCorX);
          xCor = (tempCor < 0) ? 0 : (tempCor > imageWidth) ? imageWidth : tempCor;
          tempCor = round(sCorY);
          yCor = (tempCor < 0) ? 0 : (tempCor > imageHeight) ? imageHeight : tempCor;
          /* To achieve rotation invariance, each simple gradient is rotated aligned with
           * the line direction and clockwise orthogonal direction.*/
          assert(yCor >= 0 && yCor < imgSize.height);
          assert(xCor >= 0 && xCor < imgSize.width);
          dx = pdxImg[yCor * realWidth + xCor];
          dy = pdyImg[yCor * realWidth + xCor];
          gDL = dx * dL[0] + dy * dL[1];
          gDO = dx * dO[0] + dy * dO[1];
          if (gDL > 0) {
            pgdLRowSum += gDL;
          } else {
            ngdLRowSum -= gDL;
          }
          if (gDO > 0) {
            pgdORowSum += gDO;
          } else {
            ngdORowSum -= gDO;
          }
          sCorX += dL[0];
          sCorY += dL[1];
          //					gDLMat[hID][wID] = gDL;
        }
        sCorX0 -= dL[1];
        sCorY0 += dL[0];
        coefInGaussion = gaussCoefG_[hID];
        pgdLRowSum = coefInGaussion * pgdLRowSum;
        ngdLRowSum = coefInGaussion * ngdLRowSum;
        pgdL2RowSum = pgdLRowSum * pgdLRowSum;
        ngdL2RowSum = ngdLRowSum * ngdLRowSum;
        pgdORowSum = coefInGaussion * pgdORowSum;
        ngdORowSum = coefInGaussion * ngdORowSum;
        pgdO2RowSum = pgdORowSum * pgdORowSum;
        ngdO2RowSum = ngdORowSum * ngdORowSum;
        //compute {g_dL |g_dL>0 }, {g_dL |g_dL<0 },
        //{g_dO |g_dO>0 }, {g_dO |g_dO<0 } of each band in the line support region
        //first, current row belong to current band;
        bandID = hID / widthOfBand_;
        coefInGaussion = gaussCoefL_[hID % widthOfBand_ + widthOfBand_];
        pgdLBandSum[bandID] += coefInGaussion * pgdLRowSum;
        ngdLBandSum[bandID] += coefInGaussion * ngdLRowSum;
        pgdL2BandSum[bandID] += coefInGaussion * coefInGaussion * pgdL2RowSum;
        ngdL2BandSum[bandID] += coefInGaussion * coefInGaussion * ngdL2RowSum;
        pgdOBandSum[bandID] += coefInGaussion * pgdORowSum;
        ngdOBandSum[bandID] += coefInGaussion * ngdORowSum;
        pgdO2BandSum[bandID] += coefInGaussion * coefInGaussion * pgdO2RowSum;
        ngdO2BandSum[bandID] += coefInGaussion * coefInGaussion * ngdO2RowSum;
        /* In order to reduce boundary effect along the line gradient direction,
         * a row's gradient will contribute not only to its current band, but also
         * to its nearest upper and down band with gaussCoefL_.*/
        bandID--;
        if (bandID >= 0) {//the band above the current band
          coefInGaussion = gaussCoefL_[hID % widthOfBand_ + 2 * widthOfBand_];
          pgdLBandSum[bandID] += coefInGaussion * pgdLRowSum;
          ngdLBandSum[bandID] += coefInGaussion * ngdLRowSum;
          pgdL2BandSum[bandID] += coefInGaussion * coefInGaussion * pgdL2RowSum;
          ngdL2BandSum[bandID] += coefInGaussion * coefInGaussion * ngdL2RowSum;
          pgdOBandSum[bandID] += coefInGaussion * pgdORowSum;
          ngdOBandSum[bandID] += coefInGaussion * ngdORowSum;
          pgdO2BandSum[bandID] += coefInGaussion * coefInGaussion * pgdO2RowSum;
          ngdO2BandSum[bandID] += coefInGaussion * coefInGaussion * ngdO2RowSum;
        }
        bandID = bandID + 2;
        if (bandID < numOfBand_) {//the band below the current band
          coefInGaussion = gaussCoefL_[hID % widthOfBand_];
          pgdLBandSum[bandID] += coefInGaussion * pgdLRowSum;
          ngdLBandSum[bandID] += coefInGaussion * ngdLRowSum;
          pgdL2BandSum[bandID] += coefInGaussion * coefInGaussion * pgdL2RowSum;
          ngdL2BandSum[bandID] += coefInGaussion * coefInGaussion * ngdL2RowSum;
          pgdOBandSum[bandID] += coefInGaussion * pgdORowSum;
          ngdOBandSum[bandID] += coefInGaussion * ngdORowSum;
          pgdO2BandSum[bandID] += coefInGaussion * coefInGaussion * pgdO2RowSum;
          ngdO2BandSum[bandID] += coefInGaussion * coefInGaussion * ngdO2RowSum;
        }
      }

      //construct line descriptor
      descriptors[lineIDInScaleVec][lineIDInSameLine] = cv::Mat::zeros(1, descriptorSize, CV_32FC1);
      desVec = descriptors[lineIDInScaleVec][lineIDInSameLine].ptr<float>();
      short desID;
      /*Note that the first and last bands only have (lengthOfLSP * widthOfBand_ * 2.0) pixels which are counted. */
      float invN2 = 1.0 / (widthOfBand_ * 2.0);
      float invN3 = 1.0 / (widthOfBand_ * 3.0);
      float invN, temp;
      for (bandID = 0; bandID < numOfBand_; bandID++) {
        if (bandID == 0 || bandID == numOfBand_ - 1) {
          invN = invN2;
        } else { invN = invN3; }
        desID = bandID * 8;
        temp = pgdLBandSum[bandID] * invN;
        desVec[desID] = temp;//mean value of pgdL;
        desVec[desID + 4] = sqrt(pgdL2BandSum[bandID] * invN - temp * temp);//std value of pgdL;
        temp = ngdLBandSum[bandID] * invN;
        desVec[desID + 1] = temp;//mean value of ngdL;
        desVec[desID + 5] = sqrt(ngdL2BandSum[bandID] * invN - temp * temp);//std value of ngdL;

        temp = pgdOBandSum[bandID] * invN;
        desVec[desID + 2] = temp;//mean value of pgdO;
        desVec[desID + 6] = sqrt(pgdO2BandSum[bandID] * invN - temp * temp);//std value of pgdO;
        temp = ngdOBandSum[bandID] * invN;
        desVec[desID + 3] = temp;//mean value of ngdO;
        desVec[desID + 7] = sqrt(ngdO2BandSum[bandID] * invN - temp * temp);//std value of ngdO;
      }
      //normalize;
      float tempM, tempS;
      tempM = 0;
      tempS = 0;
      for (short i = 0; i < numOfBand_; i++) {
        tempM += desVec[8 * i + 0] * desVec[8 * i + 0];
        tempM += desVec[8 * i + 1] * desVec[8 * i + 1];
        tempM += desVec[8 * i + 2] * desVec[8 * i + 2];
        tempM += desVec[8 * i + 3] * desVec[8 * i + 3];
        tempS += desVec[8 * i + 4] * desVec[8 * i + 4];
        tempS += desVec[8 * i + 5] * desVec[8 * i + 5];
        tempS += desVec[8 * i + 6] * desVec[8 * i + 6];
        tempS += desVec[8 * i + 7] * desVec[8 * i + 7];
      }
      tempM = 1 / (eps + sqrt(tempM));
      tempS = 1 / (eps + sqrt(tempS));
      for (short i = 0; i < numOfBand_; i++) {
        desVec[8 * i] = desVec[8 * i] * tempM;
        desVec[8 * i + 1] *= tempM;
        desVec[8 * i + 2] *= tempM;
        desVec[8 * i + 3] *= tempM;
        desVec[8 * i + 4] *= tempS;
        desVec[8 * i + 5] *= tempS;
        desVec[8 * i + 6] *= tempS;
        desVec[8 * i + 7] *= tempS;
      }
      /*In order to reduce the influence of non-linear illumination,
       *a threshold is used to limit the value of element in the unit feature
       *vector no larger than this threshold. In Z.Wang's work, a value of 0.4 is found
       *empirically to be a proper threshold.*/
      for (short i = 0; i < descriptorSize; i++) {
        if (desVec[i] > 0.4) {
          desVec[i] = 0.4;
        }
      }
      //re-normalize desVec;
      temp = 0;
      for (short i = 0; i < descriptorSize; i++) {
        temp += desVec[i] * desVec[i];
      }
      temp = 1 / (eps + sqrt(temp));
      for (short i = 0; i < descriptorSize; i++) {
        desVec[i] = desVec[i] * temp;
      }
    }//end for(short lineIDInSameLine = 0; lineIDInSameLine<sameLineSize; lineIDInSameLine++)
  }//end for(short lineIDInScaleVec = 0; lineIDInScaleVec<numOfFinalLine; lineIDInScaleVec++)

  //TODO Should we clean the structures somehow?
  return 0;
}

/*Match line by their descriptors.
 *The function will use opencv FlannBasedMatcher to mathc lines. */
int LineBandDescriptor::matchLineByDescriptor(eth::ScaleLines &keyLinesLeft,
                                              eth::ScaleLines &keyLinesRight,
                                              std::vector<std::vector<cv::Mat>> &descriptorsLeft,
                                              std::vector<std::vector<cv::Mat>> &descriptorsRight,
                                              std::vector<short> &matchLeft,
                                              std::vector<short> &matchRight,
                                              int criteria) {
  assert(!descriptorsLeft.empty() && !descriptorsLeft[0].empty());
  assert(!descriptorsRight.empty() && !descriptorsRight[0].empty());

  int leftSize = keyLinesLeft.size();
  int rightSize = keyLinesRight.size();
  if (leftSize < 1 || rightSize < 1) {
    return -1;
  }

  matchLeft.clear();
  matchRight.clear();

  int desDim = descriptorsLeft[0][0].cols;
  float *desL, *desR, *desMax, *desOld;
  if (criteria == NearestNeighbor) {
    float minDis, dis, temp;
    int corresId;
    for (int idL = 0; idL < leftSize; idL++) {
      short sameLineSize = keyLinesLeft[idL].size();
      minDis = 100;
      for (short lineIDInSameLines = 0; lineIDInSameLines < sameLineSize; lineIDInSameLines++) {
        desOld = descriptorsLeft[idL][lineIDInSameLines].ptr<float>();
        for (int idR = 0; idR < rightSize; idR++) {
          short sameLineSizeR = keyLinesRight[idR].size();
          for (short lineIDInSameLinesR = 0; lineIDInSameLinesR < sameLineSizeR; lineIDInSameLinesR++) {
            desL = desOld;
            desR = descriptorsRight[idR][lineIDInSameLinesR].ptr<float>();
            desMax = desR + desDim;
            dis = 0;
            while (desR < desMax) {
              temp = *(desL++) - *(desR++);
              dis += temp * temp;
            }
            dis = sqrt(dis);
            if (dis < minDis) {
              minDis = dis;
              corresId = idR;
            }
          }
        }//end for(int idR=0; idR<rightSize; idR++)
      }//end for(short lineIDInSameLines = 0; lineIDInSameLines<sameLineSize; lineIDInSameLines++)
      if (minDis < LowestThreshold) {
        matchLeft.push_back(idL);
        matchRight.push_back(corresId);
      }
    }// end for(int idL=0; idL<leftSize; idL++)
  }
  return 0;
}

}  // namespace eth
