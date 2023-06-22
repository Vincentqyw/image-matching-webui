/*
 * EDLineDetector.cpp
 *
 *  Created on: Nov 30, 2011
 *      Author: lz
 */
#include "EDLineDetector.h"
#include <opencv2/opencv.hpp>
#include <BresenhamAlgorithm.h>

#define LBD_Horizontal  1//if |dx|<|dy|;
#define LBD_Vertical    2//if |dy|<=|dx|;
#define LBD_UpDir       1
#define LBD_RightDir    2
#define LBD_DownDir     3
#define LBD_LeftDir     4
#define LBD_TryTime     6
#define LBD_SkipEdgePoint 2

#define LBD_RELATIVE_ERROR_FACTOR   100.0
#define LBD_M_LN10   2.30258509299404568402
#define LBD_log_gamma(x)    ((x)>15.0?log_gamma_windschitl(x):log_gamma_lanczos(x))

//#define DEBUGEdgeDrawing

namespace eth {
/** Compare doubles by relative error.
     The resulting rounding error after floating point computations
     depend on the specific operations done. The same number computed by
     different algorithms could present different rounding errors. For a
     useful comparison, an estimation of the relative rounding error
     should be considered and compared to a factor times EPS. The factor
     should be related to the accumulated rounding error in the chain of
     computation. Here, as a simplification, a fixed factor is used.
 */
static int double_equal(double a, double b) {
  double abs_diff, aa, bb, abs_max;
  /* trivial case */
  if (a == b) return true;
  abs_diff = fabs(a - b);
  aa = fabs(a);
  bb = fabs(b);
  abs_max = aa > bb ? aa : bb;
  /* DBL_MIN is the smallest normalized number, thus, the smallest
    number whose relative error is bounded by DBL_EPSILON. For
    smaller numbers, the same quantization steps as for DBL_MIN
    are used. Then, for smaller numbers, a meaningful "relative"
    error should be computed by dividing the difference by DBL_MIN. */
  if (abs_max < DBL_MIN) abs_max = DBL_MIN;
  /* equal if relative error <= factor x eps */
  return (abs_diff / abs_max) <= (LBD_RELATIVE_ERROR_FACTOR * DBL_EPSILON);
}
/** Computes the natural logarithm of the absolute value of
     the gamma function of x using the Lanczos approximation.
     See http://www.rskey.org/gamma.htm
     The formula used is
     @f[
       \Gamma(x) = \frac{ \sum_{n=0}^{N} q_n x^n }{ \Pi_{n=0}^{N} (x+n) }
                   (x+5.5)^{x+0.5} e^{-(x+5.5)}
     @f]
     so
     @f[
       \log\Gamma(x) = \log\left( \sum_{n=0}^{N} q_n x^n \right)
                       + (x+0.5) \log(x+5.5) - (x+5.5) - \sum_{n=0}^{N} \log(x+n)
     @f]
     and
       q0 = 75122.6331530,
       q1 = 80916.6278952,
       q2 = 36308.2951477,
       q3 = 8687.24529705,
       q4 = 1168.92649479,
       q5 = 83.8676043424,
       q6 = 2.50662827511.
 */
static double log_gamma_lanczos(double x) {
  static double q[7] = {75122.6331530, 80916.6278952, 36308.2951477,
                        8687.24529705, 1168.92649479, 83.8676043424,
                        2.50662827511};
  double a = (x + 0.5) * log(x + 5.5) - (x + 5.5);
  double b = 0.0;
  int n;
  for (n = 0; n < 7; n++) {
    a -= log(x + (double) n);
    b += q[n] * pow(x, (double) n);
  }
  return a + log(b);
}
/** Computes the natural logarithm of the absolute value of
     the gamma function of x using Windschitl method.
     See http://www.rskey.org/gamma.htm
     The formula used is
     @f[
         \Gamma(x) = \sqrt{\frac{2\pi}{x}} \left( \frac{x}{e}
                     \sqrt{ x\sinh(1/x) + \frac{1}{810x^6} } \right)^x
     @f]
     so
     @f[
         \log\Gamma(x) = 0.5\log(2\pi) + (x-0.5)\log(x) - x
                       + 0.5x\log\left( x\sinh(1/x) + \frac{1}{810x^6} \right).
     @f]
     This formula is a good approximation when x > 15.
 */
static double log_gamma_windschitl(double x) {
  return 0.918938533204673 + (x - 0.5) * log(x) - x
      + 0.5 * x * log(x * sinh(1 / x) + 1 / (810.0 * pow(x, 6.0)));
}
/** Computes -log10(NFA).
     NFA stands for Number of False Alarms:
     @f[
         \mathrm{NFA} = NT \cdot B(n,k,p)
     @f]
     - NT       - number of tests
     - B(n,k,p) - tail of binomial distribution with parameters n,k and p:
     @f[
         B(n,k,p) = \sum_{j=k}^n
                    \left(\begin{array}{c}n\\j\end{array}\right)
                    p^{j} (1-p)^{n-j}
     @f]
     The value -log10(NFA) is equivalent but more intuitive than NFA:
     - -1 corresponds to 10 mean false alarms
     -  0 corresponds to 1 mean false alarm
     -  1 corresponds to 0.1 mean false alarms
     -  2 corresponds to 0.01 mean false alarms
     -  ...
     Used this way, the bigger the value, better the detection,
     and a logarithmic scale is used.
     @param n,k,p binomial parameters.
     @param logNT logarithm of Number of Tests
     The computation is based in the gamma function by the following
     relation:
     @f[
         \left(\begin{array}{c}n\\k\end{array}\right)
         = \frac{ \Gamma(n+1) }{ \Gamma(k+1) \cdot \Gamma(n-k+1) }.
     @f]
     We use efficient algorithms to compute the logarithm of
     the gamma function.
     To make the computation faster, not all the sum is computed, part
     of the terms are neglected based on a bound to the error obtained
     (an error of 10% in the result is accepted).
 */
static double nfa(int n, int k, double p, double logNT) {
  double tolerance = 0.1;       /* an error of 10% in the result is accepted */
  double log1term, term, bin_term, mult_term, bin_tail, err, p_term;
  int i;

  /* check parameters */
  if (n < 0 || k < 0 || k > n || p <= 0.0 || p >= 1.0) {
    std::cout << "nfa: wrong n, k or p values." << std::endl;
    exit(0);
  }
  /* trivial cases */
  if (n == 0 || k == 0) return -logNT;
  if (n == k) return -logNT - (double) n * log10(p);

  /* probability term */
  p_term = p / (1.0 - p);

  /* compute the first term of the series */
  /*
    binomial_tail(n,k,p) = sum_{i=k}^n bincoef(n,i) * p^i * (1-p)^{n-i}
    where bincoef(n,i) are the binomial coefficients.
    But
      bincoef(n,k) = gamma(n+1) / ( gamma(k+1) * gamma(n-k+1) ).
    We use this to compute the first term. Actually the log of it.
   */
  log1term = LBD_log_gamma((double) n + 1.0) - LBD_log_gamma((double) k + 1.0)
      - LBD_log_gamma((double) (n - k) + 1.0)
      + (double) k * log(p) + (double) (n - k) * log(1.0 - p);
  term = exp(log1term);

  /* in some cases no more computations are needed */
  if (double_equal(term, 0.0)) {  /* the first term is almost zero */
    if ((double) k > (double) n * p)     /* at begin or end of the tail?  */
      return -log1term / LBD_M_LN10 - logNT;  /* end: use just the first term  */
    else
      return -logNT;                      /* begin: the tail is roughly 1  */
  }

  /* compute more terms if needed */
  bin_tail = term;
  for (i = k + 1; i <= n; i++) {
    /*    As
        term_i = bincoef(n,i) * p^i * (1-p)^(n-i)
      and
        bincoef(n,i)/bincoef(n,i-1) = n-i+1 / i,
      then,
        term_i / term_i-1 = (n-i+1)/i * p/(1-p)
      and
        term_i = term_i-1 * (n-i+1)/i * p/(1-p).
      p/(1-p) is computed only once and stored in 'p_term'.
     */
    bin_term = (double) (n - i + 1) / (double) i;
    mult_term = bin_term * p_term;
    term *= mult_term;
    bin_tail += term;
    if (bin_term < 1.0) {
      /* When bin_term<1 then mult_term_j<mult_term_i for j>i.
        Then, the error on the binomial tail when truncated at
        the i term can be bounded by a geometric series of form
        term_i * sum mult_term_i^j.                            */
      err = term * ((1.0 - pow(mult_term, (double) (n - i + 1))) /
          (1.0 - mult_term) - 1.0);
      /* One wants an error at most of tolerance*final_result, or:
        tolerance * abs(-log10(bin_tail)-logNT).
        Now, the error that can be accepted on bin_tail is
        given by tolerance*final_result divided by the derivative
        of -log10(x) when x=bin_tail. that is:
        tolerance * abs(-log10(bin_tail)-logNT) / (1/bin_tail)
        Finally, we truncate the tail if the error is less than:
        tolerance * abs(-log10(bin_tail)-logNT) * bin_tail        */
      if (err < tolerance * fabs(-log10(bin_tail) - logNT) * bin_tail) break;
    }
  }
  return -log10(bin_tail) - logNT;
}

EDLineDetector::EDLineDetector() {
  //	cout<<"Call EDLineDetector constructor function"<<endl;
  //set parameters for line segment detection
  ksize_ = 5;
  sigma_ = 1.0;
  gradienThreshold_ = 25;
  anchorThreshold_ = 2;//8
  scanIntervals_ = 2;//2
  minLineLen_ = 15;
  lineFitErrThreshold_ = 1.4;
  InitEDLine_();
}

EDLineDetector::EDLineDetector(EDLineParam param) {
  //set parameters for line segment detection
  ksize_ = param.ksize;
  sigma_ = param.sigma;
  gradienThreshold_ = param.gradientThreshold;
  anchorThreshold_ = param.anchorThreshold;
  scanIntervals_ = param.scanIntervals;
  minLineLen_ = param.minLineLen;
  lineFitErrThreshold_ = param.lineFitErrThreshold;
  InitEDLine_();
}

void EDLineDetector::InitEDLine_() {
  bValidate_ = true;
  fitMatT = cv::Mat_<float>::zeros(2, minLineLen_);
  fitVec = cv::Mat_<float>::zeros(minLineLen_, 1);
  fitMatT.row(1).setTo(1);
  dxImg_ = cv::Mat(1, 1, CV_16SC1);
  dyImg_ = cv::Mat(1, 1, CV_16SC1);
  gImgWO_ = cv::Mat(1, 1, CV_8SC1);
  pFirstPartEdgeX_ = NULL;
  pFirstPartEdgeY_ = NULL;
  pFirstPartEdgeS_ = NULL;
  pSecondPartEdgeX_ = NULL;
  pSecondPartEdgeY_ = NULL;
  pSecondPartEdgeS_ = NULL;
  pAnchorX_ = NULL;
  pAnchorY_ = NULL;
}

EDLineDetector::~EDLineDetector() {
  //	cout<<"Call EDLineDetector destructor function"<<endl;
  if (!dxImg_.empty()) {
    dxImg_.release();
    dyImg_.release();
    gImgWO_.release();
  }
  if (pFirstPartEdgeX_ != NULL) {
    delete[] pFirstPartEdgeX_;
    delete[] pFirstPartEdgeY_;
    delete[] pSecondPartEdgeX_;
    delete[] pSecondPartEdgeY_;
    delete[] pAnchorX_;
    delete[] pAnchorY_;
  }
  if (pFirstPartEdgeS_ != NULL) {
    delete[] pFirstPartEdgeS_;
    delete[] pSecondPartEdgeS_;
  }
}

int EDLineDetector::EdgeDrawing(cv::Mat &image, EdgeChains &edgeChains, bool smoothed) {
  if (image.type() != CV_8UC1) {
    std::cout << "EDLineDetector->EdgeDrawing() error: image type should be CV_8UC1" << std::endl;
    return -1;
  }

  imageWidth = image.cols;
  imageHeight = image.rows;
  unsigned int pixelNum = imageWidth * imageHeight;

  cv::Mat cvImage;
  if (!smoothed) {//input image hasn't been smoothed.
    cv::GaussianBlur(image, cvImage, cv::Size(ksize_, ksize_), sigma_);
  } else {
    cvImage = image;
  }

  unsigned int edgePixelArraySize = pixelNum / 5;
  unsigned int maxNumOfEdge = edgePixelArraySize / 20;
  //compute dx, dy images
  if (gImg_.cols != imageWidth || gImg_.rows != imageHeight) {
    if (!dxImg_.empty()) {
      dxImg_.release();
      dyImg_.release();
      gImgWO_.release();
    }
    if (pFirstPartEdgeX_ != NULL) {
      delete[] pFirstPartEdgeX_;
      delete[] pFirstPartEdgeY_;
      delete[] pSecondPartEdgeX_;
      delete[] pSecondPartEdgeY_;
      delete[] pFirstPartEdgeS_;
      delete[] pSecondPartEdgeS_;
      delete[] pAnchorX_;
      delete[] pAnchorY_;
    }

    gImgWO_ = cv::Mat(imageHeight, imageWidth, CV_8SC1);
    gImg_.create(imageHeight, imageWidth);
    dirImg_.create(imageHeight, imageWidth);
    edgeImage_.create(imageHeight, imageWidth);
    pFirstPartEdgeX_ = new unsigned int[edgePixelArraySize];
    pFirstPartEdgeY_ = new unsigned int[edgePixelArraySize];
    pSecondPartEdgeX_ = new unsigned int[edgePixelArraySize];
    pSecondPartEdgeY_ = new unsigned int[edgePixelArraySize];
    pFirstPartEdgeS_ = new unsigned int[maxNumOfEdge];
    pSecondPartEdgeS_ = new unsigned int[maxNumOfEdge];
    pAnchorX_ = new unsigned int[edgePixelArraySize];
    pAnchorY_ = new unsigned int[edgePixelArraySize];
  }
  octaveImg = image;
  cv::Sobel(cvImage, dxImg_, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
  cv::Sobel(cvImage, dyImg_, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);

  //compute gradient and direction images
  short *pdxImg = dxImg_.ptr<short>();
  short *pdyImg = dyImg_.ptr<short>();
  unsigned char *pgImgWO = gImgWO_.data;
  unsigned char *pgImg = gImg_.data;
  unsigned char *pdirImg = dirImg_.data;
  short dxABS, dyABS, dxAdy;
  unsigned char temp;
  for (int i = 0; i < pixelNum; i++) {
    //compute gradient image
    //		pgImg[i] = sqrt(pdxImg[i]*pdxImg[i]+pdyImg[i]*pdyImg[i]);
    dxABS = abs(*(pdxImg++));
    dyABS = abs(*(pdyImg++));
    dxAdy = dxABS + dyABS;
    temp = (unsigned char) (dxAdy / 4);
    *(pgImgWO++) = temp;
    if (dxAdy < gradienThreshold_) {//detect possible edge areas
      *(pgImg++) = 0;
    } else {
      *(pgImg++) = temp;//G = (|dx|+|dy|)/4; Scale the value to [0,255].
    }
    //compute direction image
    if (dxABS < dyABS) {
      *(pdirImg++) = LBD_Horizontal;
    } else {
      *(pdirImg++) = LBD_Vertical;
    }
  }

  pdxImg = dxImg_.ptr<short>();
  pdyImg = dyImg_.ptr<short>();
  pgImg = gImg_.data;
  pdirImg = dirImg_.data;

  //extract the anchors in the gradient image, store into a vector
  memset(pAnchorX_, 0, edgePixelArraySize * sizeof(unsigned int));//initialization
  memset(pAnchorY_, 0, edgePixelArraySize * sizeof(unsigned int));
  unsigned int anchorsSize = 0;
  int indexInArray;
  unsigned char gValue1, gValue2, gValue3;
  for (unsigned int w = 1; w < imageWidth - 1; w = w + scanIntervals_) {
    for (unsigned int h = 1; h < imageHeight - 1; h = h + scanIntervals_) {
      indexInArray = h * imageWidth + w;
      gValue1 = pdirImg[indexInArray];
      if (gValue1 == LBD_Horizontal) {//if the direction of pixel is horizontal, then compare with up and down
        gValue2 = pgImg[indexInArray];
        if (gValue2 >= pgImg[indexInArray - imageWidth] + anchorThreshold_
            && gValue2 >= pgImg[indexInArray + imageWidth] + anchorThreshold_) {// (w,h) is accepted as an anchor
          pAnchorX_[anchorsSize] = w;
          pAnchorY_[anchorsSize++] = h;
        }
      } else if (gValue1 == LBD_Vertical) {//it is vertical edge, should be compared with left and right
        gValue2 = pgImg[indexInArray];
        if (gValue2 >= pgImg[indexInArray - 1] + anchorThreshold_
            && gValue2 >= pgImg[indexInArray + 1] + anchorThreshold_) {// (w,h) is accepted as an anchor
          pAnchorX_[anchorsSize] = w;
          pAnchorY_[anchorsSize++] = h;
        }
      }
    }
  }
  if (anchorsSize > edgePixelArraySize) {
    std::cout << "anchor size is larger than its maximal size. anchorsSize=" << anchorsSize
              << ", maximal size = " << edgePixelArraySize << std::endl;
    return -1;
  }

  //link the anchors by smart routing
  edgeImage_.setTo(0);
  unsigned char *pEdgeImg = edgeImage_.data;
  memset(pFirstPartEdgeX_, 0, edgePixelArraySize * sizeof(unsigned int));//initialization
  memset(pFirstPartEdgeY_, 0, edgePixelArraySize * sizeof(unsigned int));
  memset(pSecondPartEdgeX_, 0, edgePixelArraySize * sizeof(unsigned int));
  memset(pSecondPartEdgeY_, 0, edgePixelArraySize * sizeof(unsigned int));
  memset(pFirstPartEdgeS_, 0, maxNumOfEdge * sizeof(unsigned int));
  memset(pSecondPartEdgeS_, 0, maxNumOfEdge * sizeof(unsigned int));
  unsigned int offsetPFirst = 0, offsetPSecond = 0;
  unsigned int offsetPS = 0;

  unsigned int x, y;
  unsigned int lastX, lastY;
  unsigned char lastDirection;//up = 1, right = 2, down = 3, left = 4;
  unsigned char shouldGoDirection;//up = 1, right = 2, down = 3, left = 4;
  int edgeLenFirst, edgeLenSecond;
  for (unsigned int i = 0; i < anchorsSize; i++) {
    x = pAnchorX_[i];
    y = pAnchorY_[i];
    indexInArray = y * imageWidth + x;
    if (pEdgeImg[indexInArray]) {//if anchor i is already been an edge pixel.
      continue;
    }
    /*The walk stops under 3 conditions:
     * 1. We move out of the edge areas, i.e., the thresholded gradient value
     *    of the current pixel is 0.
     * 2. The current direction of the edge changes, i.e., from horizontal
     *    to vertical or vice versa.?? (This is turned out not correct. From the online edge draw demo
     *    we can figure out that authors don't implement this rule either because their extracted edge
     *    chain could be a circle which means pixel directions would definitely be different
     *    in somewhere on the chain.)
     * 3. We encounter a previously detected edge pixel. */
    pFirstPartEdgeS_[offsetPS] = offsetPFirst;
    if (pdirImg[indexInArray]
        == LBD_Horizontal) {//if the direction of this pixel is horizontal, then go left and right.
      //fist go right, pixel direction may be different during linking.
      lastDirection = LBD_RightDir;
      while (pgImg[indexInArray] > 0 && !pEdgeImg[indexInArray]) {
        pEdgeImg[indexInArray] = 1;        // Mark this pixel as an edge pixel
        pFirstPartEdgeX_[offsetPFirst] = x;
        pFirstPartEdgeY_[offsetPFirst++] = y;
        shouldGoDirection = 0;//unknown
        if (pdirImg[indexInArray] == LBD_Horizontal) {//should go left or right
          if (lastDirection == LBD_UpDir || lastDirection == LBD_DownDir) {//change the pixel direction now
            if (x > lastX) {//should go right
              shouldGoDirection = LBD_RightDir;
            } else {//should go left
              shouldGoDirection = LBD_LeftDir;
            }
          }
          lastX = x;
          lastY = y;
          if (lastDirection == LBD_RightDir || shouldGoDirection == LBD_RightDir) {//go right
            if (x == imageWidth - 1 || y == 0 || y == imageHeight - 1) {//reach the image border
              break;
            }
            // Look at 3 neighbors to the right and pick the one with the max. gradient value
            gValue1 = pgImg[indexInArray - imageWidth + 1];
            gValue2 = pgImg[indexInArray + 1];
            gValue3 = pgImg[indexInArray + imageWidth + 1];
            if (gValue1 >= gValue2 && gValue1 >= gValue3) {//up-right
              x = x + 1;
              y = y - 1;
            } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {//down-right
              x = x + 1;
              y = y + 1;
            } else {//straight-right
              x = x + 1;
            }
            lastDirection = LBD_RightDir;
          } else if (lastDirection == LBD_LeftDir || shouldGoDirection == LBD_LeftDir) {//go left
            if (x == 0 || y == 0 || y == imageHeight - 1) {//reach the image border
              break;
            }
            // Look at 3 neighbors to the left and pick the one with the max. gradient value
            gValue1 = pgImg[indexInArray - imageWidth - 1];
            gValue2 = pgImg[indexInArray - 1];
            gValue3 = pgImg[indexInArray + imageWidth - 1];
            if (gValue1 >= gValue2 && gValue1 >= gValue3) {//up-left
              x = x - 1;
              y = y - 1;
            } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {//down-left
              x = x - 1;
              y = y + 1;
            } else {//straight-left
              x = x - 1;
            }
            lastDirection = LBD_LeftDir;
          }
        } else {//should go up or down.
          if (lastDirection == LBD_RightDir || lastDirection == LBD_LeftDir) {//change the pixel direction now
            if (y > lastY) {//should go down
              shouldGoDirection = LBD_DownDir;
            } else {//should go up
              shouldGoDirection = LBD_UpDir;
            }
          }
          lastX = x;
          lastY = y;
          if (lastDirection == LBD_DownDir || shouldGoDirection == LBD_DownDir) {//go down
            if (x == 0 || x == imageWidth - 1 || y == imageHeight - 1) {//reach the image border
              break;
            }
            // Look at 3 neighbors to the down and pick the one with the max. gradient value
            gValue1 = pgImg[indexInArray + imageWidth + 1];
            gValue2 = pgImg[indexInArray + imageWidth];
            gValue3 = pgImg[indexInArray + imageWidth - 1];
            if (gValue1 >= gValue2 && gValue1 >= gValue3) {//down-right
              x = x + 1;
              y = y + 1;
            } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {//down-left
              x = x - 1;
              y = y + 1;
            } else {//straight-down
              y = y + 1;
            }
            lastDirection = LBD_DownDir;
          } else if (lastDirection == LBD_UpDir || shouldGoDirection == LBD_UpDir) {//go up
            if (x == 0 || x == imageWidth - 1 || y == 0) {//reach the image border
              break;
            }
            // Look at 3 neighbors to the up and pick the one with the max. gradient value
            gValue1 = pgImg[indexInArray - imageWidth + 1];
            gValue2 = pgImg[indexInArray - imageWidth];
            gValue3 = pgImg[indexInArray - imageWidth - 1];
            if (gValue1 >= gValue2 && gValue1 >= gValue3) {//up-right
              x = x + 1;
              y = y - 1;
            } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {//up-left
              x = x - 1;
              y = y - 1;
            } else {//straight-up
              y = y - 1;
            }
            lastDirection = LBD_UpDir;
          }
        }
        indexInArray = y * imageWidth + x;
      }//end while go right
      //then go left, pixel direction may be different during linking.
      x = pAnchorX_[i];
      y = pAnchorY_[i];
      indexInArray = y * imageWidth + x;
      pEdgeImg[indexInArray] = 0;//mark the anchor point be a non-edge pixel and
      lastDirection = LBD_LeftDir;
      pSecondPartEdgeS_[offsetPS] = offsetPSecond;
      while (pgImg[indexInArray] > 0 && !pEdgeImg[indexInArray]) {
        pEdgeImg[indexInArray] = 1;        // Mark this pixel as an edge pixel
        pSecondPartEdgeX_[offsetPSecond] = x;
        pSecondPartEdgeY_[offsetPSecond++] = y;
        shouldGoDirection = 0;//unknown
        if (pdirImg[indexInArray] == LBD_Horizontal) {//should go left or right
          if (lastDirection == LBD_UpDir || lastDirection == LBD_DownDir) {//change the pixel direction now
            if (x > lastX) {//should go right
              shouldGoDirection = LBD_RightDir;
            } else {//should go left
              shouldGoDirection = LBD_LeftDir;
            }
          }
          lastX = x;
          lastY = y;
          if (lastDirection == LBD_RightDir || shouldGoDirection == LBD_RightDir) {//go right
            if (x == imageWidth - 1 || y == 0 || y == imageHeight - 1) {//reach the image border
              break;
            }
            // Look at 3 neighbors to the right and pick the one with the max. gradient value
            gValue1 = pgImg[indexInArray - imageWidth + 1];
            gValue2 = pgImg[indexInArray + 1];
            gValue3 = pgImg[indexInArray + imageWidth + 1];
            if (gValue1 >= gValue2 && gValue1 >= gValue3) {//up-right
              x = x + 1;
              y = y - 1;
            } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {//down-right
              x = x + 1;
              y = y + 1;
            } else {//straight-right
              x = x + 1;
            }
            lastDirection = LBD_RightDir;
          } else if (lastDirection == LBD_LeftDir || shouldGoDirection == LBD_LeftDir) {//go left
            if (x == 0 || y == 0 || y == imageHeight - 1) {//reach the image border
              break;
            }
            // Look at 3 neighbors to the left and pick the one with the max. gradient value
            gValue1 = pgImg[indexInArray - imageWidth - 1];
            gValue2 = pgImg[indexInArray - 1];
            gValue3 = pgImg[indexInArray + imageWidth - 1];
            if (gValue1 >= gValue2 && gValue1 >= gValue3) {//up-left
              x = x - 1;
              y = y - 1;
            } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {//down-left
              x = x - 1;
              y = y + 1;
            } else {//straight-left
              x = x - 1;
            }
            lastDirection = LBD_LeftDir;
          }
        } else {//should go up or down.
          if (lastDirection == LBD_RightDir || lastDirection == LBD_LeftDir) {//change the pixel direction now
            if (y > lastY) {//should go down
              shouldGoDirection = LBD_DownDir;
            } else {//should go up
              shouldGoDirection = LBD_UpDir;
            }
          }
          lastX = x;
          lastY = y;
          if (lastDirection == LBD_DownDir || shouldGoDirection == LBD_DownDir) {//go down
            if (x == 0 || x == imageWidth - 1 || y == imageHeight - 1) {//reach the image border
              break;
            }
            // Look at 3 neighbors to the down and pick the one with the max. gradient value
            gValue1 = pgImg[indexInArray + imageWidth + 1];
            gValue2 = pgImg[indexInArray + imageWidth];
            gValue3 = pgImg[indexInArray + imageWidth - 1];
            if (gValue1 >= gValue2 && gValue1 >= gValue3) {//down-right
              x = x + 1;
              y = y + 1;
            } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {//down-left
              x = x - 1;
              y = y + 1;
            } else {//straight-down
              y = y + 1;
            }
            lastDirection = LBD_DownDir;
          } else if (lastDirection == LBD_UpDir || shouldGoDirection == LBD_UpDir) {//go up
            if (x == 0 || x == imageWidth - 1 || y == 0) {//reach the image border
              break;
            }
            // Look at 3 neighbors to the up and pick the one with the max. gradient value
            gValue1 = pgImg[indexInArray - imageWidth + 1];
            gValue2 = pgImg[indexInArray - imageWidth];
            gValue3 = pgImg[indexInArray - imageWidth - 1];
            if (gValue1 >= gValue2 && gValue1 >= gValue3) {//up-right
              x = x + 1;
              y = y - 1;
            } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {//up-left
              x = x - 1;
              y = y - 1;
            } else {//straight-up
              y = y - 1;
            }
            lastDirection = LBD_UpDir;
          }
        }
        indexInArray = y * imageWidth + x;
      }//end while go left
      //end anchor is Horizontal
    } else {//the direction of this pixel is vertical, go up and down
      //fist go down, pixel direction may be different during linking.
      lastDirection = LBD_DownDir;
      while (pgImg[indexInArray] > 0 && !pEdgeImg[indexInArray]) {
        pEdgeImg[indexInArray] = 1;        // Mark this pixel as an edge pixel
        pFirstPartEdgeX_[offsetPFirst] = x;
        pFirstPartEdgeY_[offsetPFirst++] = y;
        shouldGoDirection = 0;//unknown
        if (pdirImg[indexInArray] == LBD_Horizontal) {//should go left or right
          if (lastDirection == LBD_UpDir || lastDirection == LBD_DownDir) {//change the pixel direction now
            if (x > lastX) {//should go right
              shouldGoDirection = LBD_RightDir;
            } else {//should go left
              shouldGoDirection = LBD_LeftDir;
            }
          }
          lastX = x;
          lastY = y;
          if (lastDirection == LBD_RightDir || shouldGoDirection == LBD_RightDir) {//go right
            if (x == imageWidth - 1 || y == 0 || y == imageHeight - 1) {//reach the image border
              break;
            }
            // Look at 3 neighbors to the right and pick the one with the max. gradient value
            gValue1 = pgImg[indexInArray - imageWidth + 1];
            gValue2 = pgImg[indexInArray + 1];
            gValue3 = pgImg[indexInArray + imageWidth + 1];
            if (gValue1 >= gValue2 && gValue1 >= gValue3) {//up-right
              x = x + 1;
              y = y - 1;
            } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {//down-right
              x = x + 1;
              y = y + 1;
            } else {//straight-right
              x = x + 1;
            }
            lastDirection = LBD_RightDir;
          } else if (lastDirection == LBD_LeftDir || shouldGoDirection == LBD_LeftDir) {//go left
            if (x == 0 || y == 0 || y == imageHeight - 1) {//reach the image border
              break;
            }
            // Look at 3 neighbors to the left and pick the one with the max. gradient value
            gValue1 = pgImg[indexInArray - imageWidth - 1];
            gValue2 = pgImg[indexInArray - 1];
            gValue3 = pgImg[indexInArray + imageWidth - 1];
            if (gValue1 >= gValue2 && gValue1 >= gValue3) {//up-left
              x = x - 1;
              y = y - 1;
            } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {//down-left
              x = x - 1;
              y = y + 1;
            } else {//straight-left
              x = x - 1;
            }
            lastDirection = LBD_LeftDir;
          }
        } else {//should go up or down.
          if (lastDirection == LBD_RightDir || lastDirection == LBD_LeftDir) {//change the pixel direction now
            if (y > lastY) {//should go down
              shouldGoDirection = LBD_DownDir;
            } else {//should go up
              shouldGoDirection = LBD_UpDir;
            }
          }
          lastX = x;
          lastY = y;
          if (lastDirection == LBD_DownDir || shouldGoDirection == LBD_DownDir) {//go down
            if (x == 0 || x == imageWidth - 1 || y == imageHeight - 1) {//reach the image border
              break;
            }
            // Look at 3 neighbors to the down and pick the one with the max. gradient value
            gValue1 = pgImg[indexInArray + imageWidth + 1];
            gValue2 = pgImg[indexInArray + imageWidth];
            gValue3 = pgImg[indexInArray + imageWidth - 1];
            if (gValue1 >= gValue2 && gValue1 >= gValue3) {//down-right
              x = x + 1;
              y = y + 1;
            } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {//down-left
              x = x - 1;
              y = y + 1;
            } else {//straight-down
              y = y + 1;
            }
            lastDirection = LBD_DownDir;
          } else if (lastDirection == LBD_UpDir || shouldGoDirection == LBD_UpDir) {//go up
            if (x == 0 || x == imageWidth - 1 || y == 0) {//reach the image border
              break;
            }
            // Look at 3 neighbors to the up and pick the one with the max. gradient value
            gValue1 = pgImg[indexInArray - imageWidth + 1];
            gValue2 = pgImg[indexInArray - imageWidth];
            gValue3 = pgImg[indexInArray - imageWidth - 1];
            if (gValue1 >= gValue2 && gValue1 >= gValue3) {//up-right
              x = x + 1;
              y = y - 1;
            } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {//up-left
              x = x - 1;
              y = y - 1;
            } else {//straight-up
              y = y - 1;
            }
            lastDirection = LBD_UpDir;
          }
        }
        indexInArray = y * imageWidth + x;
      }//end while go down
      //then go up, pixel direction may be different during linking.
      lastDirection = LBD_UpDir;
      x = pAnchorX_[i];
      y = pAnchorY_[i];
      indexInArray = y * imageWidth + x;
      pEdgeImg[indexInArray] = 0;//mark the anchor point be a non-edge pixel and
      pSecondPartEdgeS_[offsetPS] = offsetPSecond;
      while (pgImg[indexInArray] > 0 && !pEdgeImg[indexInArray]) {
        pEdgeImg[indexInArray] = 1;        // Mark this pixel as an edge pixel
        pSecondPartEdgeX_[offsetPSecond] = x;
        pSecondPartEdgeY_[offsetPSecond++] = y;
        shouldGoDirection = 0;//unknown
        if (pdirImg[indexInArray] == LBD_Horizontal) {//should go left or right
          if (lastDirection == LBD_UpDir || lastDirection == LBD_DownDir) {//change the pixel direction now
            if (x > lastX) {//should go right
              shouldGoDirection = LBD_RightDir;
            } else {//should go left
              shouldGoDirection = LBD_LeftDir;
            }
          }
          lastX = x;
          lastY = y;
          if (lastDirection == LBD_RightDir || shouldGoDirection == LBD_RightDir) {//go right
            if (x == imageWidth - 1 || y == 0 || y == imageHeight - 1) {//reach the image border
              break;
            }
            // Look at 3 neighbors to the right and pick the one with the max. gradient value
            gValue1 = pgImg[indexInArray - imageWidth + 1];
            gValue2 = pgImg[indexInArray + 1];
            gValue3 = pgImg[indexInArray + imageWidth + 1];
            if (gValue1 >= gValue2 && gValue1 >= gValue3) {//up-right
              x = x + 1;
              y = y - 1;
            } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {//down-right
              x = x + 1;
              y = y + 1;
            } else {//straight-right
              x = x + 1;
            }
            lastDirection = LBD_RightDir;
          } else if (lastDirection == LBD_LeftDir || shouldGoDirection == LBD_LeftDir) {//go left
            if (x == 0 || y == 0 || y == imageHeight - 1) {//reach the image border
              break;
            }
            // Look at 3 neighbors to the left and pick the one with the max. gradient value
            gValue1 = pgImg[indexInArray - imageWidth - 1];
            gValue2 = pgImg[indexInArray - 1];
            gValue3 = pgImg[indexInArray + imageWidth - 1];
            if (gValue1 >= gValue2 && gValue1 >= gValue3) {//up-left
              x = x - 1;
              y = y - 1;
            } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {//down-left
              x = x - 1;
              y = y + 1;
            } else {//straight-left
              x = x - 1;
            }
            lastDirection = LBD_LeftDir;
          }
        } else {//should go up or down.
          if (lastDirection == LBD_RightDir || lastDirection == LBD_LeftDir) {//change the pixel direction now
            if (y > lastY) {//should go down
              shouldGoDirection = LBD_DownDir;
            } else {//should go up
              shouldGoDirection = LBD_UpDir;
            }
          }
          lastX = x;
          lastY = y;
          if (lastDirection == LBD_DownDir || shouldGoDirection == LBD_DownDir) {//go down
            if (x == 0 || x == imageWidth - 1 || y == imageHeight - 1) {//reach the image border
              break;
            }
            // Look at 3 neighbors to the down and pick the one with the max. gradient value
            gValue1 = pgImg[indexInArray + imageWidth + 1];
            gValue2 = pgImg[indexInArray + imageWidth];
            gValue3 = pgImg[indexInArray + imageWidth - 1];
            if (gValue1 >= gValue2 && gValue1 >= gValue3) {//down-right
              x = x + 1;
              y = y + 1;
            } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {//down-left
              x = x - 1;
              y = y + 1;
            } else {//straight-down
              y = y + 1;
            }
            lastDirection = LBD_DownDir;
          } else if (lastDirection == LBD_UpDir || shouldGoDirection == LBD_UpDir) {//go up
            if (x == 0 || x == imageWidth - 1 || y == 0) {//reach the image border
              break;
            }
            // Look at 3 neighbors to the up and pick the one with the max. gradient value
            gValue1 = pgImg[indexInArray - imageWidth + 1];
            gValue2 = pgImg[indexInArray - imageWidth];
            gValue3 = pgImg[indexInArray - imageWidth - 1];
            if (gValue1 >= gValue2 && gValue1 >= gValue3) {//up-right
              x = x + 1;
              y = y - 1;
            } else if (gValue3 >= gValue2 && gValue3 >= gValue1) {//up-left
              x = x - 1;
              y = y - 1;
            } else {//straight-up
              y = y - 1;
            }
            lastDirection = LBD_UpDir;
          }
        }
        indexInArray = y * imageWidth + x;
      }//end while go up
    }//end anchor is Vertical
    //only keep the edge chains whose length is larger than the minLineLen_;
    edgeLenFirst = offsetPFirst - pFirstPartEdgeS_[offsetPS];
    edgeLenSecond = offsetPSecond - pSecondPartEdgeS_[offsetPS];
    if (edgeLenFirst + edgeLenSecond < minLineLen_ + 1) {//short edge, drop it
      offsetPFirst = pFirstPartEdgeS_[offsetPS];
      offsetPSecond = pSecondPartEdgeS_[offsetPS];
    } else {
      offsetPS++;
    }
  }
  //store the last index
  pFirstPartEdgeS_[offsetPS] = offsetPFirst;
  pSecondPartEdgeS_[offsetPS] = offsetPSecond;
  if (offsetPS > maxNumOfEdge) {
    std::cout << "Edge drawing Error: The total number of edges is larger than MaxNumOfEdge, "
                 "numofedge = " << offsetPS << ", MaxNumOfEdge=" << maxNumOfEdge << std::endl;
    return -1;
  }
  if (offsetPFirst > edgePixelArraySize || offsetPSecond > edgePixelArraySize) {
    std::cout << "Edge drawing Error: The total number of edge pixels is larger than MaxNumOfEdgePixels, "
                 "numofedgePixel1 = " << offsetPFirst << ",  numofedgePixel2 = " << offsetPSecond <<
              ", MaxNumOfEdgePixel=" << edgePixelArraySize << std::endl;
    return -1;
  }

  /*now all the edge information are stored in pFirstPartEdgeX_, pFirstPartEdgeY_,
   *pFirstPartEdgeS_,  pSecondPartEdgeX_, pSecondPartEdgeY_, pSecondPartEdgeS_;
   *we should reorganize them into edgeChains for easily using.	*/
  int tempID;
  edgeChains.xCors.resize(offsetPFirst + offsetPSecond);
  edgeChains.yCors.resize(offsetPFirst + offsetPSecond);
  edgeChains.sId.resize(offsetPS + 1);
  unsigned int *pxCors = edgeChains.xCors.data();
  unsigned int *pyCors = edgeChains.yCors.data();
  unsigned int *psId = edgeChains.sId.data();
  offsetPFirst = 0;
  offsetPSecond = 0;
  unsigned int indexInCors = 0;
  unsigned int numOfEdges = 0;
  for (unsigned int edgeId = 0; edgeId < offsetPS; edgeId++) {
    //step1, put the first and second parts edge coordinates together from edge start to edge end
    psId[numOfEdges++] = indexInCors;
    indexInArray = pFirstPartEdgeS_[edgeId];
    offsetPFirst = pFirstPartEdgeS_[edgeId + 1];
    for (tempID = offsetPFirst - 1; tempID >= indexInArray; tempID--) {//add first part edge
      pxCors[indexInCors] = pFirstPartEdgeX_[tempID];
      pyCors[indexInCors++] = pFirstPartEdgeY_[tempID];
    }
    indexInArray = pSecondPartEdgeS_[edgeId];
    offsetPSecond = pSecondPartEdgeS_[edgeId + 1];
    for (tempID = indexInArray + 1; tempID < offsetPSecond; tempID++) {//add second part edge
      pxCors[indexInCors] = pSecondPartEdgeX_[tempID];
      pyCors[indexInCors++] = pSecondPartEdgeY_[tempID];
    }
  }
  psId[numOfEdges] = indexInCors;//the end index of the last edge
  edgeChains.numOfEdges = numOfEdges;

#ifdef DEBUGEdgeDrawing
  /*Show the extracted edge cvImage in color. Each chain is in different color.*/
  cv::Mat cvColorImg(imageHeight, imageWidth, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Vec3b s;
  int lowest = 100, highest = 255;
  int range = (highest - lowest) + 1;
  for (unsigned int i = 0; i < edgeChains.numOfEdges; i++) {
    s = cv::Vec3b(lowest + int(rand() % range), lowest + int(rand() % range), lowest + int(rand() % range));
    for (indexInCors = psId[i]; indexInCors < psId[i + 1]; indexInCors++) {
      cvColorImg.at<cv::Vec3b>(pyCors[indexInCors], pxCors[indexInCors]) = s;
    }
  }

  cv::namedWindow("EdgeColorImage", cv::WINDOW_AUTOSIZE);
  cv::imshow("EdgeColorImage", cvColorImg);
  cv::waitKey(0);
#endif
  return 1;
}

int EDLineDetector::EDline(cv::Mat &image, LineChains &lines, bool smoothed) {
  //first, call EdgeDrawing function to extract edges
  EdgeChains edges;
  if (!EdgeDrawing(image, edges, smoothed)) {
    std::cout << "Line Detection not finished" << std::endl;
    return -1;
  }
  //	bValidate_ =false;
  //detect lines
  unsigned int linePixelID = edges.sId[edges.numOfEdges];
  lines.xCors.resize(linePixelID);
  lines.yCors.resize(linePixelID);
  lines.sId.resize(5 * edges.numOfEdges);
  unsigned int *pEdgeXCors = edges.xCors.data();
  unsigned int *pEdgeYCors = edges.yCors.data();
  unsigned int *pEdgeSID = edges.sId.data();
  unsigned int *pLineXCors = lines.xCors.data();
  unsigned int *pLineYCors = lines.yCors.data();
  unsigned int *pLineSID = lines.sId.data();
  logNT_ = 2.0 * (log10((double) imageWidth) + log10((double) imageHeight));
  double lineFitErr;//the line fit error;
  cv::Vec2d lineEquation;//[a,b] for lines y=ax+b(horizontal) or x=ay+b(vertical)
  lineEquations_.clear();
  lineEndpoints.clear();
  lineDirection_.clear();
  unsigned char *pdirImg = dirImg_.data;
  unsigned int numOfLines = 0;
  unsigned int offsetInEdgeArrayS, offsetInEdgeArrayE, newOffsetS;//start index and end index
  unsigned int offsetInLineArray = 0;
  float direction;//line direction

  for (unsigned int edgeID = 0; edgeID < edges.numOfEdges; edgeID++) {
    offsetInEdgeArrayS = pEdgeSID[edgeID];
    offsetInEdgeArrayE = pEdgeSID[edgeID + 1];
    while (offsetInEdgeArrayE
        > offsetInEdgeArrayS + minLineLen_) {//extract line segments from an edge, may find more than one segments
      //find an initial line segment
      while (offsetInEdgeArrayE > offsetInEdgeArrayS + minLineLen_) {
        lineFitErr = LeastSquaresLineFit_(pEdgeXCors, pEdgeYCors, offsetInEdgeArrayS, lineEquation);
        if (lineFitErr <= lineFitErrThreshold_) break;//ok, an initial line segment detected
        offsetInEdgeArrayS +=
            LBD_SkipEdgePoint; //skip the first two pixel in the chain and try with the remaining pixels
      }
      if (lineFitErr > lineFitErrThreshold_) break; //no line is detected
      //An initial line segment is detected. Try to extend this line segment
      pLineSID[numOfLines] = offsetInLineArray;
      double coef1;//for a line ax+by+c=0, coef1 = 1/sqrt(a^2+b^2);
      double pointToLineDis;//for a line ax+by+c=0 and a point(xi, yi), pointToLineDis = coef1*|a*xi+b*yi+c|
      bool bExtended = true;
      bool bFirstTry = true;
      int numOfOutlier;//to against noise, we accept a few outlier of a line.
      int tryTimes = 0;
      if (pdirImg[pEdgeYCors[offsetInEdgeArrayS] * imageWidth + pEdgeXCors[offsetInEdgeArrayS]]
          == LBD_Horizontal) {//y=ax+b, i.e. ax-y+b=0
        while (bExtended) {
          tryTimes++;
          if (bFirstTry) {
            bFirstTry = false;
            for (int i = 0; i < minLineLen_; i++) {//First add the initial line segment to the line array
              pLineXCors[offsetInLineArray] = pEdgeXCors[offsetInEdgeArrayS];
              pLineYCors[offsetInLineArray++] = pEdgeYCors[offsetInEdgeArrayS++];
            }
          } else {//after each try, line is extended, line equation should be re-estimated
            //adjust the line equation
            lineFitErr = LeastSquaresLineFit_(pLineXCors, pLineYCors, pLineSID[numOfLines],
                                              newOffsetS, offsetInLineArray, lineEquation);
          }
          coef1 = 1 / sqrt(lineEquation[0] * lineEquation[0] + 1);
          numOfOutlier = 0;
          newOffsetS = offsetInLineArray;
          while (offsetInEdgeArrayE > offsetInEdgeArrayS) {
            pointToLineDis = fabs(lineEquation[0] * pEdgeXCors[offsetInEdgeArrayS] -
                pEdgeYCors[offsetInEdgeArrayS] + lineEquation[1]) * coef1;
            pLineXCors[offsetInLineArray] = pEdgeXCors[offsetInEdgeArrayS];
            pLineYCors[offsetInLineArray++] = pEdgeYCors[offsetInEdgeArrayS++];
            if (pointToLineDis > lineFitErrThreshold_) {
              numOfOutlier++;
              if (numOfOutlier > 3) break;
            } else {//we count number of connective outliers.
              numOfOutlier = 0;
            }
          }
          //pop back the last few outliers from lines and return them to edge chain
          offsetInLineArray -= numOfOutlier;
          offsetInEdgeArrayS -= numOfOutlier;
          if (offsetInLineArray - newOffsetS > 0 && tryTimes < LBD_TryTime) {//some new pixels are added to the line
          } else {
            bExtended = false;//no new pixels are added.
          }
        }
        //the line equation coefficients,for line w1x+w2y+w3 =0, we normalize it to make w1^2+w2^2 = 1.
        cv::Vec3d lineEqu(lineEquation[0] * coef1, -1 * coef1, lineEquation[1] * coef1);
        if (LineValidation_(pLineXCors,
                            pLineYCors,
                            pLineSID[numOfLines],
                            offsetInLineArray,
                            lineEqu,
                            direction)) {//check the line
          //store the line equation coefficients
          lineEquations_.push_back(lineEqu);
          /*At last, compute the line endpoints and store them.
           *we project the first and last pixels in the pixelChain onto the best fit line
           *to get the line endpoints.
           *xp= (w2^2*x0-w1*w2*y0-w3*w1)/(w1^2+w2^2)
           *yp= (w1^2*y0-w1*w2*x0-w3*w2)/(w1^2+w2^2)  */
          cv::Vec4f lineEndP;//line endpoints
          double a1 = lineEqu[1] * lineEqu[1];
          double a2 = lineEqu[0] * lineEqu[0];
          double a3 = lineEqu[0] * lineEqu[1];
          double a4 = lineEqu[2] * lineEqu[0];
          double a5 = lineEqu[2] * lineEqu[1];
          unsigned int Px = pLineXCors[pLineSID[numOfLines]];//first pixel
          unsigned int Py = pLineYCors[pLineSID[numOfLines]];
          lineEndP[0] = a1 * Px - a3 * Py - a4;//x
          lineEndP[1] = a2 * Py - a3 * Px - a5;//y
          Px = pLineXCors[offsetInLineArray - 1];//last pixel
          Py = pLineYCors[offsetInLineArray - 1];
          lineEndP[2] = a1 * Px - a3 * Py - a4;//x
          lineEndP[3] = a2 * Py - a3 * Px - a5;//y
          lineEndpoints.push_back(lineEndP);
          lineDirection_.push_back(direction);
          numOfLines++;
        } else {
          offsetInLineArray = pLineSID[numOfLines];// line was not accepted, the offset is set back
        }
      } else {//x=ay+b, i.e. x-ay-b=0
        while (bExtended) {
          tryTimes++;
          if (bFirstTry) {
            bFirstTry = false;
            for (int i = 0; i < minLineLen_; i++) {//First add the initial line segment to the line array
              pLineXCors[offsetInLineArray] = pEdgeXCors[offsetInEdgeArrayS];
              pLineYCors[offsetInLineArray++] = pEdgeYCors[offsetInEdgeArrayS++];
            }
          } else {//after each try, line is extended, line equation should be re-estimated
            //adjust the line equation
            lineFitErr = LeastSquaresLineFit_(pLineXCors, pLineYCors, pLineSID[numOfLines],
                                              newOffsetS, offsetInLineArray, lineEquation);
          }
          coef1 = 1 / sqrt(1 + lineEquation[0] * lineEquation[0]);
          numOfOutlier = 0;
          newOffsetS = offsetInLineArray;
          while (offsetInEdgeArrayE > offsetInEdgeArrayS) {
            pointToLineDis = fabs(pEdgeXCors[offsetInEdgeArrayS] -
                lineEquation[0] * pEdgeYCors[offsetInEdgeArrayS] - lineEquation[1]) * coef1;
            pLineXCors[offsetInLineArray] = pEdgeXCors[offsetInEdgeArrayS];
            pLineYCors[offsetInLineArray++] = pEdgeYCors[offsetInEdgeArrayS++];
            if (pointToLineDis > lineFitErrThreshold_) {
              numOfOutlier++;
              if (numOfOutlier > 3) break;
            } else {//we count number of connective outliers.
              numOfOutlier = 0;
            }
          }
          //pop back the last few outliers from lines and return them to edge chain
          offsetInLineArray -= numOfOutlier;
          offsetInEdgeArrayS -= numOfOutlier;
          if (offsetInLineArray - newOffsetS > 0 && tryTimes < LBD_TryTime) {//some new pixels are added to the line
          } else {
            bExtended = false;//no new pixels are added.
          }
        }
        //the line equation coefficients,for line w1x+w2y+w3 =0, we normalize it to make w1^2+w2^2 = 1.
        cv::Vec3d lineEqu(1 * coef1, -lineEquation[0] * coef1, -lineEquation[1] * coef1);
        if (LineValidation_(pLineXCors,
                            pLineYCors,
                            pLineSID[numOfLines],
                            offsetInLineArray,
                            lineEqu,
                            direction)) {//check the line
          //store the line equation coefficients
          lineEquations_.push_back(lineEqu);
          /*At last, compute the line endpoints and store them.
           *we project the first and last pixels in the pixelChain onto the best fit line
           *to get the line endpoints.
           *xp= (w2^2*x0-w1*w2*y0-w3*w1)/(w1^2+w2^2)
           *yp= (w1^2*y0-w1*w2*x0-w3*w2)/(w1^2+w2^2)  */
          cv::Vec4f lineEndP;//line endpoints
          double a1 = lineEqu[1] * lineEqu[1];
          double a2 = lineEqu[0] * lineEqu[0];
          double a3 = lineEqu[0] * lineEqu[1];
          double a4 = lineEqu[2] * lineEqu[0];
          double a5 = lineEqu[2] * lineEqu[1];
          unsigned int Px = pLineXCors[pLineSID[numOfLines]];//first pixel
          unsigned int Py = pLineYCors[pLineSID[numOfLines]];
          lineEndP[0] = a1 * Px - a3 * Py - a4;//x
          lineEndP[1] = a2 * Py - a3 * Px - a5;//y
          Px = pLineXCors[offsetInLineArray - 1];//last pixel
          Py = pLineYCors[offsetInLineArray - 1];
          lineEndP[2] = a1 * Px - a3 * Py - a4;//x
          lineEndP[3] = a2 * Py - a3 * Px - a5;//y
          lineEndpoints.push_back(lineEndP);
          lineDirection_.push_back(direction);
          numOfLines++;
        } else {
          offsetInLineArray = pLineSID[numOfLines];// line was not accepted, the offset is set back
        }
      }
      //Extract line segments from the remaining pixel; Current chain has been shortened already.
    }
  }//end for(unsigned int edgeID=0; edgeID<edges.numOfEdges; edgeID++)
  if (numOfLines > 0) {
    pLineSID[numOfLines] = offsetInLineArray;
  }
  lines.numOfLines = numOfLines;
#ifdef DEBUGEDLine
  cout<<"Time to detect lines"<<endl;
  /*Show the extracted lines in color. Each line is in different color.*/
  IplImage* cvColorImg = cvCreateImage(cvSize(imageWidth,imageHeight),IPL_DEPTH_8U, 3);
  cvSet(cvColorImg, cvScalar(0,0,0));
  CvScalar s;
  srand((unsigned)time(0));
  int lowest=100, highest=255;
  int range=(highest-lowest)+1;
  //	CvPoint point;
  //	CvFont  font;
  //	cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX ,1.0,1.0,0,1);
  int r, g, b; //the color of lines
  for(unsigned int i=0; i<lines.numOfLines; i++){
      r = lowest+int(rand()%range);
      g = lowest+int(rand()%range);
      b = lowest+int(rand()%range);
      s.val[0] = b; s.val[1] = g;  s.val[2] = r;
      for(offsetInLineArray = pLineSID[i]; offsetInLineArray<pLineSID[i+1]; offsetInLineArray++){
          cvSet2D(cvColorImg,pLineYCors[offsetInLineArray],pLineXCors[offsetInLineArray],s);
      }
      //		iter = lines[i].begin();
      //		point = cvPoint(iter->x,iter->y);
      //		char buf[10];
      //		sprintf( buf,   "%d ",  i);
      //		cvPutText(cvColorImg,buf,point,&font,CV_RGB(r,g,b));
  }
  cvNamedWindow("LineColorImage", CV_WINDOW_AUTOSIZE);
  cvShowImage("LineColorImage", cvColorImg);
  cvWaitKey(0);
  cvReleaseImage(&cvColorImg);
#endif

  float dx, dy;
  bool shouldChange;
  for (int i = 0; i < lines.numOfLines; i++) {
    cv::Vec4f &endpoints = lineEndpoints[i];
    direction = lineDirection_[i];
    dx = endpoints[2] - endpoints[0];
    dy = endpoints[3] - endpoints[1];

    float gradx = 0, grady = 0;
    for (Pixel &px : bresenham(endpoints[0], endpoints[1], endpoints[2], endpoints[3])) {
      gradx += dxImg_.at<short>(px.y, px.x);
      grady += dyImg_.at<short>(px.y, px.x);
    }

    // For a 90 degrees rotation: dx, dy = -dy, dx
    shouldChange = (-dy * gradx + dx * grady) < 0;
    if (shouldChange) endpoints = {endpoints[2], endpoints[3], endpoints[0], endpoints[1]};
    dx = endpoints[2] - endpoints[0];
    dy = endpoints[3] - endpoints[1];
    lineDirection_[i] = std::atan2(dy, dx);
  }
  
  return 1;
}

double EDLineDetector::LeastSquaresLineFit_(unsigned int *xCors, unsigned int *yCors,
                                            unsigned int offsetS, cv::Vec2d &lineEquation) {
  float *pMatT;
  float *pATA;
  double fitError = 0;
  double coef;
  unsigned char *pdirImg = dirImg_.data;
  unsigned int offset = offsetS;
  /*If the first pixel in this chain is horizontal,
   *then we try to find a horizontal line, y=ax+b;*/
  if (pdirImg[yCors[offsetS] * imageWidth + xCors[offsetS]] == LBD_Horizontal) {
    /*Build the system,and solve it using least square regression: mat * [a,b]^T = vec
     * [x0,1]         [y0]
     * [x1,1] [a]     [y1]
     *    .   [b]  =   .
     * [xn,1]         [yn]*/
    pMatT = fitMatT.ptr<float>();//fitMatT = [x0, x1, ... xn; 1,1,...,1];
    for (int i = 0; i < minLineLen_; i++) {
      //*(pMatT+minLineLen_) = 1; //the value are not changed;
      *(pMatT++) = xCors[offsetS];
      fitVec(i) = yCors[offsetS++];
    }
    // Multiplies with transpose of A and returns result M*A'.
    // fitMatT.MultiplyWithTransposeOf(fitMatT, ATA);
    ATA = fitMatT * fitMatT.t();
    assert(fitMatT.cols == fitVec.rows);
    ATV = fitMatT * fitVec;
    /* [a,b]^T = Inv(mat^T * mat) * mat^T * vec */
    pATA = ATA.ptr<float>();
    coef = 1.0 / (double(pATA[0]) * double(pATA[3]) - double(pATA[1]) * double(pATA[2]));
    //		lineEquation = svd.Invert(ATA) * matT * vec;
    lineEquation[0] = coef * (double(pATA[3]) * double(ATV(0)) - double(pATA[1]) * double(ATV(1)));
    lineEquation[1] = coef * (double(pATA[0]) * double(ATV(1)) - double(pATA[2]) * double(ATV(0)));
    /*compute line fit error */
    for (int i = 0; i < minLineLen_; i++) {
      coef = double(yCors[offset]) - double(xCors[offset++]) * lineEquation[0] - lineEquation[1];
      fitError += coef * coef;
    }
    return sqrt(fitError);
  }
  /*If the first pixel in this chain is vertical,
   *then we try to find a vertical line, x=ay+b;*/
  if (pdirImg[yCors[offsetS] * imageWidth + xCors[offsetS]] == LBD_Vertical) {
    /*Build the system,and solve it using least square regression: mat * [a,b]^T = vec
     * [y0,1]         [x0]
     * [y1,1] [a]     [x1]
     *    .   [b]  =   .
     * [yn,1]         [xn]*/
    pMatT = fitMatT.ptr<float>();//fitMatT = [y0, y1, ... yn; 1,1,...,1];
    for (int i = 0; i < minLineLen_; i++) {
      //*(pMatT+minLineLen_) = 1;//the value are not changed;
      *(pMatT++) = yCors[offsetS];
      fitVec(i) = xCors[offsetS++];
    }
    // Multiplies with transpose of A and returns result M*A'.
    // fitMatT.MultiplyWithTransposeOf(fitMatT, ATA);
    ATA = fitMatT * fitMatT.t();
    assert(fitMatT.cols == fitVec.rows);
    ATV = fitMatT * fitVec;
    /* [a,b]^T = Inv(mat^T * mat) * mat^T * vec */
    pATA = ATA.ptr<float>();
    coef = 1.0 / (double(pATA[0]) * double(pATA[3]) - double(pATA[1]) * double(pATA[2]));
    //		lineEquation = svd.Invert(ATA) * matT * vec;
    lineEquation[0] = coef * (double(pATA[3]) * double(ATV(0)) - double(pATA[1]) * double(ATV(1)));
    lineEquation[1] = coef * (double(pATA[0]) * double(ATV(1)) - double(pATA[2]) * double(ATV(0)));
    /*compute line fit error */
    for (int i = 0; i < minLineLen_; i++) {
      coef = double(xCors[offset]) - double(yCors[offset++]) * lineEquation[0] - lineEquation[1];
      fitError += coef * coef;
    }
    return sqrt(fitError);
  }
  return 0;
}

double EDLineDetector::LeastSquaresLineFit_(unsigned int *xCors, unsigned int *yCors,
                                            unsigned int offsetS, unsigned int newOffsetS,
                                            unsigned int offsetE, cv::Vec2d &lineEquation) {
  int length = offsetE - offsetS;
  int newLength = offsetE - newOffsetS;
  if (length <= 0 || newLength <= 0) {
    std::cout << "EDLineDetector::LeastSquaresLineFit_ Error:"
                 " the expected line index is wrong...offsetE = "
              << offsetE << ", offsetS=" << offsetS << ", newOffsetS=" << newOffsetS << std::endl;
    return -1;
  }
  cv::Mat_<float> matT(2, newLength);
  cv::Mat_<float> vec(newLength, 1);
  float *pMatT;
  float *pATA;
  //	double fitError = 0;
  double coef;
  unsigned char *pdirImg = dirImg_.data;
  /*If the first pixel in this chain is horizontal,
   *then we try to find a horizontal line, y=ax+b;*/
  if (pdirImg[yCors[offsetS] * imageWidth + xCors[offsetS]] == LBD_Horizontal) {
    /*Build the new system,and solve it using least square regression: mat * [a,b]^T = vec
     * [x0',1]         [y0']
     * [x1',1] [a]     [y1']
     *    .    [b]  =   .
     * [xn',1]         [yn']*/
    pMatT = matT.ptr<float>();//matT = [x0', x1', ... xn'; 1,1,...,1]
    for (int i = 0; i < newLength; i++) {
      *(pMatT + newLength) = 1;
      *(pMatT++) = xCors[newOffsetS];
      vec(i) = yCors[newOffsetS++];
    }
    /* [a,b]^T = Inv(ATA + mat^T * mat) * (ATV + mat^T * vec) */
    // Multiplies with transpose of A and returns result M*A'.
    // matT.MultiplyWithTransposeOf(matT, tempMatLineFit);
    tempMatLineFit = matT * matT.t();
    assert(matT.cols == vec.rows);
    tempVecLineFit = matT * vec;
    ATA = ATA + tempMatLineFit;
    ATV = ATV + tempVecLineFit;
    pATA = ATA.ptr<float>();
    coef = 1.0 / (double(pATA[0]) * double(pATA[3]) - double(pATA[1]) * double(pATA[2]));
    lineEquation[0] = coef * (double(pATA[3]) * double(ATV(0)) - double(pATA[1]) * double(ATV(1)));
    lineEquation[1] = coef * (double(pATA[0]) * double(ATV(1)) - double(pATA[2]) * double(ATV(0)));
    /*compute line fit error */
    //		for(int i=0; i<length; i++){
    //			coef = double(yCors[offsetS]) - double(xCors[offsetS++]) * lineEquation[0] - lineEquation[1];
    //			fitError += coef*coef;
    //		}
    return 0;
  }
  /*If the first pixel in this chain is vertical,
   *then we try to find a vertical line, x=ay+b;*/
  if (pdirImg[yCors[offsetS] * imageWidth + xCors[offsetS]] == LBD_Vertical) {
    /*Build the system,and solve it using least square regression: mat * [a,b]^T = vec
     * [y0',1]         [x0']
     * [y1',1] [a]     [x1']
     *    .    [b]  =   .
     * [yn',1]         [xn']*/
    pMatT = matT.ptr<float>();//matT = [y0', y1', ... yn'; 1,1,...,1]
    for (int i = 0; i < newLength; i++) {
      *(pMatT + newLength) = 1;
      *(pMatT++) = yCors[newOffsetS];
      vec(i) = xCors[newOffsetS++];
    }
    /* [a,b]^T = Inv(ATA + mat^T * mat) * (ATV + mat^T * vec) */
    // Multiplies with transpose of A and returns result M*A'.
    // matT.MultiplyWithTransposeOf(matT, tempMatLineFit);
    tempMatLineFit = matT * matT.t();
    assert(matT.cols == vec.rows);
    tempVecLineFit = matT * vec;
    ATA = ATA + tempMatLineFit;
    ATV = ATV + tempVecLineFit;
    pATA = ATA.ptr<float>();
    coef = 1.0 / (double(pATA[0]) * double(pATA[3]) - double(pATA[1]) * double(pATA[2]));
    lineEquation[0] = coef * (double(pATA[3]) * double(ATV(0)) - double(pATA[1]) * double(ATV(1)));
    lineEquation[1] = coef * (double(pATA[0]) * double(ATV(1)) - double(pATA[2]) * double(ATV(0)));
    /*compute line fit error */
    //		for(int i=0; i<length; i++){
    //			coef = double(xCors[offsetS]) - double(yCors[offsetS++]) * lineEquation[0] - lineEquation[1];
    //			fitError += coef*coef;
    //		}
  }
  return 0;
}

bool EDLineDetector::LineValidation_(unsigned int *xCors, unsigned int *yCors,
                                     unsigned int offsetS, unsigned int offsetE,
                                     cv::Vec3d &lineEquation, float &direction) {
  if (bValidate_) {
    int n = offsetE - offsetS;
    /*first compute the direction of line, make sure that the dark side always be the
     *left side of a line.*/
    int meanGradientX = 0, meanGradientY = 0;
    short *pdxImg = dxImg_.ptr<short>();
    short *pdyImg = dyImg_.ptr<short>();
    double dx, dy;
    cv::Mat_<double> pointDirection(1, n);
    double *pPointDirection = pointDirection.ptr<double>();
    int index;
    for (int i = 0; i < n; i++) {
      index = yCors[offsetS] * imageWidth + xCors[offsetS++];
      meanGradientX += pdxImg[index];
      meanGradientY += pdyImg[index];
      dx = (double) pdxImg[index];
      dy = (double) pdyImg[index];
      *(pPointDirection++) = atan2(-dx, dy);
    }
    dx = fabs(lineEquation[1]);
    dy = fabs(lineEquation[0]);
    if (meanGradientX == 0 && meanGradientY == 0) {//not possible, if happens, it must be a wrong line,
      return false;
    }
    if (meanGradientX > 0 && meanGradientY >= 0) {//first quadrant, and positive direction of X axis.
      direction = atan2(-dy, dx);//line direction is in fourth quadrant
    }
    if (meanGradientX <= 0 && meanGradientY > 0) {//second quadrant, and positive direction of Y axis.
      direction = atan2(dy, dx);//line direction is in first quadrant
    }
    if (meanGradientX < 0 && meanGradientY <= 0) {//third quadrant, and negative direction of X axis.
      direction = atan2(dy, -dx);//line direction is in second quadrant
    }
    if (meanGradientX >= 0 && meanGradientY < 0) {//fourth quadrant, and negative direction of Y axis.
      direction = atan2(-dy, -dx);//line direction is in third quadrant
    }
    /*then check whether the line is on the border of the image. We don't keep the border line.*/
    if (fabs(direction) < 0.15 || M_PI - fabs(direction) < 0.15) {//Horizontal line
      if (fabs(lineEquation[2]) < 10 || fabs(imageHeight - fabs(lineEquation[2])) < 10) {//upper border or lower border
        return false;
      }
    }
    if (fabs(fabs(direction) - M_PI * 0.5) < 0.15) {//Vertical line
      if (fabs(lineEquation[2]) < 10 || fabs(imageWidth - fabs(lineEquation[2])) < 10) {//left border or right border
        return false;
      }
    }
    //count the aligned points on the line which have the same direction as the line.
    double disDirection;
    int k = 0;
    for (int i = 0; i < n; i++) {
      disDirection = fabs(direction - pointDirection(i));
      if (fabs(2 * M_PI - disDirection) < 0.392699
          || disDirection < 0.392699) {//same direction, pi/8 = 0.392699081698724
        k++;
      }
    }
    //now compute NFA(Number of False Alarms)
    double ret = nfa(n, k, 0.125, logNT_);

    return (ret > 0); //0 corresponds to 1 mean false alarm
  } else {
    return true;
  }
}

int EDLineDetector::EDline(cv::Mat &image, bool smoothed) {
  if (!EDline(image, lines_, smoothed)) {
    return -1;
  }
  lineSalience_.clear();
  lineSalience_.resize(lines_.numOfLines);
  unsigned char *pgImg = gImgWO_.data;
  unsigned int indexInLineArray;
  unsigned int *pXCor = lines_.xCors.data();
  unsigned int *pYCor = lines_.yCors.data();
  unsigned int *pSID = lines_.sId.data();
  for (unsigned int i = 0; i < lineSalience_.size(); i++) {
    int salience = 0;
    for (indexInLineArray = pSID[i]; indexInLineArray < pSID[i + 1]; indexInLineArray++) {
      salience += pgImg[pYCor[indexInLineArray] * imageWidth + pXCor[indexInLineArray]];
    }
    lineSalience_[i] = (float) salience;
  }
  return 1;
}
eth::Segments EDLineDetector::detect(const cv::Mat &image) {
  cv::Mat tmp(image);
  this->EDline(tmp, true);
  return lineEndpoints;
}

eth::SalientSegments EDLineDetector::detectSalient(const cv::Mat &image) {
  cv::Mat tmp(image);
  this->EDline(tmp, true);

  eth::SalientSegments result(lineEndpoints.size());
  for (int i = 0; i < lineEndpoints.size(); i++) {
    result[i].segment = lineEndpoints[i];
    result[i].salience = lineSalience_[i];
  }
  return result;
}

std::shared_ptr<eth::SegmentsDetector> EDLineDetector::clone() const {

  EDLineParam params{ksize_, sigma_, float(gradienThreshold_), float(anchorThreshold_),
                     int(scanIntervals_), minLineLen_, lineFitErrThreshold_};
  return std::make_shared<EDLineDetector>(params);
}

}  // namespace eth