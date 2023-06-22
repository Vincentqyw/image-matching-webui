/*
 * EDLineDetector.hh
 *
 *  Created on: Nov 30, 2011
 *      Author: lz
 */

#ifndef EDLINEDETECTOR_HH_
#define EDLINEDETECTOR_HH_

#include <vector>
#include <list>

#include <opencv2/opencv.hpp>
#include "multiscale/OctaveKeyLineDetector.h"
#include "SegmentsDetector.h"

namespace eth {
//
//struct Pixel {
//  unsigned int x;//X coordinate
//  unsigned int y;//Y coordinate
//};
struct EdgeChains {
  std::vector<unsigned int> xCors;//all the x coordinates of edge points
  std::vector<unsigned int> yCors;//all the y coordinates of edge points
  std::vector<unsigned int> sId;  //the start index of each edge in the coordinate arrays
  unsigned int numOfEdges;//the number of edges whose length are larger than minLineLen; numOfEdges < sId.size;
};
struct LineChains {
  std::vector<unsigned int> xCors;//all the x coordinates of line points
  std::vector<unsigned int> yCors;//all the y coordinates of line points
  std::vector<unsigned int> sId;  //the start index of each line in the coordinate arrays
  unsigned int numOfLines;//the number of lines whose length are larger than minLineLen; numOfLines < sId.size;
};

typedef std::list<Pixel> PixelChain;//each edge is a pixel chain


struct EDLineParam {
  int ksize;
  float sigma;
  float gradientThreshold;
  float anchorThreshold;
  int scanIntervals;
  int minLineLen;
  double lineFitErrThreshold;
};

/* This class is used to detect lines from input image.
 * First, edges are extracted from input image following the method presented in Cihan Topal and
 * Cuneyt Akinlar's paper:"Edge Drawing: A Heuristic Approach to Robust Real-Time Edge Detection", 2010.
 * Then, lines are extracted from the edge image following the method presented in Cuneyt Akinlar and
 * Cihan Topal's paper:"EDLines: A real-time line segment detector with a false detection control", 2011
 * PS: The linking step of edge detection has a little bit difference with the Edge drawing algorithm
 *     described in the paper. The edge chain doesn't stop when the pixel direction is changed.
 */
class EDLineDetector : public eth::OctaveKeyLineDetector {
 public:
  EDLineDetector();
  EDLineDetector(EDLineParam param);
  ~EDLineDetector();

  /*extract edges from image
   *image:    In, gray image;
   *edges:    Out, store the edges, each edge is a pixel chain
   *smoothed: In, flag to mark whether the image has already been smoothed by Gaussian filter.
   *return -1: error happen
   */
  int EdgeDrawing(cv::Mat &image, EdgeChains &edgeChains, bool smoothed = false);
  /*extract lines from image
   *image:    In, gray image;
   *lines:    Out, store the extracted lines,
   *smoothed: In, flag to mark whether the image has already been smoothed by Gaussian filter.
   *return -1: error happen
   */
  int EDline(cv::Mat &image, LineChains &lines, bool smoothed = false);
  /*extract line from image, and store them*/
  int EDline(cv::Mat &image, bool smoothed = false);

  eth::Segments detect(const cv::Mat &image) override;  // NOLINT(runtime/references)

  eth::SalientSegments detectSalient(const cv::Mat &image) override;

  std::string getName() const override { return "EdLinesLilianZhang"; }

  eth::SegmentsDetectorPtr clone() const override;

  bool doesSmooth() const override { return false; }

  const cv::Mat &getOctaveImg() const override { return octaveImg; }
  inline const cv::Mat &getDxImg() const override { return dxImg_; }
  inline const cv::Mat &getDyImg() const override { return dyImg_; }
  inline cv::Size getImgSize() const override { return cv::Size(imageWidth, imageHeight); }
  inline const std::vector<float> &getSegmentsDirection() const override { return lineDirection_; }
  inline const std::vector<float> &getSegmentsSalience() const override { return lineSalience_; }
  inline const std::vector<cv::Vec3d> &getLineEquations() const override { return lineEquations_; }
  inline size_t getNumberOfPixels(size_t segmentId) const override {
    assert(segmentId < lines_.numOfLines);
    return lines_.sId[segmentId + 1] - lines_.sId[segmentId];
  }

 private:
  cv::Mat gImgWO_;//store the gradient image without threshold;
  cv::Mat dxImg_;//store the dxImg;
  cv::Mat dyImg_;//store the dyImg;
  cv::Mat octaveImg;

  unsigned int imageWidth;
  unsigned int imageHeight;

  //store the line direction
  std::vector<float> lineDirection_;
  //store the line salience, which is the summation of gradients of pixels on line
  std::vector<float> lineSalience_;
  //store the line Equation coefficients, vec3=[w1,w2,w3] for line w1*x + w2*y + w3=0;
  std::vector<cv::Vec3d> lineEquations_;
  //store the detected line chains (pixels)
  LineChains lines_;

  void InitEDLine_();
  /*For an input edge chain, find the best fit line, the default chain length is minLineLen_
   *xCors:  In, pointer to the X coordinates of pixel chain;
   *yCors:  In, pointer to the Y coordinates of pixel chain;
   *offsetS:In, start index of this chain in array;
   *lineEquation: Out, [a,b] which are the coefficient of lines y=ax+b(horizontal) or x=ay+b(vertical);
   *return:  line fit error; -1:error happens;
   */
  double LeastSquaresLineFit_(unsigned int *xCors, unsigned int *yCors,
                              unsigned int offsetS, cv::Vec2d &lineEquation);
  /*For an input pixel chain, find the best fit line. Only do the update based on new points.
   *For A*x=v,  Least square estimation of x = Inv(A^T * A) * (A^T * v);
   *If some new observations are added, i.e, [A; A'] * x = [v; v'],
   *then x' = Inv(A^T * A + (A')^T * A') * (A^T * v + (A')^T * v');
   *xCors:  In, pointer to the X coordinates of pixel chain;
   *yCors:  In, pointer to the Y coordinates of pixel chain;
   *offsetS:In, start index of this chain in array;
   *newOffsetS: In, start index of extended part;
   *offsetE:In, end index of this chain in array;
   *lineEquation: Out, [a,b] which are the coefficient of lines y=ax+b(horizontal) or x=ay+b(vertical);
   *return:  line fit error; -1:error happens;
   */
  double LeastSquaresLineFit_(unsigned int *xCors, unsigned int *yCors,
                              unsigned int offsetS, unsigned int newOffsetS,
                              unsigned int offsetE, cv::Vec2d &lineEquation);
  /* Validate line based on the Helmholtz principle, which basically states that
   * for a structure to be perceptually meaningful, the expectation of this structure
   * by chance must be very low.
   */
  bool LineValidation_(unsigned int *xCors, unsigned int *yCors,
                       unsigned int offsetS, unsigned int offsetE,
                       cv::Vec3d &lineEquation, float &direction);
  bool bValidate_;//flag to decide whether line will be validated
  int ksize_; //the size of Gaussian kernel: ksize X ksize, default value is 5.
  float sigma_;//the sigma of Gaussian kernal, default value is 1.0.
  /*the threshold of pixel gradient magnitude.
   *Only those pixel whose gradient magnitude are larger than this threshold will be
   *taken as possible edge points. Default value is 36*/
  short gradienThreshold_;
  /*If the pixel's gradient value is bigger than both of its neighbors by a
   *certain threshold (ANCHOR_THRESHOLD), the pixel is marked to be an anchor.
   *Default value is 8*/
  unsigned char anchorThreshold_;
  /*anchor testing can be performed at different scan intervals, i.e.,
   *every row/column, every second row/column etc.
   *Default value is 2*/
  unsigned int scanIntervals_;
  int minLineLen_;//minimal acceptable line length
  /*For example, there two edges in the image:
   *edge1 = [(7,4), (8,5), (9,6),| (10,7)|, (11, 8), (12,9)] and
   *edge2 = [(14,9), (15,10), (16,11), (17,12),| (18, 13)|, (19,14)] ; then we store them as following:
   *pFirstPartEdgeX_ = [10, 11, 12, 18, 19];//store the first part of each edge[from middle to end]
   *pFirstPartEdgeY_ = [7,  8,  9,  13, 14];
   *pFirstPartEdgeS_ = [0,3,5];// the index of start point of first part of each edge
   *pSecondPartEdgeX_ = [10, 9, 8, 7, 18, 17, 16, 15, 14];//store the second part of each edge[from middle to front]
   *pSecondPartEdgeY_ = [7,  6, 5, 4, 13, 12, 11, 10, 9];//anchor points(10, 7) and (18, 13) are stored again
   *pSecondPartEdgeS_ = [0, 4, 9];// the index of start point of second part of each edge
   *This type of storage order is because of the order of edge detection process.
   *For each edge, start from one anchor point, first go right, then go left or first go down, then go up*/
  unsigned int *pFirstPartEdgeX_;//store the X coordinates of the first part of the pixels for chains
  unsigned int *pFirstPartEdgeY_;//store the Y coordinates of the first part of the pixels for chains
  unsigned int *pFirstPartEdgeS_;//store the start index of every edge chain in the first part arrays
  unsigned int *pSecondPartEdgeX_;//store the X coordinates of the second part of the pixels for chains
  unsigned int *pSecondPartEdgeY_;//store the Y coordinates of the second part of the pixels for chains
  unsigned int *pSecondPartEdgeS_;//store the start index of every edge chain in the second part arrays
  unsigned int *pAnchorX_;//store the X coordinates of anchors
  unsigned int *pAnchorY_;//store the Y coordinates of anchors
  cv::Mat_<uint8_t> edgeImage_;
  /*The threshold of line fit error;
   *If lineFitErr is large than this threshold, then
   *the pixel chain is not accepted as a single line segment.*/
  double lineFitErrThreshold_;

  cv::Mat_<uint8_t> gImg_;//store the gradient image;
  cv::Mat_<uint8_t> dirImg_;//store the direction image
  double logNT_;
  cv::Mat_<float> ATA;     //the previous matrix of A^T * A;
  cv::Mat_<float> ATV;    //the previous vector of A^T * V;
  cv::Mat_<float> fitMatT;     //the matrix used in line fit function;
  cv::Mat_<float> fitVec;    //the vector used in line fit function;
  cv::Mat_<float> tempMatLineFit;     //the matrix used in line fit function;
  cv::Mat_<float> tempVecLineFit;    //the vector used in line fit function;
};
}  // namespace eth
#endif /* EDLINEDETECTOR_HH_ */
