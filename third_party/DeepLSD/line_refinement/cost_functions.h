// #ifndef COST_FUNCTIONS_H
// #define COST_FUNCTIONS_H

/**
 *  Regress line endpoints from a line distance field with optimization.
 *  Can process each line independently or all together.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <array>
#include <algorithm>

#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"


using namespace std;
using namespace ceres;


static const int n_samples = 10;


/** Interpolate f at position (x, y).
 */
template <typename JetT>
JetT bilinear_interpolator(const double* f, JetT x, JetT y, int rows, int cols)
{
    // Check if the point is out of boundary
    if(x.a < 0)
        return bilinear_interpolator(f, (JetT) 0, y, rows, cols);
    if(x.a > cols - 1)
        return bilinear_interpolator(f, (JetT) (cols - 1), y, rows, cols);
    if(y.a < 0)
        return bilinear_interpolator(f, x, (JetT) 0, rows, cols);
    if(y.a > rows - 1)
        return bilinear_interpolator(f, x, (JetT) (rows - 1), rows, cols);

    // Compute the value of f in all 4 corners of (x, y)
    JetT fll = (JetT) f[(int) ceres::floor(y).a * cols + (int) ceres::floor(x).a];
    JetT flh = (JetT) f[(int) ceres::ceil(y).a * cols + (int) ceres::floor(x).a];
    JetT fhl = (JetT) f[(int) ceres::floor(y).a * cols + (int) ceres::ceil(x).a];
    JetT fhh = (JetT) f[(int) ceres::ceil(y).a * cols + (int) ceres::ceil(x).a];

    // Interpolate with weighting coefficients
    JetT interp = (fll * (ceres::ceil(x) - x) * (ceres::ceil(y) - y)
                   + flh * (ceres::ceil(x) - x) * (y - ceres::floor(y))
                   + fhl * (x - ceres::floor(x)) * (ceres::ceil(y) - y)
                   + fhh * (x - ceres::floor(x)) * (y - ceres::floor(y)));
    return interp;
}


double bilinear_interpolator(const double* f, double x, double y,
                             int rows, int cols)
{
    // Check if the point is out of boundary
    if(x < 0)
        return bilinear_interpolator(f, 0, y, rows, cols);
    if(x > cols - 1)
        return bilinear_interpolator(f, cols - 1, y, rows, cols);
    if(y < 0)
        return bilinear_interpolator(f, x, 0, rows, cols);
    if(y > rows - 1)
        return bilinear_interpolator(f, x, rows - 1, rows, cols);

    // Compute the value of f in all 4 corners of (x, y)
    double fll = f[(int) ceres::floor(y) * cols + (int) ceres::floor(x)];
    double flh = f[(int) ceres::ceil(y) * cols + (int) ceres::floor(x)];
    double fhl = f[(int) ceres::floor(y) * cols + (int) ceres::ceil(x)];
    double fhh = f[(int) ceres::ceil(y) * cols + (int) ceres::ceil(x)];

    // Interpolate with weighting coefficients
    double interp = (fll * (ceres::ceil(x) - x) * (ceres::ceil(y) - y)
                     + flh * (ceres::ceil(x) - x) * (y - ceres::floor(y))
                     + fhl * (x - ceres::floor(x)) * (ceres::ceil(y) - y)
                     + fhh * (x - ceres::floor(x)) * (y - ceres::floor(y)));
    return interp;
}


/** Nearest neighbor interpolation of f at position (x, y), with f.cols = cols.
 */
template <typename JetT>
JetT nn_interpolator(const double* f, JetT x, JetT y, int rows, int cols)
{
    // Check if the point is out of boundary
    if(x.a < 0)
        return nn_interpolator(f, (JetT) 0, y, rows, cols);
    if(x.a > cols - 1)
        return nn_interpolator(f, (JetT) (cols - 1), y, rows, cols);
    if(y.a < 0)
        return nn_interpolator(f, x, (JetT) 0, rows, cols);
    if(y.a > rows - 1)
        return nn_interpolator(f, x, (JetT) (rows - 1), rows, cols);
    return (JetT) f[(int) round(y.a) * cols + (int) round(x.a)];
}


double nn_interpolator(const double* f, double x, double y, int rows, int cols)
{
    // Check if the point is out of boundary
    if(x < 0)
        return nn_interpolator(f, 0, y, rows, cols);
    if(x > cols - 1)
        return nn_interpolator(f, cols - 1, y, rows, cols);
    if(y < 0)
        return nn_interpolator(f, x, 0, rows, cols);
    if(y > rows - 1)
        return nn_interpolator(f, x, rows - 1, rows, cols);
    return f[(int) round(y) * cols + (int) round(x)];
}


/** Cost functor that minimizes the DF between the endpoints.
 */
struct DfCostFunctor {
  explicit DfCostFunctor(
      const BiCubicInterpolator<Grid2D<double, 1>>& df_interpolator,
      double len, double cx, double cy)
      : df_interpolator_(df_interpolator), len_(len), cx_(cx), cy_(cy) {}

  template <typename T>
  bool operator()(const T* perp_dist, const T* ori, T* residuals) const {
    // Get the displaced center
    T dcx = cx_ + *perp_dist * ceres::cos(*ori + M_PI_2);
    T dcy = cy_ + *perp_dist * ceres::sin(*ori + M_PI_2);

    // Get the endpoints
    T x1 = dcx - len_ / 2 * ceres::cos(*ori);
    T y1 = dcy - len_ / 2 * ceres::sin(*ori);

    double alpha;
    for(int i=0; i <= n_samples; i++)
    {
        alpha = (double) i / (double) n_samples;
        df_interpolator_.Evaluate(y1 + alpha * len_ * ceres::sin(*ori),
                                  x1 + alpha * len_ * ceres::cos(*ori),
                                  &residuals[i]);
    }
    return true;
  }

  static CostFunction* Create(
      const BiCubicInterpolator<Grid2D<double, 1>>& df_interpolator,
      double len, double cx, double cy) {
    return new AutoDiffCostFunction<DfCostFunctor, n_samples + 1, 1, 1>(
        new DfCostFunctor(df_interpolator, len, cx, cy));
  }

 private:
  const BiCubicInterpolator<Grid2D<double, 1>>& df_interpolator_;
  const double len_;
  const double cx_;
  const double cy_;
};


/** Cost functor that forces the gradient angle to be consistent with the line.
 */
struct GradCostFunctor {
  explicit GradCostFunctor(const double* cos, const double* sin,
                               int rows, int cols, double len, double cx, double cy)
    : cos_(cos), sin_(sin), rows_(rows), cols_(cols), len_(len), cx_(cx), cy_(cy) {}

  template <typename T>
  bool operator()(const T* perp_dist, const T* ori, T* residuals) const {
    // Compute the unit vector direction
    T vec_x = ceres::cos(*ori);
    T vec_y = ceres::sin(*ori);

    // Get the displaced center
    T dcx = cx_ + *perp_dist * ceres::cos(*ori + M_PI_2);
    T dcy = cy_ + *perp_dist * ceres::sin(*ori + M_PI_2);

    // Get the endpoints
    T x1 = dcx - len_ / 2 * vec_x;
    T y1 = dcy - len_ / 2 * vec_y;

    T x, y, cos, sin;
    double alpha;
    for(int i=1; i < n_samples; i++)  // Ignore endpoints
    {
        alpha = (double) i / (double) n_samples;
        x = x1 + alpha * len_ * vec_x;
        y = y1 + alpha * len_ * vec_y;
        // cos = bilinear_interpolator(cos_, x, y, rows_, cols_);
        // sin = bilinear_interpolator(sin_, x, y, rows_, cols_);
        cos = nn_interpolator(cos_, x, y, rows_, cols_);
        sin = nn_interpolator(sin_, x, y, rows_, cols_);
        residuals[i - 1] = 1. - (cos * vec_x + sin * vec_y);
    }
    return true;
  }

  static CostFunction* Create(
      const double* cos, const double* sin, int rows,
      int cols, double len, double cx, double cy) {
    return new AutoDiffCostFunction<GradCostFunctor, n_samples - 1, 1, 1>(
        new GradCostFunctor(cos, sin, rows, cols, len, cx, cy));
  }

 private:
  const double* cos_;
  const double* sin_;
  int rows_;
  int cols_;
  double len_;
  double cx_;
  double cy_;
};


/** Cost functor that minimizes the maximum orthogonal distance between
    the endpoints of a line and its vanishing direction.
    Only the lines are optimized.
 */
struct LineVpCostFunctor {
  explicit LineVpCostFunctor(
      double len, double cx, double cy, const double* vp)
      : len_(len), cx_(cx), cy_(cy), vp_(vp) {}

  template <typename T>
  bool operator()(const T* perp_dist, const T* ori, T* residuals) const {
    // Get the displaced center
    T dcx = cx_ + *perp_dist * ceres::cos(*ori + M_PI_2);
    T dcy = cy_ + *perp_dist * ceres::sin(*ori + M_PI_2);

    // Get the endpoints
    T x1 = dcx - len_ / 2 * ceres::cos(*ori);
    T y1 = dcy - len_ / 2 * ceres::sin(*ori);

    // Line passing through the VP and the center of the line segment
    // l = cross(dc, vp)
    T l1 = dcy * vp_[2] - vp_[1];
    T l2 = vp_[0] - dcx * vp_[2];
    T l3 = dcx * vp_[1] - dcy * vp_[0];

    // Residual = max orthogonal distance of the two endpoints to l
    residuals[0] = (ceres::abs(x1 * l1 + y1 * l2 + l3)
                    / ceres::sqrt(l1 * l1 + l2 * l2));

    return true;
  }

  static CostFunction* Create(
      double len, double cx, double cy, double* vp) {
    return new AutoDiffCostFunction<LineVpCostFunctor, 1, 1, 1>(
        new LineVpCostFunctor(len, cx, cy, vp));
  }

 private:
  const double len_;
  const double cx_;
  const double cy_;
  const double* vp_;
};


/** Cost functor that minimizes the maximum orthogonal distance between
    the endpoints of a line and its vanishing direction.
    Only the vanishing points are optimized.
 */
struct VpCostFunctor {
  explicit VpCostFunctor(
      double x1, double y1, double dcx, double dcy)
      : x1_(x1), y1_(y1), dcx_(dcx), dcy_(dcy) {}

  template <typename T>
  bool operator()(const T* vp, T* residuals) const {
    // Line passing through the VP and the center of the line segment
    // l = cross(dc, vp)
    T l1 = dcy_ * vp[2] - vp[1];
    T l2 = vp[0] - dcx_ * vp[2];
    T l3 = dcx_ * vp[1] - dcy_ * vp[0];

    // Residual = max orthogonal distance of the two endpoints to l
    residuals[0] = (ceres::abs(x1_ * l1 + y1_ * l2 + l3)
                    / ceres::sqrt(l1 * l1 + l2 * l2));

    return true;
  }

  static CostFunction* Create(double x1, double y1, double dcx, double dcy) {
    return new AutoDiffCostFunction<VpCostFunctor, 1, 3>(
        new VpCostFunctor(x1, y1, dcx, dcy));
  }

 private:
  const double x1_;
  const double y1_;
  const double dcx_;
  const double dcy_;
};

// #endif
