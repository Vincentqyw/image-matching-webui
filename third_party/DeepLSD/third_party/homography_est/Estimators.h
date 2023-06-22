#ifndef HOMOGRAPHY_EST_ESTIMATORS_H_
#define HOMOGRAPHY_EST_ESTIMATORS_H_

#include <vector>
#include <Eigen/Eigen>
#include "hest.h"

namespace hest {

Eigen::Matrix3d dlt_homography(
    const std::vector<Eigen::Vector3d> &p1,
    const std::vector<Eigen::Vector3d> &p2);

Eigen::Matrix3d est_rotation(
    const std::vector<Eigen::Vector2d> &p1,
    const std::vector<Eigen::Vector2d> &p2,
    const std::vector<LineSegment> &ls1,
    const std::vector<LineSegment> &ls2);

// Maps the endpoints and computes the point-to-line error in the first image
double computeHomographyLineResidual(
    const LineSegment &line_segment1,
    const LineSegment &line_segment2,
    const Eigen::Matrix3d &H);

// Maps the endpoints and computes the point-to-line error in the first image
double computeRotationLineResidual(
    const LineSegment &line_segment1,
    const LineSegment &line_segment2,
    const Eigen::Matrix3d &R);

double computeHomographyPointResidual(
    const Eigen::Vector2d &p1,
    const Eigen::Vector2d &p2,
    const Eigen::Matrix3d &H);

double computeRotationPointResidual(
    const Eigen::Vector2d &p1,
    const Eigen::Vector2d &p2,
    const Eigen::Matrix3d &R);

class LineHomographyEstimator {
 public:
  LineHomographyEstimator(const std::vector<LineSegment> &ls1,
                          const std::vector<LineSegment> &ls2,
                          const bool rot)
      : line_segments1(ls1), line_segments2(ls2), pure_rotation(rot) {}

  inline int min_sample_size() const {
    if (pure_rotation) {
      return 2;
    } else {
      return 4;
    }
  }

  inline int non_minimal_sample_size() const { return 4; }

  inline int num_data() const { return line_segments1.size(); }

  int MinimalSolver(const std::vector<int> &sample,
                    std::vector<Eigen::Matrix3d> *H) const {
    std::vector<LineSegment> ls1, ls2;
    ls1.resize(sample.size());
    ls2.resize(sample.size());
    for (size_t k = 0; k < sample.size(); ++k) {
      ls1[k] = line_segments1[sample[k]];
      ls2[k] = line_segments2[sample[k]];
    }
    H->clear();
    H->push_back(estimateHomographyLineSegments(ls1, ls2, pure_rotation));    
    return 1;
  }

  // Returns 0 if no model could be estimated and 1 otherwise.
  // Implemented by a simple linear least squares solver.
  int NonMinimalSolver(const std::vector<int> &sample,
                       Eigen::Matrix3d *H) const {
    std::vector<LineSegment> ls1, ls2;
    ls1.resize(sample.size());
    ls2.resize(sample.size());
    for (size_t k = 0; k < sample.size(); ++k) {
      ls1[k] = line_segments1[sample[k]];
      ls2[k] = line_segments2[sample[k]];
    }
    *H = estimateHomographyLineSegments(ls1, ls2, pure_rotation);    
    return 1;
  }

  // Evaluates the line on the i-th data point.
  double EvaluateModelOnPoint(const Eigen::Matrix3d &H, int i) const {
    double res;
    if (pure_rotation) {
      res = computeRotationLineResidual(line_segments1[i], line_segments2[i], H);
    } else {
      res = computeHomographyLineResidual(line_segments1[i], line_segments2[i], H);
    }
    return res * res;
  }

  // Linear least squares solver. Calls NonMinimalSolver.
  inline void LeastSquares(const std::vector<int> &sample,
                           Eigen::Matrix3d *H) const {
    std::vector<LineSegment> ls1, ls2;
    ls1.resize(sample.size());
    ls2.resize(sample.size());
    for (size_t k = 0; k < sample.size(); ++k) {
      ls1[k] = line_segments1[sample[k]];
      ls2[k] = line_segments2[sample[k]];
    }
    refineHomography(ls1, ls2, *H, pure_rotation);
  }

 protected:
  // Matrix holding the 2D points through which the line is fitted.
  const std::vector<LineSegment> &line_segments1;
  const std::vector<LineSegment> &line_segments2;
  const bool pure_rotation;
};

class PointHomographyEstimator {
 public:
  PointHomographyEstimator(const std::vector<Eigen::Vector2d> &points1,
                           const std::vector<Eigen::Vector2d> &points2,
                           const bool rot)
      : pts1(points1), pts2(points2), pure_rotation(rot) {}

  inline int min_sample_size() const {
    if (pure_rotation) {
      return 2;
    } else {
      return 4;
    }
  }

  inline int non_minimal_sample_size() const {
    return 4;
  }

  inline int num_data() const { return pts1.size(); }

  int MinimalSolver(const std::vector<int> &sample,
                    std::vector<Eigen::Matrix3d> *H) const {
    std::vector<Eigen::Vector2d> ps1, ps2;
    ps1.resize(sample.size());
    ps2.resize(sample.size());
    for (size_t k = 0; k < sample.size(); ++k) {
      ps1[k] = pts1[sample[k]];
      ps2[k] = pts2[sample[k]];
    }
    H->clear();
    H->push_back(estimateHomographyPoints(ps1, ps2, pure_rotation));
    return 1;
  }

  // Returns 0 if no model could be estimated and 1 otherwise.
  // Implemented by a simple linear least squares solver.
  int NonMinimalSolver(const std::vector<int> &sample,
                       Eigen::Matrix3d *H) const {
    std::vector<Eigen::Vector2d> ps1, ps2;
    ps1.resize(sample.size());
    ps2.resize(sample.size());
    for (size_t k = 0; k < sample.size(); ++k) {
      ps1[k] = pts1[sample[k]];
      ps2[k] = pts2[sample[k]];
    }
    *H = estimateHomographyPoints(ps1, ps2, pure_rotation);
    return 1;    
  }

  // Evaluates the line on the i-th data point.
  double EvaluateModelOnPoint(const Eigen::Matrix3d &H, int i) const {
    double res;
    if (pure_rotation) {
      res = computeRotationPointResidual(pts1[i], pts2[i], H);
    } else {
      res = computeHomographyPointResidual(pts1[i], pts2[i], H);
    }
    return res * res;
  }

  // Linear least squares solver. Calls NonMinimalSolver.
  inline void LeastSquares(const std::vector<int> &sample,
                           Eigen::Matrix3d *H) const {
    std::vector<Eigen::Vector2d> ps1, ps2;
    ps1.resize(sample.size());
    ps2.resize(sample.size());
    for (size_t k = 0; k < sample.size(); ++k) {
      ps1[k] = pts1[sample[k]];
      ps2[k] = pts2[sample[k]];
    }
    refineHomography(ps1, ps2, *H, pure_rotation);
  }

 protected:
  const std::vector<Eigen::Vector2d> &pts1;
  const std::vector<Eigen::Vector2d> &pts2;
  const bool pure_rotation;
};

class PointLineHomographyEstimator {
 public:
  PointLineHomographyEstimator(const std::vector<Eigen::Vector2d> &points1,
                               const std::vector<Eigen::Vector2d> &points2,
                               const std::vector<LineSegment> &ls1,
                               const std::vector<LineSegment> &ls2,
                               const bool rot)
      : pts1(points1), pts2(points2), line_segments1(ls1), line_segments2(ls2), pure_rotation(rot) {}

  inline int num_minimal_solvers() const {
    if (pure_rotation) {
      return 3;
    } else {
      return 2;
    }
  }

  void min_sample_sizes(std::vector<std::vector<int>> *min_sample_sizes) const {
    if (pure_rotation) {
      *min_sample_sizes = {{2, 0}, {1, 1}, {0, 2}};
    } else {
      *min_sample_sizes = {{4, 0}, {0, 4}};
    }
  }

  int num_data_types() const { return 2; }
  void num_data(std::vector<int> *num_data) const {
    *num_data = {static_cast<int>(pts1.size()), static_cast<int>(line_segments1.size())};
  }

  void solver_probabilities(std::vector<double> *solver_probabilites) const {
    const float n_samples = pts1.size() + line_segments1.size();
    const float ratio_pts = pts1.size() / n_samples;
    const float ratio_lines = line_segments1.size() / n_samples;
    if (pure_rotation) {
      // Probs should be proportional to the number of features of that type: Ej: 70%p , 30%l
      // pp: 0.7 * 0.7 = 0.49 ; pl: 0.7 * 0.3 + 0.3 * 0.7 = 0.42 ; ll: 0.3 * 0.3 = 0.09
      *solver_probabilites = {ratio_pts * ratio_pts,
                              2 * ratio_pts * ratio_lines,
                              ratio_lines * ratio_lines};
    } else {
      *solver_probabilites = {ratio_pts, ratio_lines};
    }
  }

  int MinimalSolver(const std::vector<std::vector<int>> &sample,
                    const int solver_idx, std::vector<Eigen::Matrix3d> *models) const {
    models->clear();

    std::vector<Eigen::Vector2d> ps1, ps2;
    ps1.resize(sample[0].size());
    ps2.resize(sample[0].size());
    for (size_t k = 0; k < sample[0].size(); ++k) {
      ps1[k] = pts1[sample[0][k]];
      ps2[k] = pts2[sample[0][k]];
    }

    std::vector<LineSegment> ls1, ls2;
    ls1.resize(sample[1].size());
    ls2.resize(sample[1].size());
    for (size_t k = 0; k < sample[1].size(); ++k) {
      ls1[k] = line_segments1[sample[1][k]];
      ls2[k] = line_segments2[sample[1][k]];
    }

    if (pure_rotation) {      
      models->push_back(est_rotation(ps1, ps2, ls1, ls2));
    } else {
      if (solver_idx == 0) {
        // Solve with 4 points
        models->push_back(estimateHomographyPoints(ps1, ps2, false));
      } else {
        // Solve with 4 lines
        models->push_back(estimateHomographyLineSegments(ls1, ls2, false));
      }
    }
    return models->size();
  }

  double EvaluateModelOnPoint(const Eigen::Matrix3d &H, int t, int i) const {
    double res;
    if (t == 0) {
      // point
      if (pure_rotation) {
        res = computeRotationPointResidual(pts1[i], pts2[i], H);
      } else {
        res = computeHomographyPointResidual(pts1[i], pts2[i], H);
      }
    } else {
      // line
      if (pure_rotation) {
        res = computeRotationLineResidual(line_segments1[i], line_segments2[i], H);
      } else {
        res = computeHomographyLineResidual(line_segments1[i], line_segments2[i], H);
      }
    }
    return res * res;
  }

  void LeastSquares(const std::vector<std::vector<int>> &sample,
                    Eigen::Matrix3d *H) const {
    // Collect point lines
    std::vector<Eigen::Vector2d> ps1, ps2;
    ps1.resize(sample[0].size());
    ps2.resize(sample[0].size());
    for (size_t k = 0; k < sample[0].size(); ++k) {
      ps1[k] = pts1[sample[0][k]];
      ps2[k] = pts2[sample[0][k]];
    }
    // Collect inlier lines
    std::vector<LineSegment> ls1, ls2;
    ls1.resize(sample[1].size());
    ls2.resize(sample[1].size());
    for (size_t k = 0; k < sample[1].size(); ++k) {
      ls1[k] = line_segments1[sample[1][k]];
      ls2[k] = line_segments2[sample[1][k]];
    }
    refineHomography(ps1, ps2, ls1, ls2, *H, pure_rotation);
  }

  void get_weights(std::vector<std::vector<double>> &dst_weights) const {
    dst_weights.resize(2);
    // Reserve memory
    dst_weights[0].resize(pts1.size());
    dst_weights[1].resize(line_segments1.size());
    // Uniform distribution for the points
    for (size_t i = 0; i < pts1.size(); i++) {
      dst_weights[0][i] = 1;
    }

    // Distribution proportional to the Sqrt of the length for the lines
    for (size_t i = 0; i < line_segments1.size(); i++) {
      dst_weights[1][i] = std::sqrt((line_segments1[i].p2 - line_segments1[i].p1).norm());
    }
  }

 protected:
  const std::vector<Eigen::Vector2d> &pts1;
  const std::vector<Eigen::Vector2d> &pts2;
  const std::vector<LineSegment> &line_segments1;
  const std::vector<LineSegment> &line_segments2;
  const bool pure_rotation;
};
}
#endif //HOMOGRAPHY_EST_ESTIMATORS_H_
