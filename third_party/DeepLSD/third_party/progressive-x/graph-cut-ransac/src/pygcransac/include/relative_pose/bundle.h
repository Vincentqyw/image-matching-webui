#ifndef POSELIB_BUNDLE_H_
#define POSELIB_BUNDLE_H_

#include "colmap_models.h"
#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace pose_lib {

struct CameraPose {
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    double alpha = 1.0; // either focal length or scale
};

typedef std::vector<CameraPose> CameraPoseVector;

struct BundleOptions {
    size_t max_iterations = 100;
    enum LossType {
        TRIVIAL,
        TRUNCATED,
        HUBER,
        CAUCHY
    } loss_type = LossType::CAUCHY;
    double loss_scale = 1.0;
    double gradient_tol = 1e-8;
    double step_tol = 1e-8;
    double initial_lambda = 1e-3;
};

// Minimizes reprojection error. Assumes identity intrinsics (calibrated camera)
// Returns number of iterations.
int bundle_adjust(const std::vector<Eigen::Vector2d> &x,
                  const std::vector<Eigen::Vector3d> &X,
                  CameraPose *pose,
                  const BundleOptions &opt = BundleOptions());

// Uses intrinsic calibration from Camera (see colmap_models.h)
// Slightly slower than bundle_adjust above
int bundle_adjust(const std::vector<Eigen::Vector2d> &x,
                  const std::vector<Eigen::Vector3d> &X,
                  const Camera &camera,
                  CameraPose *pose,
                  const BundleOptions &opt = BundleOptions());

// Relative pose refinement. Minimizes Sampson error error. Assumes identity intrinsics (calibrated camera)
// Returns number of iterations.
int refine_relpose(const cv::Mat &correspondences_,
                   const size_t *sample_,
                   const size_t &sample_size_,
                   CameraPose *pose,
                   const BundleOptions &opt = BundleOptions(),
                   const double *weights = nullptr);

// Fundamental matrix refinement. Minimizes Sampson error error.
// Returns number of iterations.
int refine_fundamental(const cv::Mat &correspondences_,
                       const size_t *sample_,
                       const size_t &sample_size_,
                       Eigen::Matrix3d *pose,
                       const BundleOptions &opt = BundleOptions(),
                       const double *weights = nullptr);

} // namespace pose_lib

#endif