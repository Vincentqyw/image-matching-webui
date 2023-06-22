#include "hest.h"
#include <ceres/ceres.h>
#include "RansacLib/ransac.h"
#include "RansacLib/hybrid_ransac.h"
#include "Estimators.h"

// Make the library work with any ceres version
#if (CERES_VERSION_MAJOR >= 2) && (CERES_VERSION_MINOR > 0)
#define SET_PARAMETRIZATION(problem, vals, param) (problem).SetManifold((vals), (param))
#define NEW_HEST_PARAMETRIZATION (new ceres::SphereManifold<9>())
#define NEW_ROT_PARAMETRIZATION (new ceres::EigenQuaternionManifold())
#else
#define SET_PARAMETRIZATION(problem, vals, param) (problem).SetParameterization((vals), (param))
#define NEW_HEST_PARAMETRIZATION (new ceres::HomogeneousVectorParameterization(9))
#define NEW_ROT_PARAMETRIZATION (new ceres::EigenQuaternionParameterization())
#endif

namespace hest {

Eigen::Matrix3d estimateHomographyLines(
    const std::vector<Eigen::Vector3d> &lines1,
    const std::vector<Eigen::Vector3d> &lines2) {
    return dlt_homography(lines2, lines1).transpose();
}


Eigen::Matrix3d estimateHomographyLineSegments(
        const std::vector<LineSegment> &line_segments1,
        const std::vector<LineSegment> &line_segments2,
        bool pure_rotation) {
    
    if(pure_rotation) {
        return est_rotation({}, {}, line_segments1, line_segments2);
    } else {
        std::vector<Eigen::Vector3d> lines1;
        std::vector<Eigen::Vector3d> lines2;

        lines1.resize(line_segments1.size());
        lines2.resize(line_segments1.size());

        for(size_t i = 0; i < line_segments1.size(); ++i) {
            lines1[i] = line_segments1[i].p1.homogeneous().cross(line_segments1[i].p2.homogeneous());
            lines2[i] = line_segments2[i].p1.homogeneous().cross(line_segments2[i].p2.homogeneous());

            lines1[i] /= lines1[i].topRows<2>().norm();
            lines2[i] /= lines2[i].topRows<2>().norm();
        }

        return estimateHomographyLines(lines1, lines2);
    }
}

Eigen::Matrix3d estimateHomographyPoints(
    const std::vector<Eigen::Vector2d> &points1,
    const std::vector<Eigen::Vector2d> &points2,
    bool pure_rotation) {

    if(pure_rotation) {
        return est_rotation(points1, points2, {}, {});
    } else {
        std::vector<Eigen::Vector3d> p1;
        std::vector<Eigen::Vector3d> p2;

        p1.resize(points1.size());
        p2.resize(points1.size());

        for(size_t i = 0; i < points1.size(); ++i) {
            p1[i] = points1[i].homogeneous();
            p2[i] = points2[i].homogeneous();
        }
        return dlt_homography(p1,p2);
    }
}

class HomographyLineCost {
public:
    explicit HomographyLineCost(const LineSegment& ls1, const LineSegment& ls2) {
        p1 = ls2.p1;
        p2 = ls2.p2;
        line1 = ls1.p1.homogeneous().cross(ls1.p2.homogeneous());
        line1 /= line1.topRows<2>().norm();
    }

    static ceres::CostFunction* Create(const LineSegment& ls1, const LineSegment& ls2){
        return (new ceres::AutoDiffCostFunction<HomographyLineCost, 2, 9>(
                new HomographyLineCost(ls1,ls2)));
    }

    template <typename T>
    bool operator()(const T* const hvec, T* residuals) const {
      
        Eigen::Map<const Eigen::Matrix<T, 3, 3>> H(hvec);

        // Map endpoints from second image into first
        Eigen::Matrix<T,3,1> z1 = H * p1.homogeneous().cast<T>();
        Eigen::Matrix<T,3,1> z2 = H * p2.homogeneous().cast<T>();
      
        // Compute distance to line in the first image
        residuals[0] = line1.dot(z1.hnormalized().homogeneous());
        residuals[1] = line1.dot(z2.hnormalized().homogeneous());
        return true;
    }

private:
  Eigen::Vector2d p1, p2;
  Eigen::Vector3d line1;
};


class RotationLineCost {
public:
    explicit RotationLineCost(const LineSegment& ls1, const LineSegment& ls2) {
        p1_1 = ls1.p1;
        p1_2 = ls1.p2;

        p2_1 = ls2.p1;
        p2_2 = ls2.p2;

        line1 = ls1.p1.homogeneous().cross(ls1.p2.homogeneous());
        line1 /= line1.topRows<2>().norm();

        line2 = ls2.p1.homogeneous().cross(ls2.p2.homogeneous());
        line2 /= line2.topRows<2>().norm();
    }

    static ceres::CostFunction* Create(const LineSegment& ls1, const LineSegment& ls2){
        return (new ceres::AutoDiffCostFunction<RotationLineCost, 4, 4>(
                new RotationLineCost(ls1,ls2)));
    }

    template <typename T>
    bool operator()(const T* const qvec, T* residuals) const {
        Eigen::Quaternion<T> q;
        q.coeffs() << qvec[0], qvec[1], qvec[2], qvec[3];
        Eigen::Matrix<T, 3, 3> R = q.toRotationMatrix();

        Eigen::Matrix<T,3,1> z2_1 = R * p2_1.homogeneous().cast<T>();
        Eigen::Matrix<T,3,1> z2_2 = R * p2_2.homogeneous().cast<T>();

        Eigen::Matrix<T,3,1> z1_1 = R.transpose() * p1_1.homogeneous().cast<T>();
        Eigen::Matrix<T,3,1> z1_2 = R.transpose() * p1_2.homogeneous().cast<T>();

        // Compute distance to line in the first image
        residuals[0] = line1.dot(z2_1.hnormalized().homogeneous());
        residuals[1] = line1.dot(z2_2.hnormalized().homogeneous());
        residuals[2] = line2.dot(z1_1.hnormalized().homogeneous());
        residuals[3] = line2.dot(z1_2.hnormalized().homogeneous());
        return true;
    }

private:
  Eigen::Vector2d p1_1, p1_2, p2_1, p2_2;
  Eigen::Vector3d line1, line2;
};


class HomographyPointCost {
public:
    explicit HomographyPointCost(const Eigen::Vector2d& point1, const Eigen::Vector2d& point2) : p1(point1), p2(point2) {}

    static ceres::CostFunction* Create(const Eigen::Vector2d& point1, const Eigen::Vector2d& point2){
        return (new ceres::AutoDiffCostFunction<HomographyPointCost, 2, 9>(
                new HomographyPointCost(point1,point2)));
    }

    template <typename T>
    bool operator()(const T* const hvec, T* residuals) const {
      
      Eigen::Map<const Eigen::Matrix<T, 3, 3>> H(hvec);

      Eigen::Matrix<T,3,1> z = H * p2.homogeneous().cast<T>();      
      Eigen::Matrix<T,2,1> zp = z.hnormalized();      

      residuals[0] = zp(0) - T(p1(0));
      residuals[1] = zp(1) - T(p1(1));
      return true;
    }

private:
  Eigen::Vector2d p1, p2;
};

class RotationPointCost {
public:
    explicit RotationPointCost(const Eigen::Vector2d& point1, const Eigen::Vector2d& point2) : p1(point1), p2(point2) {}

    static ceres::CostFunction* Create(const Eigen::Vector2d& point1, const Eigen::Vector2d& point2){
        return (new ceres::AutoDiffCostFunction<RotationPointCost, 4, 4>(
                new RotationPointCost(point1,point2)));
    }

    template <typename T>
    bool operator()(const T* const qvec, T* residuals) const {
        Eigen::Quaternion<T> q;
        q.coeffs() << qvec[0], qvec[1], qvec[2], qvec[3];
        Eigen::Matrix<T, 3, 3> R = q.toRotationMatrix();
        Eigen::Matrix<T,3,1> z1 = R * p2.homogeneous().cast<T>();      
        Eigen::Matrix<T,2,1> zp1 = z1.hnormalized();      

        Eigen::Matrix<T,3,1> z2 = R.transpose() * p1.homogeneous().cast<T>();      
        Eigen::Matrix<T,2,1> zp2 = z2.hnormalized();      

        residuals[0] = zp1(0) - T(p1(0));
        residuals[1] = zp1(1) - T(p1(1));
        residuals[2] = zp2(0) - T(p2(0));
        residuals[3] = zp2(1) - T(p2(1));

        return true;
    }

private:
  Eigen::Vector2d p1, p2;
};


void refineHomography(
    const std::vector<LineSegment> &line_segments1,
    const std::vector<LineSegment> &line_segments2,
    Eigen::Matrix3d &homography,
    bool pure_rotation) {

    if (line_segments1.empty()) {
      return;
    }

    Eigen::Quaterniond q;
    ceres::Problem problem;

    if(pure_rotation) {
        q = Eigen::Quaterniond(homography);
        for(size_t k = 0; k < line_segments1.size(); ++k) {
            ceres::CostFunction *cost_function = RotationLineCost::Create(line_segments1[k], line_segments2[k]);
            problem.AddResidualBlock(cost_function, nullptr, q.coeffs().data());
        }
        SET_PARAMETRIZATION(problem, q.coeffs().data(), NEW_ROT_PARAMETRIZATION);
    } else {
        for(size_t k = 0; k < line_segments1.size(); ++k) {
            ceres::CostFunction *cost_function = HomographyLineCost::Create(line_segments1[k], line_segments2[k]);
            problem.AddResidualBlock(cost_function, nullptr, homography.data());
        }
        SET_PARAMETRIZATION(problem, homography.data(), NEW_HEST_PARAMETRIZATION);
    }
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    //options.logging_type = ceres::LoggingType::SILENT;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.function_tolerance = 1e-10;
    options.gradient_tolerance = 1e-10;
    options.parameter_tolerance = 1e-10;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if(pure_rotation) {
        homography = q.toRotationMatrix();
    }
}


void refineHomography(
    const std::vector<Eigen::Vector2d> &pts1,
    const std::vector<Eigen::Vector2d> &pts2,
    Eigen::Matrix3d &homography,    
    bool pure_rotation) {

    if (pts1.empty()) {
      return;
    }

    Eigen::Quaterniond q;
    ceres::Problem problem;
    if(pure_rotation) {
        q = Eigen::Quaterniond(homography);
        for(size_t k = 0; k < pts1.size(); ++k) {
            ceres::CostFunction *cost_function = RotationPointCost::Create(pts1[k], pts2[k]);
            problem.AddResidualBlock(cost_function, nullptr, q.coeffs().data());
        }
        SET_PARAMETRIZATION(problem, q.coeffs().data(), NEW_ROT_PARAMETRIZATION);
    } else {
        for(size_t k = 0; k < pts1.size(); ++k) {
            ceres::CostFunction *cost_function = HomographyPointCost::Create(pts1[k], pts2[k]);
            problem.AddResidualBlock(cost_function, nullptr, homography.data());
        }
        SET_PARAMETRIZATION(problem, homography.data(), NEW_HEST_PARAMETRIZATION);
    }

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    //options.logging_type = ceres::LoggingType::SILENT;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.function_tolerance = 1e-10;
    options.gradient_tolerance = 1e-10;
    options.parameter_tolerance = 1e-10;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if(pure_rotation) {
        homography = q.toRotationMatrix();
    }
}


void refineHomography(
    const std::vector<Eigen::Vector2d> &pts1,
    const std::vector<Eigen::Vector2d> &pts2,
    const std::vector<LineSegment> &line_segments1,
    const std::vector<LineSegment> &line_segments2,
    Eigen::Matrix3d &homography,
    bool pure_rotation) {
    const size_t n_points_and_lines = pts1.size()  + line_segments1.size();
    if (n_points_and_lines < 4) {
      std::cerr << "Error: Not enough correspondences to refine homography" << std::endl;
      return;
    }

    Eigen::Quaterniond q;
    ceres::Problem problem;
    if(pure_rotation) {
        q = Eigen::Quaterniond(homography);
        for(size_t k = 0; k < pts1.size(); ++k) {
            ceres::CostFunction *cost_function = RotationPointCost::Create(pts1[k], pts2[k]);
            problem.AddResidualBlock(cost_function, nullptr, q.coeffs().data());
        }
        for(size_t k = 0; k < line_segments1.size(); ++k) {
            ceres::CostFunction *cost_function = RotationLineCost::Create(line_segments1[k], line_segments2[k]);
            problem.AddResidualBlock(cost_function, nullptr, q.coeffs().data());
        }
        SET_PARAMETRIZATION(problem, q.coeffs().data(), NEW_ROT_PARAMETRIZATION);
    } else {
        for(size_t k = 0; k < pts1.size(); ++k) {
            ceres::CostFunction *cost_function = HomographyPointCost::Create(pts1[k], pts2[k]);
            problem.AddResidualBlock(cost_function, nullptr, homography.data());
        }
        for(size_t k = 0; k < line_segments1.size(); ++k) {
            ceres::CostFunction *cost_function = HomographyLineCost::Create(line_segments1[k], line_segments2[k]);
            problem.AddResidualBlock(cost_function, nullptr, homography.data());            
        }
        SET_PARAMETRIZATION(problem, homography.data(), NEW_HEST_PARAMETRIZATION);
    }
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    //options.logging_type = ceres::LoggingType::SILENT;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.function_tolerance = 1e-10;
    options.gradient_tolerance = 1e-10;
    options.parameter_tolerance = 1e-10;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if(pure_rotation) {
        homography = q.toRotationMatrix();
    }
}


Eigen::Matrix3d ransacLineHomography(
    const std::vector<LineSegment> &line_segments1,
    const std::vector<LineSegment> &line_segments2,
    const double tol_px, bool pure_rotation, std::vector<int> *inlier_ind) {

    
    ransac_lib::LORansacOptions options;
    options.min_num_iterations_ = 1000u;
    options.max_num_iterations_ = 10000u;
    options.final_least_squares_ = true;
    options.min_sample_multiplicator_ = 100;
    options.non_min_sample_multiplier_ = 100;
    options.squared_inlier_threshold_ = tol_px * tol_px;
    LineHomographyEstimator solver(line_segments1, line_segments2, pure_rotation);
    
    ransac_lib::LocallyOptimizedMSAC<Eigen::Matrix3d,std::vector<Eigen::Matrix3d>,LineHomographyEstimator> lomsac;
    ransac_lib::RansacStatistics ransac_stats;
    Eigen::Matrix3d H;
    lomsac.EstimateModel(options,solver, &H, &ransac_stats);

    if(inlier_ind != nullptr) {
        *inlier_ind = ransac_stats.inlier_indices;
    }

    return H;
}


Eigen::Matrix3d ransacPointHomography(
    const std::vector<Eigen::Vector2d> &pts1,
    const std::vector<Eigen::Vector2d> &pts2,
    const double tol_px, const bool pure_rotation, std::vector<int> *inlier_ind) {

    
    ransac_lib::LORansacOptions options;
    options.min_num_iterations_ = 1000u;
    options.max_num_iterations_ = 10000u;
    options.final_least_squares_ = true;
    options.min_sample_multiplicator_ = 100;
    options.non_min_sample_multiplier_ = 100;
    options.squared_inlier_threshold_ = tol_px * tol_px;
    PointHomographyEstimator solver(pts1, pts2, pure_rotation);
    
    ransac_lib::LocallyOptimizedMSAC<Eigen::Matrix3d,std::vector<Eigen::Matrix3d>,PointHomographyEstimator> lomsac;
    ransac_lib::RansacStatistics ransac_stats;
    Eigen::Matrix3d H;
    lomsac.EstimateModel(options, solver, &H, &ransac_stats);

    if(inlier_ind != nullptr) {
        *inlier_ind = ransac_stats.inlier_indices;
    }

    return H;
}

Eigen::Matrix3d ransacPointLineHomography(
        const std::vector<Eigen::Vector2d> &pts1,
        const std::vector<Eigen::Vector2d> &pts2,
        const std::vector<LineSegment> &line_segments1,
        const std::vector<LineSegment> &line_segments2,
        double tol_px, bool pure_rotation,
        std::vector<int> *inlier_pts_ind,
        std::vector<int> *inlier_lin_ind) {

    ransac_lib::HybridLORansacOptions  options;
    options.min_num_iterations_ = 10000u;
    options.max_num_iterations_ = 100000u;
    options.max_num_iterations_per_solver_ = 100000u;
    options.squared_inlier_thresholds_ = {tol_px * tol_px, tol_px * tol_px};
    options.data_type_weights_ = {1.0, 1.0};
    options.final_least_squares_ = true;
    options.min_sample_multiplicator_ = 100;

    PointLineHomographyEstimator solver(pts1, pts2, line_segments1, line_segments2, pure_rotation);
//    ransac_lib::HybridLocallyOptimizedMSAC<
//            Eigen::Matrix3d,std::vector<Eigen::Matrix3d>,PointLineHomographyEstimator> lomsac;
    ransac_lib::HybridLocallyOptimizedMSAC<
        Eigen::Matrix3d,std::vector<Eigen::Matrix3d>,PointLineHomographyEstimator,
        ransac_lib::HybridBiasedSampling<PointLineHomographyEstimator>> lomsac;
    ransac_lib::HybridRansacStatistics ransac_stats;
    Eigen::Matrix3d H;
    lomsac.EstimateModel(options, solver, &H, &ransac_stats);

    if(inlier_pts_ind != nullptr) {
      inlier_pts_ind->resize(ransac_stats.inlier_indices[0].size());
      for (size_t i = 0; i < inlier_pts_ind->size(); i++) {
        (*inlier_pts_ind)[i] = ransac_stats.inlier_indices[0][i];
      }
    }
    if (inlier_lin_ind != nullptr) {
      inlier_lin_ind->resize(ransac_stats.inlier_indices[1].size());
      for (size_t i = 0; i < inlier_lin_ind->size(); i++) {
        (*inlier_lin_ind)[i] = ransac_stats.inlier_indices[1][i];
      }
    }

    return H;
}

}
