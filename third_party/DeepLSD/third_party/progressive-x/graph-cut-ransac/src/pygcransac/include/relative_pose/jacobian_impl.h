#ifndef POSELIB_JACOBIAN_IMPL_H_
#define POSELIB_JACOBIAN_IMPL_H_

#include "essential.h"

namespace pose_lib {

template <typename CameraModel, typename LossFunction>
class CameraJacobianAccumulator {
  public:
    CameraJacobianAccumulator(
        const std::vector<Eigen::Vector2d> &points2D,
        const std::vector<Eigen::Vector3d> &points3D,
        const Camera &cam, const LossFunction &loss) : x(points2D), X(points3D), camera(cam), loss_fn(loss) {}

    double residual(const CameraPose &pose) const {
        double cost = 0;
        for (int i = 0; i < x.size(); ++i) {
            const Eigen::Vector3d Z = pose.R * X[i] + pose.t;
            const double inv_z = 1.0 / Z(2);
            Eigen::Vector2d p(Z(0) * inv_z, Z(1) * inv_z);
            CameraModel::project(camera.params, p, &p);
            const double r0 = p(0) - x[i](0);
            const double r1 = p(1) - x[i](1);
            const double r_squared = r0 * r0 + r1 * r1;
            cost += loss_fn.loss(r_squared);
        }
        return cost;
    }

    // computes J.transpose() * J and J.transpose() * res
    // Only computes the lower half of JtJ
    void accumulate(const CameraPose &pose, Eigen::Matrix<double, 6, 6> &JtJ, Eigen::Matrix<double, 6, 1> &Jtr) const {

        Eigen::Matrix2d Jcam;
        Jcam.setIdentity(); // we initialize to identity here (this is for the calibrated case)
        for (int i = 0; i < x.size(); ++i) {
            const Eigen::Vector3d Z = pose.R * X[i] + pose.t;
            const Eigen::Vector2d z = Z.hnormalized();

            // Project with intrinsics
            Eigen::Vector2d zp = z;
            CameraModel::project_with_jac(camera.params, z, &zp, &Jcam);

            // Setup residual
            Eigen::Vector2d r = zp - x[i];
            const double r_squared = r.squaredNorm();
            const double weight = loss_fn.weight(r_squared) / static_cast<double>(x.size());

            if (weight == 0.0) {
                continue;
            }

            // Compute jacobian w.r.t. Z (times R)
            Eigen::Matrix<double, 2, 3> dZ;
            dZ.block<2, 2>(0, 0) = Jcam;
            dZ.col(2) = -Jcam * z;
            dZ *= 1.0 / Z(2);
            dZ *= pose.R;

            const double X0 = X[i](0);
            const double X1 = X[i](1);
            const double X2 = X[i](2);
            const double dZtdZ_0_0 = weight * dZ.col(0).dot(dZ.col(0));
            const double dZtdZ_1_0 = weight * dZ.col(1).dot(dZ.col(0));
            const double dZtdZ_1_1 = weight * dZ.col(1).dot(dZ.col(1));
            const double dZtdZ_2_0 = weight * dZ.col(2).dot(dZ.col(0));
            const double dZtdZ_2_1 = weight * dZ.col(2).dot(dZ.col(1));
            const double dZtdZ_2_2 = weight * dZ.col(2).dot(dZ.col(2));
            JtJ(0, 0) += X2 * (X2 * dZtdZ_1_1 - X1 * dZtdZ_2_1) + X1 * (X1 * dZtdZ_2_2 - X2 * dZtdZ_2_1);
            JtJ(1, 0) += -X2 * (X2 * dZtdZ_1_0 - X0 * dZtdZ_2_1) - X1 * (X0 * dZtdZ_2_2 - X2 * dZtdZ_2_0);
            JtJ(2, 0) += X1 * (X0 * dZtdZ_2_1 - X1 * dZtdZ_2_0) - X2 * (X0 * dZtdZ_1_1 - X1 * dZtdZ_1_0);
            JtJ(3, 0) += X1 * dZtdZ_2_0 - X2 * dZtdZ_1_0;
            JtJ(4, 0) += X1 * dZtdZ_2_1 - X2 * dZtdZ_1_1;
            JtJ(5, 0) += X1 * dZtdZ_2_2 - X2 * dZtdZ_2_1;
            JtJ(1, 1) += X2 * (X2 * dZtdZ_0_0 - X0 * dZtdZ_2_0) + X0 * (X0 * dZtdZ_2_2 - X2 * dZtdZ_2_0);
            JtJ(2, 1) += -X2 * (X1 * dZtdZ_0_0 - X0 * dZtdZ_1_0) - X0 * (X0 * dZtdZ_2_1 - X1 * dZtdZ_2_0);
            JtJ(3, 1) += X2 * dZtdZ_0_0 - X0 * dZtdZ_2_0;
            JtJ(4, 1) += X2 * dZtdZ_1_0 - X0 * dZtdZ_2_1;
            JtJ(5, 1) += X2 * dZtdZ_2_0 - X0 * dZtdZ_2_2;
            JtJ(2, 2) += X1 * (X1 * dZtdZ_0_0 - X0 * dZtdZ_1_0) + X0 * (X0 * dZtdZ_1_1 - X1 * dZtdZ_1_0);
            JtJ(3, 2) += X0 * dZtdZ_1_0 - X1 * dZtdZ_0_0;
            JtJ(4, 2) += X0 * dZtdZ_1_1 - X1 * dZtdZ_1_0;
            JtJ(5, 2) += X0 * dZtdZ_2_1 - X1 * dZtdZ_2_0;
            JtJ(3, 3) += dZtdZ_0_0;
            JtJ(4, 3) += dZtdZ_1_0;
            JtJ(5, 3) += dZtdZ_2_0;
            JtJ(4, 4) += dZtdZ_1_1;
            JtJ(5, 4) += dZtdZ_2_1;
            JtJ(5, 5) += dZtdZ_2_2;
            r *= weight;
            Jtr(0) += (r(0) * (X1 * dZ(0, 2) - X2 * dZ(0, 1)) + r(1) * (X1 * dZ(1, 2) - X2 * dZ(1, 1)));
            Jtr(1) += (-r(0) * (X0 * dZ(0, 2) - X2 * dZ(0, 0)) - r(1) * (X0 * dZ(1, 2) - X2 * dZ(1, 0)));
            Jtr(2) += (r(0) * (X0 * dZ(0, 1) - X1 * dZ(0, 0)) + r(1) * (X0 * dZ(1, 1) - X1 * dZ(1, 0)));
            Jtr(3) += (dZ(0, 0) * r(0) + dZ(1, 0) * r(1));
            Jtr(4) += (dZ(0, 1) * r(0) + dZ(1, 1) * r(1));
            Jtr(5) += (dZ(0, 2) * r(0) + dZ(1, 2) * r(1));
        }
    }

  private:
    const std::vector<Eigen::Vector2d> &x;
    const std::vector<Eigen::Vector3d> &X;
    const Camera &camera;
    const LossFunction &loss_fn;
};

template <typename LossFunction>
class RelativePoseJacobianAccumulator {
  public:
    RelativePoseJacobianAccumulator(
        const cv::Mat& correspondences_,
        const size_t* sample_,
        const size_t& sample_size_,
        const LossFunction &l,
        const double *w = nullptr) : 
            correspondences(&correspondences_), 
            sample(sample_),
            sample_size(sample_size_),
            loss_fn(l),
            weights(w) {}

    double residual(const CameraPose &pose) const {
        Eigen::Matrix3d E;
        essential_from_motion(pose, &E);

        Eigen::Vector2d pt1, pt2;
        double cost = 0.0;
        for (size_t k = 0; k < sample_size; ++k) 
        {
            const size_t& point_idx = sample[k];

            pt1 << correspondences->at<double>(point_idx, 0), correspondences->at<double>(point_idx, 1);
            pt2 << correspondences->at<double>(point_idx, 2), correspondences->at<double>(point_idx, 3);

            double C = pt2.homogeneous().dot(E * pt1.homogeneous());
            double nJc_sq = (E.block<2, 3>(0, 0) * pt1.homogeneous()).squaredNorm() +
                            (E.block<3, 2>(0, 0).transpose() * pt2.homogeneous()).squaredNorm();

            double r2 = (C * C) / nJc_sq;
            if (weights == nullptr)
                cost += loss_fn.loss(r2);
            else
                cost += weights[k] * loss_fn.loss(r2);
        }

        return cost;
    }

    void accumulate(const CameraPose &pose, Eigen::Matrix<double, 5, 5> &JtJ, Eigen::Matrix<double, 5, 1> &Jtr, Eigen::Matrix<double, 3, 2> &tangent_basis) const {
        // We start by setting up a basis for the updates in the translation (orthogonal to t)
        // We find the minimum element of t and cross product with the corresponding basis vector.
        // (this ensures that the first cross product is not close to the zero vector)
        if (std::abs(pose.t.x()) < std::abs(pose.t.y())) {
            // x < y
            if (std::abs(pose.t.x()) < std::abs(pose.t.z())) {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitX()).normalized();
            } else {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        } else {
            // x > y
            if (std::abs(pose.t.y()) < std::abs(pose.t.z())) {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitY()).normalized();
            } else {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        }
        tangent_basis.col(1) = tangent_basis.col(0).cross(pose.t).normalized();

        Eigen::Matrix3d E;
        essential_from_motion(pose, &E);

        // Matrices contain the jacobians of E w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dR;
        Eigen::Matrix<double, 9, 2> dt;

        // Each column is vec(E*skew(e_k)) where e_k is k:th basis vector
        dR.block<3, 1>(0, 0).setZero();
        dR.block<3, 1>(0, 1) = -E.col(2);
        dR.block<3, 1>(0, 2) = E.col(1);
        dR.block<3, 1>(3, 0) = E.col(2);
        dR.block<3, 1>(3, 1).setZero();
        dR.block<3, 1>(3, 2) = -E.col(0);
        dR.block<3, 1>(6, 0) = -E.col(1);
        dR.block<3, 1>(6, 1) = E.col(0);
        dR.block<3, 1>(6, 2).setZero();

        // Each column is vec(skew(tangent_basis[k])*R)
        dt.block<3, 1>(0, 0) = tangent_basis.col(0).cross(pose.R.col(0));
        dt.block<3, 1>(0, 1) = tangent_basis.col(1).cross(pose.R.col(0));
        dt.block<3, 1>(3, 0) = tangent_basis.col(0).cross(pose.R.col(1));
        dt.block<3, 1>(3, 1) = tangent_basis.col(1).cross(pose.R.col(1));
        dt.block<3, 1>(6, 0) = tangent_basis.col(0).cross(pose.R.col(2));
        dt.block<3, 1>(6, 1) = tangent_basis.col(1).cross(pose.R.col(2));

        for (size_t k = 0; k < sample_size; ++k) 
        {
            const size_t& point_idx = sample[k];

            Eigen::Vector2d pt1, pt2;
            pt1 << correspondences->at<double>(point_idx, 0), correspondences->at<double>(point_idx, 1);
            pt2 << correspondences->at<double>(point_idx, 2), correspondences->at<double>(point_idx, 3);

            double C = pt2.homogeneous().dot(E * pt1.homogeneous());

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << E.block<3, 2>(0, 0).transpose() * pt2.homogeneous(), E.block<2, 3>(0, 0) * pt1.homogeneous();
            const double nJ_C = J_C.norm();
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            // Compute weight from robust loss function (used in the IRLS)
            double weight = loss_fn.weight(r * r) / sample_size;
            if (weights != nullptr)
                weight = weights[k] * weight;

            if(weight == 0.0)
                continue;

            // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dF;
            dF << pt1(0) * pt2(0), pt1(0) * pt2(1), pt1(0), pt1(1) * pt2(0), pt1(1) * pt2(1), pt1(1), pt2(0), pt2(1), 1.0;
            const double s = C * inv_nJ_C * inv_nJ_C;
            dF(0) -= s * (J_C(2) * pt1(0) + J_C(0) * pt2(0));
            dF(1) -= s * (J_C(3) * pt1(0) + J_C(0) * pt2(1));
            dF(2) -= s * (J_C(0));
            dF(3) -= s * (J_C(2) * pt1(1) + J_C(1) * pt2(0));
            dF(4) -= s * (J_C(3) * pt1(1) + J_C(1) * pt2(1));
            dF(5) -= s * (J_C(1));
            dF(6) -= s * (J_C(2));
            dF(7) -= s * (J_C(3));
            dF *= inv_nJ_C;

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, 5> J;
            J.block<1, 3>(0, 0) = dF * dR;
            J.block<1, 2>(0, 3) = dF * dt;

            // Accumulate into JtJ and Jtr
            Jtr += weight * C * inv_nJ_C * J.transpose();
            JtJ(0, 0) += weight * (J(0) * J(0));
            JtJ(1, 0) += weight * (J(1) * J(0));
            JtJ(1, 1) += weight * (J(1) * J(1));
            JtJ(2, 0) += weight * (J(2) * J(0));
            JtJ(2, 1) += weight * (J(2) * J(1));
            JtJ(2, 2) += weight * (J(2) * J(2));
            JtJ(3, 0) += weight * (J(3) * J(0));
            JtJ(3, 1) += weight * (J(3) * J(1));
            JtJ(3, 2) += weight * (J(3) * J(2));
            JtJ(3, 3) += weight * (J(3) * J(3));
            JtJ(4, 0) += weight * (J(4) * J(0));
            JtJ(4, 1) += weight * (J(4) * J(1));
            JtJ(4, 2) += weight * (J(4) * J(2));
            JtJ(4, 3) += weight * (J(4) * J(3));
            JtJ(4, 4) += weight * (J(4) * J(4));
        }
    }

    private:
        const cv::Mat* correspondences;
        const size_t* sample;
        const size_t sample_size;

        const LossFunction &loss_fn;
        const double *weights;
};

// This is the SVD factorization proposed by Bartoli and Sturm in
// Non-Linear Estimation of the Fundamental Matrix With Minimal Parameters, PAMI 2004
// Though we do different updates (lie vs the euler angles used in the original paper)
struct FactorizedFundamentalMatrix {
    FactorizedFundamentalMatrix() {}
    FactorizedFundamentalMatrix(const Eigen::Matrix3d &F) {
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeFullV | Eigen::ComputeFullU);
        U = svd.matrixU();
        V = svd.matrixV();
        Eigen::Vector3d s = svd.singularValues();
        sigma = s(1) / s(0);
    }
    Eigen::Matrix3d F() const {
        return U.col(0) * V.col(0).transpose() + sigma * U.col(1) * V.col(1).transpose();
    }

    Eigen::Matrix3d U, V;
    double sigma;
};

template <typename LossFunction>
class FundamentalJacobianAccumulator {
  public:
    FundamentalJacobianAccumulator(
        const cv::Mat& correspondences_,
        const size_t* sample_,
        const size_t& sample_size_,
        const LossFunction &l,
        const double *w = nullptr)  : 
            correspondences(&correspondences_), 
            sample(sample_),
            sample_size(sample_size_),
            loss_fn(l),
            weights(w) {}

    double residual(const FactorizedFundamentalMatrix &FF) const {
        Eigen::Matrix3d F = FF.F();

        Eigen::Vector2d pt1, pt2;
        double cost = 0.0;
        for (size_t k = 0; k < sample_size; ++k) 
        {
            size_t point_idx;
            if (sample == nullptr)
                point_idx = k;
            else
                point_idx = sample[k];

            pt1 << correspondences->at<double>(point_idx, 0), correspondences->at<double>(point_idx, 1);
            pt2 << correspondences->at<double>(point_idx, 2), correspondences->at<double>(point_idx, 3);
            
            double C = pt2.homogeneous().dot(F * pt1.homogeneous());
            double nJc_sq = (F.block<2, 3>(0, 0) * pt1.homogeneous()).squaredNorm() +
                            (F.block<3, 2>(0, 0).transpose() * pt2.homogeneous()).squaredNorm();

            double r2 = (C * C) / nJc_sq;
            if (weights == nullptr)
                cost += loss_fn.loss(r2);
            else
                cost += weights[k] * loss_fn.loss(r2);
        }

        return cost;
    }

    void accumulate(const FactorizedFundamentalMatrix &FF, Eigen::Matrix<double, 7, 7> &JtJ, Eigen::Matrix<double, 7, 1> &Jtr) const {

        Eigen::Matrix3d F = FF.F();

        // Matrices contain the jacobians of F w.r.t. the factorized fundamental matrix (U,V,sigma)
        Eigen::Matrix3d d_sigma = FF.U.col(1) * FF.V.col(1).transpose();
        Eigen::Matrix<double, 9, 7> dF_dparams;
        dF_dparams << 0, F(2, 0), -F(1, 0), 0, F(0, 2), -F(0, 1), d_sigma(0, 0),
            -F(2, 0), 0, F(0, 0), 0, F(1, 2), -F(1, 1), d_sigma(1, 0),
            F(1, 0), -F(0, 0), 0, 0, F(2, 2), -F(2, 1), d_sigma(2, 0),
            0, F(2, 1), -F(1, 1), -F(0, 2), 0, F(0, 0), d_sigma(0, 1),
            -F(2, 1), 0, F(0, 1), -F(1, 2), 0, F(1, 0), d_sigma(1, 1),
            F(1, 1), -F(0, 1), 0, -F(2, 2), 0, F(2, 0), d_sigma(2, 1),
            0, F(2, 2), -F(1, 2), F(0, 1), -F(0, 0), 0, d_sigma(0, 2),
            -F(2, 2), 0, F(0, 2), F(1, 1), -F(1, 0), 0, d_sigma(1, 2),
            F(1, 2), -F(0, 2), 0, F(2, 1), -F(2, 0), 0, d_sigma(2, 2);

        for (size_t k = 0; k < sample_size; ++k) 
        {
            size_t point_idx;
            if (sample == nullptr)
                point_idx = k;
            else
                point_idx = sample[k];

            Eigen::Vector2d pt1, pt2;
            pt1 << correspondences->at<double>(point_idx, 0), correspondences->at<double>(point_idx, 1);
            pt2 << correspondences->at<double>(point_idx, 2), correspondences->at<double>(point_idx, 3);

            double C = pt2.homogeneous().dot(F * pt1.homogeneous());

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << F.block<3, 2>(0, 0).transpose() * pt2.homogeneous(), F.block<2, 3>(0, 0) * pt1.homogeneous();
            const double nJ_C = J_C.norm();
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            // Compute weight from robust loss function (used in the IRLS)
            double weight = loss_fn.weight(r * r) / sample_size;
            
            // Multiplying by the provided weights if they are available 
            if (weights != nullptr)
                weight = weights[k] * weight;
            if (weight == 0.0)
                continue;

            // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dF;
            dF << pt1(0) * pt2(0), pt1(0) * pt2(1), pt1(0), pt1(1) * pt2(0), pt1(1) * pt2(1), pt1(1), pt2(0), pt2(1), 1.0;
            const double s = C * inv_nJ_C * inv_nJ_C;
            dF(0) -= s * (J_C(2) * pt1(0) + J_C(0) * pt2(0));
            dF(1) -= s * (J_C(3) * pt1(0) + J_C(0) * pt2(1));
            dF(2) -= s * (J_C(0));
            dF(3) -= s * (J_C(2) * pt1(1) + J_C(1) * pt2(0));
            dF(4) -= s * (J_C(3) * pt1(1) + J_C(1) * pt2(1));
            dF(5) -= s * (J_C(1));
            dF(6) -= s * (J_C(2));
            dF(7) -= s * (J_C(3));
            dF *= inv_nJ_C;

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, 7> J = dF * dF_dparams;

            // Accumulate into JtJ and Jtr
            Jtr += weight * C * inv_nJ_C * J.transpose();
            for (size_t i = 0; i < 7; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    JtJ(i, j) += weight * (J(i) * J(j));
                }
            }
        }
    }

  private:
    const cv::Mat* correspondences;
    const size_t* sample;
    const size_t sample_size;

    const LossFunction &loss_fn;
    const double *weights;
};

} // namespace pose_lib

#endif