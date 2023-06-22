#include <iostream>
#include "Estimators.h"

namespace hest{

Eigen::Matrix3d dlt_homography(
    const std::vector<Eigen::Vector3d> &p1,
    const std::vector<Eigen::Vector3d> &p2) {

  Eigen::Matrix<double, Eigen::Dynamic, 9> A(2 * p1.size(), 9);
  for (size_t i = 0; i < p1.size(); ++i) {

    // p1(1:2) * (h3' * p2) = p1(3) * [h1';h2'] * p2

    // [p1(3)*p2' 0 0 0 -p1(1)*p2';
    //  0 0 0 p1(3)*p2' -p1(2)*p2'] * [h1;h2;h3] = 0

    A.row(2 * i) << p1[i](2) * p2[i].transpose(), 0.0, 0.0, 0.0, -p1[i](0) * p2[i].transpose();
    A.row(2 * i + 1) << 0.0, 0.0, 0.0, p1[i](2) * p2[i].transpose(), -p1[i](1) * p2[i].transpose();
  }

  Eigen::JacobiSVD<decltype(A)> svd(A, Eigen::ComputeFullV);

  Eigen::Matrix<double, 9, 1> h = svd.matrixV().rightCols<1>();

  Eigen::Matrix3d H;
  H.row(0) = h.block<3, 1>(0, 0).transpose();
  H.row(1) = h.block<3, 1>(3, 0).transpose();
  H.row(2) = h.block<3, 1>(6, 0).transpose();
  return H;
}

Eigen::Matrix3d proj_rot(const Eigen::Matrix3d &A, bool pos_det = true) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  // Find the rotation
  Eigen::Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
  if(R.determinant() < 0 && pos_det) {
    Eigen::Matrix3d D;
    D.setIdentity();
    D(2,2) = -1;
    R = svd.matrixU() * D * svd.matrixV().transpose();
  }
  return R;
}

Eigen::Matrix3d est_rotation(
    const std::vector<Eigen::Vector2d> &p1,
    const std::vector<Eigen::Vector2d> &p2,
    const std::vector<LineSegment> &ls1,
    const std::vector<LineSegment> &ls2) {
    
  Eigen::Matrix3d A;
  A.setZero();
  for(size_t i = 0; i < p1.size(); ++i) {
    Eigen::Vector3d x1 = p1[i].homogeneous().normalized();
    Eigen::Vector3d x2 = p2[i].homogeneous().normalized();
    A += x1 * x2.transpose();
  }

  for(size_t i = 0; i < ls1.size(); ++i) {
    // Compute the line plane normals (LPN) of the line segments
    Eigen::Vector3d n1 = ls1[i].p1.homogeneous().cross(ls1[i].p2.homogeneous()).normalized();
    Eigen::Vector3d n2 = ls2[i].p1.homogeneous().cross(ls2[i].p2.homogeneous()).normalized();

    // For perspective images with fov < 180, the normals should have a similar direction
    if(n2.dot(n1) < 0) {
      // If not, that means that the endpoint are flipped.
      // If so, it is safe to invert the line plane normal direction
      n2 = -n2;
    }
    A += n1 * n2.transpose();      
  }

  return proj_rot(A);
}

double computeHomographyLineResidual(
    const LineSegment &line_segment1,
    const LineSegment &line_segment2,
    const Eigen::Matrix3d &H) {

  Eigen::Vector3d p1 = H * line_segment2.p1.homogeneous();
  Eigen::Vector3d p2 = H * line_segment2.p2.homogeneous();

  Eigen::Vector3d line = p1.normalized().cross(p2.normalized());
  line = line / line.topRows<2>().norm();

  return std::abs(line.dot(line_segment1.p1.homogeneous())) +
      std::abs(line.dot(line_segment1.p2.homogeneous()));
}

// Maps the endpoints and computes the point-to-line error in the first image
double computeRotationLineResidual(
    const LineSegment &line_segment1,
    const LineSegment &line_segment2,
    const Eigen::Matrix3d &R) {

  Eigen::Vector3d p1_1 = R * line_segment2.p1.homogeneous();
  Eigen::Vector3d p1_2 = R * line_segment2.p2.homogeneous();
  Eigen::Vector3d p2_1 = R.transpose() * line_segment1.p1.homogeneous();
  Eigen::Vector3d p2_2 = R.transpose() * line_segment1.p2.homogeneous();

  Eigen::Vector3d line1 = p1_1.normalized().cross(p1_2.normalized());
  line1 = line1 / line1.topRows<2>().norm();

  Eigen::Vector3d line2 = p2_1.normalized().cross(p2_2.normalized());
  line2 = line2 / line2.topRows<2>().norm();

  const double e1 = std::abs(line1.dot(line_segment1.p1.homogeneous()));
  const double e2 = std::abs(line1.dot(line_segment1.p2.homogeneous()));
  const double e3 = std::abs(line2.dot(line_segment2.p1.homogeneous()));
  const double e4 = std::abs(line2.dot(line_segment2.p2.homogeneous()));

  return std::sqrt(e1 * e1 + e2 * e2 + e3 * e3 + e4 * e4);
}

double computeHomographyPointResidual(
    const Eigen::Vector2d &p1,
    const Eigen::Vector2d &p2,
    const Eigen::Matrix3d &H) {

  Eigen::Vector3d z = H * p2.homogeneous();
  return (p1 - z.hnormalized()).norm();
}

double computeRotationPointResidual(
    const Eigen::Vector2d &p1,
    const Eigen::Vector2d &p2,
    const Eigen::Matrix3d &R) {

  Eigen::Vector3d z1 = R * p2.homogeneous();
  Eigen::Vector3d z2 = R.transpose() * p1.homogeneous();

  const double err1 = (p1 - z1.hnormalized()).norm();
  const double err2 = (p2 - z2.hnormalized()).norm();

  return std::sqrt(err1 * err1 + err2 * err2);
}

}