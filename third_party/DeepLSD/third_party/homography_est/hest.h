
#ifndef HEST_H
#define HEST_H

#include <Eigen/Dense>

namespace hest {

    struct LineSegment {
        LineSegment() {}
        LineSegment(const Eigen::Vector2d &p1, const Eigen::Vector2d &p2) : p1(p1), p2(p2) { }

        Eigen::Vector2d p1;
        Eigen::Vector2d p2;
    };

    // Estimates a homography such that
    //   points1 = H * points2
    Eigen::Matrix3d estimateHomographyPoints(
        const std::vector<Eigen::Vector2d> &points1,
        const std::vector<Eigen::Vector2d> &points2,
        bool pure_rotation);    
    
    
    // Note that this homography still maps second image to first!
    Eigen::Matrix3d estimateHomographyLines(
        const std::vector<Eigen::Vector3d> &lines1,
        const std::vector<Eigen::Vector3d> &lines2);

    // Note that this homography still maps second image to first!
    Eigen::Matrix3d estimateHomographyLineSegments(
        const std::vector<LineSegment> &line_segments1,
        const std::vector<LineSegment> &line_segments2,
        bool pure_rotation);

    void refineHomography(
        const std::vector<LineSegment> &line_segments1,
        const std::vector<LineSegment> &line_segments2,
        Eigen::Matrix3d &homography,
        bool pure_rotation);
    
    void refineHomography(
        const std::vector<Eigen::Vector2d> &pts1,
        const std::vector<Eigen::Vector2d> &pts2,
        Eigen::Matrix3d &homography,
        bool pure_rotation);

    void refineHomography(
        const std::vector<Eigen::Vector2d> &pts1,
        const std::vector<Eigen::Vector2d> &pts2,
        const std::vector<LineSegment> &line_segments1,
        const std::vector<LineSegment> &line_segments2,
        Eigen::Matrix3d &homography,
        bool pure_rotation);

    Eigen::Matrix3d ransacLineHomography(
        const std::vector<LineSegment> &line_segments1,
        const std::vector<LineSegment> &line_segments2,
        double tol_px,
        bool pure_rotation,
        std::vector<int> *inlier_ind);

    Eigen::Matrix3d ransacPointHomography(
        const std::vector<Eigen::Vector2d> &pts1,
        const std::vector<Eigen::Vector2d> &pts2,
        double tol_px,
        bool pure_rotation,
        std::vector<int> *inlier_ind);

   Eigen::Matrix3d ransacPointLineHomography(
        const std::vector<Eigen::Vector2d> &pts1,
        const std::vector<Eigen::Vector2d> &pts2,
        const std::vector<LineSegment> &line_segments1,
        const std::vector<LineSegment> &line_segments2,
        double tol_px,
        bool pure_rotation,
        std::vector<int> *inlier_pts_ind,
        std::vector<int> *inlier_lin_ind);

}


#endif