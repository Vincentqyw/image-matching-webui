#include <gtest/gtest.h>
#include <Eigen/Eigen>
#include <random>
#include "hest.h"
#include "Estimators.h"
//#include <opencv2/opencv.hpp>

static const std::vector<Eigen::Vector2d> EXAMPLE_SIMPLE_PTS1 = {
    {-0.3, -0.3},
    {-0.3, 0.5},
    {0.5, 0.5},
    {0.5, -0.3},
    {0.0, 0.0},
};

static const std::vector<hest::LineSegment> EXAMPLE_SIMPLE_LINES1 = {
    {{-0.5, -0.6}, {0.4, 0.3}},
    {{-0.6, 0.1}, {-0.2, 0.3}},
    {{0.5, -0.4}, {0.8, 0.7}},
    {{-0.1, 0.2}, {0.3, -0.1}},
};


Eigen::Matrix3d euler2rot(double pitch, double yaw, double roll) {
  Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitZ());

  Eigen::Quaternion<double> q = rollAngle * yawAngle * pitchAngle;

  Eigen::Matrix3d rotationMatrix = q.matrix();
  return rotationMatrix;
}

std::vector<Eigen::Vector2d> transform_points(const std::vector<Eigen::Vector2d> &pts, const Eigen::Matrix3d &R) {
  std::vector<Eigen::Vector2d> result;
  result.reserve(pts.size());
  for (const auto &p : pts) {
    result.emplace_back((R * p.homogeneous()).hnormalized());
  }
  return result;
}

std::vector<hest::LineSegment> transform_lines(const std::vector<hest::LineSegment> &lines, const Eigen::Matrix3d &R) {
  std::vector<hest::LineSegment> result;
  result.reserve(lines.size());
  for (const auto &l : lines) {
    result.emplace_back((R * l.p1.homogeneous()).hnormalized(), (R * l.p2.homogeneous()).hnormalized());
  }
  return result;
}

std::vector<Eigen::Vector2d> draw_random_pts(int N, std::mt19937 *gen = nullptr) {
  std::random_device rd;
  std::mt19937 rand_gen(rd());
  if (gen == nullptr) gen = &rand_gen;
  std::uniform_real_distribution<float> dis(-1.0, 1.0);
  std::vector<Eigen::Vector2d> result(N);
  for (int i = 0; i < N; i++) {
    result[i] = Eigen::Vector2d::NullaryExpr(2, 1, [&]() { return dis(*gen); });;
  }

  return result;
}

std::vector<hest::LineSegment> draw_random_lines(int N, std::mt19937 *gen = nullptr) {
  std::random_device rd;
  std::mt19937 rand_gen(rd());
  if (gen == nullptr) gen = &rand_gen;
  std::uniform_real_distribution<float> dis(-1.0, 1.0);
  std::vector<hest::LineSegment> result(N);
  for (int i = 0; i < N; i++) {
    result[i].p1 = Eigen::Vector2d::NullaryExpr(2, 1, [&]() { return dis(*gen); });
    result[i].p2 = Eigen::Vector2d::NullaryExpr(2, 1, [&]() { return dis(*gen); });
  }

  return result;
}

void add_gaussian_noise(std::vector<Eigen::Vector2d> &pts, double normal_std, std::mt19937 *gen = nullptr) {
  std::random_device rd;
  std::mt19937 rand_gen(rd());
  if (gen == nullptr) gen = &rand_gen;
  std::normal_distribution<double> dis(0.0, normal_std);

  for (auto &p : pts) {
    p[0] += dis(*gen);
    p[1] += dis(*gen);
  }
}

void add_gaussian_noise(std::vector<hest::LineSegment> &lines, double normal_std, std::mt19937 *gen = nullptr) {
  std::random_device rd;
  std::mt19937 rand_gen(rd());
  if (gen == nullptr) gen = &rand_gen;
  std::normal_distribution<double> dis(0.0, normal_std);

  for (auto &l : lines) {
    l.p1[0] += dis(*gen);
    l.p1[1] += dis(*gen);
    l.p2[0] += dis(*gen);
    l.p2[1] += dis(*gen);
  }
}

TEST(TestPureRot, SyntheticRot) {

  Eigen::Matrix3d R_gt = euler2rot(0.4, 0.3, 0.0);
  std::cout << "R_gt:\n" << R_gt << std::endl;

  std::vector<Eigen::Vector2d> points2 = EXAMPLE_SIMPLE_PTS1;
  auto points1 = transform_points(points2, R_gt);

  std::vector<hest::LineSegment> line_segments2 = EXAMPLE_SIMPLE_LINES1;
  auto line_segments1 = transform_lines(line_segments2, R_gt);

  std::vector<int> inlier_pts_ind, inlier_lin_ind;
  double inlier_th = 0.01;
  Eigen::Matrix3d R_est = hest::ransacPointLineHomography(points1, points2, line_segments1, line_segments2,
                                                          inlier_th, true, &inlier_pts_ind, &inlier_lin_ind);
  std::cout << "R=\n" << R_est << "\nR_gt=\n" << R_gt << "\n";
  ASSERT_TRUE(R_gt.isApprox(R_est, 0.3));

  std::vector<int> inlier_pts_ind2, inlier_lin_ind2;
  Eigen::Matrix3d R_est2 = hest::ransacPointLineHomography(points2, points1, line_segments2, line_segments1,
                                                           inlier_th, true, &inlier_pts_ind2, &inlier_lin_ind2);

  ASSERT_TRUE(R_gt.transpose().isApprox(R_est2, 0.3));
  ASSERT_EQ(inlier_pts_ind2.size(), points1.size());
  ASSERT_EQ(inlier_lin_ind2.size(), line_segments1.size());

  // Add an oultier point and line
  points1.insert(points1.begin() + 2, Eigen::Vector2d{0.2, -0.2});
  points2.insert(points2.begin() + 2, Eigen::Vector2d{-0.9, 0.9});
  line_segments1.emplace_back(Eigen::Vector2d{0.2, -0.2}, Eigen::Vector2d{0.0, 1.0});
  line_segments2.emplace_back(Eigen::Vector2d{-0.9, 0.3}, Eigen::Vector2d{0.9, 0.5});
  std::vector<int> inlier_pts_ind3, inlier_lin_ind3;
  Eigen::Matrix3d R_est3 = hest::ransacPointLineHomography(points2, points1, line_segments2, line_segments1,
                                                           inlier_th, true, &inlier_pts_ind2, &inlier_lin_ind2);

  ASSERT_TRUE(R_gt.transpose().isApprox(R_est3, 0.3));
  ASSERT_EQ(inlier_pts_ind2.size(), points1.size() - 1);
  ASSERT_EQ(inlier_lin_ind2.size(), line_segments1.size() - 1);


//  Eigen::Matrix3d K({{200, 0, 200},
//                     {0, 200, 200},
//                     {0, 0, 1}});
//
//  cv::Mat img(400, 400, CV_8UC3, CV_RGB(255, 255, 255));
//  for (auto p : transform_points(points1, K))
//    cv::circle(img, cv::Point2d(p.x(), p.y()), 4, CV_RGB(255, 0, 255), -1);
//  for (auto p : transform_points(points2, K))
//    cv::circle(img, cv::Point2d(p.x(), p.y()), 4, CV_RGB(0, 255, 0), -1);
//
//  cv::imshow("img", img);
//  cv::waitKey();
}

TEST(TestPureRot, SolverOnlyPts) {
  // Should be able to obtain a good Rotation based only on points
  Eigen::Matrix3d R_gt = euler2rot(0.4, 0.3, 0.0);
//  std::cout << "R_gt:\n" << R_gt << std::endl;

  std::vector<Eigen::Vector2d> pts2 = EXAMPLE_SIMPLE_PTS1;
  std::vector<Eigen::Vector2d> pts1 = transform_points(pts2, R_gt);
  std::vector<hest::LineSegment> line_segments1, line_segments2;

  Eigen::Matrix3d model = hest::estimateHomographyPoints(pts1, pts2, true);
  ASSERT_TRUE(R_gt.isApprox(model, 0.01));

  model = hest::estimateHomographyPoints(pts2, pts1, true);
  ASSERT_TRUE(R_gt.transpose().isApprox(model, 0.01));

  // Create estimator object
  hest::PointLineHomographyEstimator solver(pts1, pts2, line_segments1, line_segments2, true);
  for(int i = 0; i < static_cast<int>(pts1.size()); ++i) {
    for(int j = 0; j < i; ++j) {
      std::vector<std::vector<int>> sample = {{i, j}, {}};
      int solver_idx = 0;
      std::vector<Eigen::Matrix3d> models;
      solver.MinimalSolver(sample, solver_idx, &models);
      ASSERT_TRUE(R_gt.isApprox(models[0], 0.01));
    }
  }
}

TEST(TestPureRot, SolverOnlyLines) {
  // Should be able to obtain a good Rotation based only on lines
  Eigen::Matrix3d R_gt = euler2rot(0.4, 0.3, 0);

  std::vector<Eigen::Vector2d> pts1, pts2;
  std::vector<hest::LineSegment> line_segments2 = EXAMPLE_SIMPLE_LINES1;
  std::vector<hest::LineSegment> line_segments1 = transform_lines(line_segments2, R_gt);

  Eigen::Matrix3d model = hest::estimateHomographyLineSegments(line_segments1, line_segments2, true);
  ASSERT_TRUE(R_gt.isApprox(model, 0.01));

  model = hest::estimateHomographyLineSegments(line_segments2, line_segments1, true);
  ASSERT_TRUE(R_gt.transpose().isApprox(model, 0.01));

  // Create estimator object
  hest::PointLineHomographyEstimator solver(pts1, pts2, line_segments1, line_segments2, true);
  for(int i = 0; i < static_cast<int>(line_segments1.size()); ++i) {
      for(int j = 0; j < i; ++j) {
        std::vector<std::vector<int>> sample = {{}, {i, j}};
        int solver_idx = 2;
        std::vector<Eigen::Matrix3d> models;
        solver.MinimalSolver(sample, solver_idx, &models);
        ASSERT_TRUE(R_gt.isApprox(models[0], 0.01));
      }
  }
}

TEST(TestPureRot, SolverOnlyLinesSwapEndpoints) {
  // Check that if we swap the endpoints of the line segment, we still obtain the same solution
  Eigen::Matrix3d R_gt = euler2rot(0.4, 0.3, 0.0);
//  std::cout << "R_gt:\n" << R_gt << std::endl;

  std::vector<Eigen::Vector2d> pts1, pts2;
  std::vector<hest::LineSegment> line_segments1 = EXAMPLE_SIMPLE_LINES1;
  for (size_t i = 0; i < line_segments1.size(); i++) {
    std::vector<hest::LineSegment> line_segments2 = transform_lines(line_segments1, R_gt);
    line_segments2[i] = hest::LineSegment(line_segments2[i].p2, line_segments2[i].p1);

    Eigen::Matrix3d model1;
    model1 = hest::estimateHomographyLineSegments(line_segments2, line_segments1, true);
    ASSERT_TRUE(R_gt.isApprox(model1, 0.01));

    Eigen::Matrix3d model2;
    model2 = hest::estimateHomographyLineSegments(line_segments1, line_segments2, true);
    ASSERT_TRUE(R_gt.transpose().isApprox(model2, 0.01));
  }
}

TEST(TestPureRot, SolverPointsAndLines) {
  // Should be able to obtain a good Rotation based only on both points and lines
  Eigen::Matrix3d R_gt = euler2rot(0.4, 0.3, 0.0);
  std::cout << "R_gt:\n" << R_gt << std::endl;

  std::vector<Eigen::Vector2d> pts2 = EXAMPLE_SIMPLE_PTS1;
  std::vector<Eigen::Vector2d> pts1 = transform_points(pts2, R_gt);
  std::vector<hest::LineSegment> line_segments2 = EXAMPLE_SIMPLE_LINES1;
  std::vector<hest::LineSegment> line_segments1 = transform_lines(line_segments2, R_gt);

  // Create estimator object
  hest::PointLineHomographyEstimator solver(pts1, pts2, line_segments1, line_segments2, true);

  for(int i = 0; i < static_cast<int>(pts1.size()); ++i) {
    for(int j = 0; j < static_cast<int>(line_segments1.size()); ++j) {
      std::vector<std::vector<int>> sample = {{i}, {j}};
      int solver_idx = 1;
      std::vector<Eigen::Matrix3d> models;
      solver.MinimalSolver(sample, solver_idx, &models);
      if((models[0]-R_gt).norm() > 0.01) {
        std::cout << "i,j = " << i << ", " << j << ", R\n" << models[0] << "\n";
      }
      ASSERT_TRUE(R_gt.isApprox(models[0], 0.01));
    }
  }

//  std::cout << models.size() << std::endl;
//  for (auto &m : models) {
//    auto est_lines1 = transform_lines(line_segments2, m);
//    double residuals = 0;
//    for (size_t i = 0 ; i < est_lines1.size() ; i++){
//      residuals += (line_segments1[i].p1 - est_lines1[i].p1).squaredNorm();
//      residuals += (line_segments1[i].p2 - est_lines1[i].p2).squaredNorm();
//    }
//    std::cout << "MSE: " << residuals / double(2 * line_segments1.size()) << std::endl;
//    std::cout << "Model:\n" << m << std::endl;
//  }
}

TEST(TestPureRot, SolverLeastSquares) {
  // Should be able to obtain a good Rotation based only on both points and lines
  Eigen::Matrix3d R_gt = euler2rot(0.4, 0.3, 0.0);
  std::cout << "R_gt:\n" << R_gt << std::endl;

  std::vector<Eigen::Vector2d> pts2 = EXAMPLE_SIMPLE_PTS1;
  std::vector<Eigen::Vector2d> pts1 = transform_points(pts2, R_gt);
  std::vector<hest::LineSegment> line_segments2 = EXAMPLE_SIMPLE_LINES1;
  std::vector<hest::LineSegment> line_segments1 = transform_lines(line_segments2, R_gt);

  // Create estimator object
  hest::PointLineHomographyEstimator solver(pts1, pts2, line_segments1, line_segments2, true);
  std::vector<std::vector<int>> sample = {{0, 1, 2, 3, 4}, {0, 1, 2, 3}};
  // Initialize the least square optimization with a rotation close to the optimal one
  auto R_noise = euler2rot(0.01, 0.05, 0.0);
  Eigen::Matrix3d model = R_gt * R_noise;
  solver.LeastSquares(sample, &model);

  auto est_lines1 = transform_lines(line_segments2, model);
  double residuals = 0;
  for (size_t i = 0 ; i < est_lines1.size() ; i++){
    residuals += (line_segments1[i].p1 - est_lines1[i].p1).squaredNorm();
    residuals += (line_segments1[i].p2 - est_lines1[i].p2).squaredNorm();
  }
  std::cout << "MSE: " << residuals / double(2 * line_segments1.size()) << std::endl;
  std::cout << "Model:\n" << model << std::endl;

  ASSERT_TRUE(R_gt.isApprox(model, 0.01));
}

TEST(TestPureRot, EvaluateModelOnPoint) {

}

TEST(TestPureRot, MoreFeatureLessError) {
  // In this test we take local features contaminated by gaussian noisy and check that if we add lines, they
  // contribute in reducing the final estimation error
  Eigen::Matrix3d R_gt = euler2rot(0.1, -0.2, 0.0);
//  std::cout << "R_gt:\n" << R_gt << std::endl;

  const int seed = 3;
  std::mt19937 gen(seed);
  const int npts = 100, nlines = 1000;
  std::vector<Eigen::Vector2d> pts1 = draw_random_pts(npts, &gen);
  std::vector<hest::LineSegment> line_segments1 = draw_random_lines(nlines, &gen);
  // Get perfect equivalents in image2
  std::vector<Eigen::Vector2d> pts2 = transform_points(pts1, R_gt);
  std::vector<hest::LineSegment> line_segments2 = transform_lines(line_segments1, R_gt);

  // Add Gaussian noise to the endpoints
  const double error_std = 0.01;
  std::mt19937 *pgen = &gen; // Set to NULL if you want random noise
  add_gaussian_noise(pts2, error_std, pgen);
  add_gaussian_noise(line_segments2, error_std, pgen);

  // Estimate rotation with the function used by the minimal solver
  Eigen::Matrix3d only_pts_model1 = hest::est_rotation(pts2, pts1, {}, {});
  Eigen::Matrix3d pts_lines_model1 = hest::est_rotation(pts2, pts1, line_segments2, line_segments1);
  // Check that the error obtained with lines is smaller than the one with only points
  double only_pts_error1 = Eigen::AngleAxisd(R_gt * only_pts_model1.transpose()).angle();
  double pts_lines_error1 = Eigen::AngleAxisd(R_gt * pts_lines_model1.transpose()).angle();
  ASSERT_GE(only_pts_error1, pts_lines_error1);
//  std::cout << "only_pts_model1: \n" << only_pts_model1 << std::endl;
//  std::cout << "pts_lines_model1: \n" << pts_lines_model1 << std::endl;
//  std::cout << "only_pts_error1: " << only_pts_error1 << std::endl;
//  std::cout << "pts_lines_error1: " << pts_lines_error1 << std::endl;

  hest::PointLineHomographyEstimator solver(pts2, pts1, line_segments2, line_segments1, true);
  std::vector<std::vector<int>> sample(2);
  for (int j=0; j < npts; j++) sample[0].push_back(j);
  Eigen::Matrix3d only_pts_model2 = Eigen::Matrix3d::Identity();
  solver.LeastSquares(sample, &only_pts_model2);

  for (int j=0; j < nlines; j++) sample[1].push_back(j);
  Eigen::Matrix3d pts_lines_model2 = Eigen::Matrix3d::Identity();
  solver.LeastSquares(sample, &pts_lines_model2);


  // Check that the error obtained with lines is smaller than the one with only points
  double only_pts_error2 = Eigen::AngleAxisd(R_gt * only_pts_model2.transpose()).angle();
  double pts_lines_error2 = Eigen::AngleAxisd(R_gt * pts_lines_model2.transpose()).angle();
  ASSERT_GE(only_pts_error2, pts_lines_error2);
//  std::cout << "only_pts_model2: \n" << only_pts_model2 << std::endl;
//  std::cout << "pts_lines_model2: \n" << pts_lines_model2 << std::endl;
//  std::cout << "only_pts_error2: " << only_pts_error2 << std::endl;
//  std::cout << "pts_lines_error2: " << pts_lines_error2 << std::endl;
//
//    Eigen::Matrix3d K({{200, 0, 200},
//                       {0, 200, 200},
//                       {0, 0, 1}});
//
//    cv::Mat img(400, 400, CV_8UC3, CV_RGB(255, 255, 255));
//    for (auto p : transform_points(pts2, K))
//      cv::circle(img, cv::Point2d(p.x(), p.y()), 4, CV_RGB(255, 0, 255), -1);
//    for (auto p : transform_lines(line_segments2, K))
//      cv::line(img, cv::Point2d(p.p1.x(), p.p1.y()),  cv::Point2d(p.p2.x(), p.p2.y()), CV_RGB(0, 255, 0), 1);
//
//    cv::imshow("img", img);
//    cv::waitKey();
}