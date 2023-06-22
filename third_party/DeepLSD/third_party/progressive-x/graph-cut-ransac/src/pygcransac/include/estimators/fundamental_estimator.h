// Copyright (C) 2019 Czech Technical University.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of Czech Technical University nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Daniel Barath (barath.daniel@sztaki.mta.hu)
#pragma once

#define _USE_MATH_DEFINES

#include <math.h>
#include <cmath>
#include <random>
#include <vector>

#include <unsupported/Eigen/Polynomials>
#include <Eigen/Eigen>

#include "estimator.h"
#include "homography_estimator.h"
#include "model.h"
#include "../neighborhood/grid_neighborhood_graph.h"
#include "../samplers/uniform_sampler.h"

#include "GCRANSAC.h"

#include "solver_fundamental_matrix_seven_point.h"
#include "solver_fundamental_matrix_plane_and_parallax.h"
#include "solver_fundamental_matrix_eight_point.h"

namespace gcransac
{
	namespace estimator
	{
		// This is the estimator class for estimating a fundamental matrix between two images. A model estimation method and error calculation method are implemented
		template<class _MinimalSolverEngine,  // The solver used for estimating the model from a minimal sample
			class _NonMinimalSolverEngine> // The solver used for estimating the model from a non-minimal sample
			class FundamentalMatrixEstimator : public Estimator < cv::Mat, Model >
		{
		protected:
			// Minimal solver engine used for estimating a model from a minimal sample
			const std::shared_ptr<_MinimalSolverEngine> minimal_solver;

			// Non-minimal solver engine used for estimating a model from a bigger than minimal sample
			const std::shared_ptr<_NonMinimalSolverEngine> non_minimal_solver;

			// The lower bound of the inlier ratio which is required to pass the validity test.
			// The validity test measures what proportion of the inlier (by Sampson distance) is inlier
			// when using symmetric epipolar distance. 
			const double minimum_inlier_ratio_in_validity_check;

			// A flag deciding if DEGENSAC should be used. DEGENSAC handles the cases when the points of the model originates from 
			// a single plane, or almost from single plane.
			// DEGENSAC paper: Chum, Ondrej, Tomas Werner, and Jiri Matas. "Two-view geometry estimation unaffected by a dominant plane." 
			// 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition(CVPR'05). Vol. 1. IEEE, 2005.
			const bool use_degensac;
			
			// The threshold (in pixels) for deciding if a sample is H-degenerate in DEGENSAC
			const double homography_threshold,
				// The squared threshold (in pixels) for deciding if a sample is H-degenerate in DEGENSAC
				squared_homography_threshold;

		public:
			FundamentalMatrixEstimator(const double minimum_inlier_ratio_in_validity_check_ = 0.5,
				const bool use_degensac_ = true,
				const double homography_threshold_ = 2.0) :
				// Minimal solver engine used for estimating a model from a minimal sample
				minimal_solver(std::make_shared<_MinimalSolverEngine>()),
				// Non-minimal solver engine used for estimating a model from a bigger than minimal sample
				non_minimal_solver(std::make_shared<_NonMinimalSolverEngine>()),
				// The lower bound of the inlier ratio which is required to pass the validity test.
				// It is clamped to be in interval [0, 1].
				minimum_inlier_ratio_in_validity_check(std::clamp(minimum_inlier_ratio_in_validity_check_, 0.0, 1.0)),
				// A flag deciding if DEGENSAC should be used. DEGENSAC handles the cases when the points of the model originates from 
				// a single plane, or almost from single plane.
				use_degensac(use_degensac_),
				// The threshold (in pixels) for deciding if a sample is H-degenerate in DEGENSAC
				homography_threshold(homography_threshold_),
				// The squared threshold (in pixels) for deciding if a sample is H-degenerate in DEGENSAC
				squared_homography_threshold(homography_threshold_ * homography_threshold_)
			{}

			~FundamentalMatrixEstimator() {}

			_MinimalSolverEngine *getMinimalSolver() {
				return minimal_solver.get();
			}

			_NonMinimalSolverEngine *getNonMinimalSolver() {
				return non_minimal_solver.get();
			}

			// The size of a minimal sample_ required for the estimation
			static constexpr size_t maximumMinimalSolutions() {
				return _MinimalSolverEngine::maximumSolutions();
			}

			// The size of a non-minimal sample required for the estimation
			static constexpr size_t nonMinimalSampleSize() {
				return _NonMinimalSolverEngine::sampleSize();
			}

			// The size of a minimal sample required for the estimation
			static constexpr size_t sampleSize() {
				return _MinimalSolverEngine::sampleSize();
			}

			// A flag deciding if the points can be weighted when the non-minimal fitting is applied 
			static constexpr bool isWeightingApplicable() {
				return true;
			}

			// The size of a sample when doing inner RANSAC on a non-minimal sample
			OLGA_INLINE size_t inlierLimit() const {
				return 7 * sampleSize();
			}

			OLGA_INLINE bool estimateModel(const cv::Mat& data,
				const size_t *sample,
				std::vector<Model>* models) const
			{
				// Model calculation by the seven point algorithm
				constexpr size_t sample_size = sampleSize();

				// Estimate the model parameters by the minimal solver
				minimal_solver->estimateModel(data,
					sample,
					sample_size,
					*models);

				// Orientation constraint check 
				for (short model_idx = models->size() - 1; model_idx >= 0; --model_idx)
					if (!isOrientationValid(models->at(model_idx).descriptor,
						data,
						sample,
						sample_size))
						models->erase(models->begin() + model_idx);

				// The estimation was successfull if at least one model is kept
				return models->size() > 0;
			}

			// The sampson distance between a point correspondence and an essential matrix
			OLGA_INLINE double sampsonDistance(const cv::Mat& point_,
				const Eigen::Matrix3d& descriptor_) const
			{
				const double squared_distance = squaredSampsonDistance(point_, descriptor_);
				return sqrt(squared_distance);
			}

			// The sampson distance between a point_ correspondence and an essential matrix
			OLGA_INLINE double squaredSampsonDistance(const cv::Mat& point_,
				const Eigen::Matrix3d& descriptor_) const
			{
				const double* s = reinterpret_cast<double *>(point_.data);
				const double 
					&x1 = *s,
					&y1 = *(s + 1),
					&x2 = *(s + 2),
					&y2 = *(s + 3);

				const double 
					&e11 = descriptor_(0, 0),
					&e12 = descriptor_(0, 1),
					&e13 = descriptor_(0, 2),
					&e21 = descriptor_(1, 0),
					&e22 = descriptor_(1, 1),
					&e23 = descriptor_(1, 2),
					&e31 = descriptor_(2, 0),
					&e32 = descriptor_(2, 1),
					&e33 = descriptor_(2, 2);

				double rxc = e11 * x2 + e21 * y2 + e31;
				double ryc = e12 * x2 + e22 * y2 + e32;
				double rwc = e13 * x2 + e23 * y2 + e33;
				double r = (x1 * rxc + y1 * ryc + rwc);
				double rx = e11 * x1 + e12 * y1 + e13;
				double ry = e21 * x1 + e22 * y1 + e23;

				return r * r /
					(rxc * rxc + ryc * ryc + rx * rx + ry * ry);
			}

			// The symmetric epipolar distance between a point_ correspondence and an essential matrix
			OLGA_INLINE double squaredSymmetricEpipolarDistance(const cv::Mat& point_,
				const Eigen::MatrixXd& descriptor_) const
			{
				const double* s = reinterpret_cast<double *>(point_.data);
				const double 
					&x1 = *s,
					&y1 = *(s + 1),
					&x2 = *(s + 2),
					&y2 = *(s + 3);

				const double 
					&e11 = descriptor_(0, 0),
					&e12 = descriptor_(0, 1),
					&e13 = descriptor_(0, 2),
					&e21 = descriptor_(1, 0),
					&e22 = descriptor_(1, 1),
					&e23 = descriptor_(1, 2),
					&e31 = descriptor_(2, 0),
					&e32 = descriptor_(2, 1),
					&e33 = descriptor_(2, 2);

				const double rxc = e11 * x2 + e21 * y2 + e31;
				const double ryc = e12 * x2 + e22 * y2 + e32;
				const double rwc = e13 * x2 + e23 * y2 + e33;
				const double r = (x1 * rxc + y1 * ryc + rwc);
				const double rx = e11 * x1 + e12 * y1 + e13;
				const double ry = e21 * x1 + e22 * y1 + e23;
				const double a = rxc * rxc + ryc * ryc;
				const double b = rx * rx + ry * ry;

				return r * r * (a + b) / (a * b);
			}

			OLGA_INLINE double squaredResidual(const cv::Mat& point_,
				const Model& model_) const
			{
				return squaredResidual(point_, model_.descriptor);
			}

			// The squared residual function used for deciding which points are inliers
			OLGA_INLINE double squaredResidual(const cv::Mat& point_,
				const Eigen::MatrixXd& descriptor_) const
			{
				return squaredSampsonDistance(point_, descriptor_);
			}

			// The residual function used for deciding which points are inliers
			OLGA_INLINE double residual(const cv::Mat& point_,
				const Model& model_) const
			{
				return residual(point_, model_.descriptor);
			}

			// The residual function used for deciding which points are inliers
			OLGA_INLINE double residual(const cv::Mat& point_,
				const Eigen::MatrixXd& descriptor_) const
			{
				return sampsonDistance(point_, descriptor_);
			}

			// Validate the model by checking the number of inlier with symmetric epipolar distance
			// instead of Sampson distance. In general, Sampson distance is more accurate but less
			// robust to degenerate solutions than the symmetric epipolar distance. Therefore,
			// every so-far-the-best model is checked if it has enough inlier with symmetric
			// epipolar distance as well. 
			bool isValidModel(Model& model_,
				const cv::Mat& data_,
				const std::vector<size_t> &inliers_,
				const size_t *minimal_sample_,
				const double threshold_,
				bool &model_updated_) const
			{
				// Validate the model by checking the number of inlier with symmetric epipolar distance
				// instead of Sampson distance. In general, Sampson distance is more accurate but less
				// robust to degenerate solutions than the symmetric epipolar distance. Therefore,
				// every so-far-the-best model is checked if it has enough inlier with symmetric
				bool passed = false;
				size_t inlier_number = 0; // Number of inlier if using symmetric epipolar distance
				const Eigen::Matrix3d &descriptor = model_.descriptor; // The decriptor of the current model
				constexpr size_t sample_size = sampleSize(); // Size of a minimal sample
				// Minimum number of inliers which should be inlier as well when using symmetric epipolar distance instead of Sampson distance
				const size_t minimum_inlier_number =
					MAX(sample_size, inliers_.size() * minimum_inlier_ratio_in_validity_check);
				// Squared inlier-outlier threshold
				const double squared_threshold = threshold_ * threshold_;

				// Iterate through the inliers_ determined by Sampson distance
				for (const auto &idx : inliers_)
					// Calculate the residual using symmetric epipolar distance and check if
					// it is smaller than the threshold_.
					if (squaredSymmetricEpipolarDistance(data_.row(idx), descriptor) < squared_threshold)
						// Increase the inlier number and terminate if enough inliers_ have been found.
						if (++inlier_number >= minimum_inlier_number)
						{
							passed = true;
							break;
						}

				// If the fundamental matrix has not passed the symmetric epipolar tests,
				// terminate.
				if (!passed)
					return false;

				// Validate the model by checking if the scene is dominated by a single plane.
				if (use_degensac)
					return applyDegensac(model_,
						data_,
						inliers_,
						minimal_sample_,
						threshold_,
						model_updated_);

				// If DEGENSAC is not applied and the model passed the previous tests,
				// assume that it is a valid model.
				return true;
			}

			//  Evaluate the H-degenerate sample test and apply DEGENSAC if needed
			inline bool applyDegensac(Model& model_, // The input model to be tested
				const cv::Mat& data_, // All data points
				const std::vector<size_t> &inliers_, // The inliers of the input model
				const size_t *minimal_sample_, // The minimal sample used for estimating the model
				const double threshold_, // The inlier-outlier threshold
				bool &model_updated_) const // A flag saying if the model has been updated here
			{
				// Set the flag initially to false since the model has not been yet updated.
				model_updated_ = false;

				// The possible triplets of points
				constexpr size_t triplets[] = {
					0, 1, 2,
					3, 4, 5,
					0, 1, 6,
					3, 4, 6,
					2, 5, 6 };
				constexpr size_t number_of_triplets = 5; // The number of triplets to be tested
				const size_t columns = data_.cols; // The number of columns in the data matrix

				// The fundamental matrix coming from the minimal sample
				const Eigen::Matrix3d &fundamental_matrix =
					model_.descriptor.block<3, 3>(0, 0);
				
				// Applying SVD decomposition to the estimated fundamental matrix
				Eigen::JacobiSVD<Eigen::Matrix3d> svd(
					fundamental_matrix,
					Eigen::ComputeFullU | Eigen::ComputeFullV);
				
				// Calculate the epipole in the second image
				const Eigen::Vector3d epipole =
					svd.matrixU().rightCols<1>().head<3>() / svd.matrixU()(2,2);
				
				// The calculate the cross-produced matrix of the epipole
				Eigen::Matrix3d epipolar_cross;
				epipolar_cross << 0, -epipole(2), epipole(1),
					epipole(2), 0, -epipole(0),
					-epipole(1), epipole(0), 0;

				const Eigen::Matrix3d A = 
					epipolar_cross * model_.descriptor;

				// A flag deciding if the sample is H-degenerate
				bool h_degenerate_sample = false;
				// The homography which the H-degenerate part of the sample implies
				Eigen::Matrix3d best_homography;
				// Iterate through the triplets of points in the sample
				for (size_t triplet_idx = 0; triplet_idx < number_of_triplets; ++triplet_idx)
				{
					// The index of the first point of the triplet
					const size_t triplet_offset = triplet_idx * 3;
					// The indices of the other points
					const size_t point_1_idx = minimal_sample_[triplets[triplet_offset]],
						point_2_idx = minimal_sample_[triplets[triplet_offset + 1]],
						point_3_idx = minimal_sample_[triplets[triplet_offset + 2]];

					// A pointer to the first point's first coordinate
					const double *point_1_ptr =
						reinterpret_cast<double *>(data_.data) + point_1_idx * columns;
					// A pointer to the second point's first coordinate
					const double *point_2_ptr =
						reinterpret_cast<double *>(data_.data) + point_2_idx * columns;
					// A pointer to the third point's first coordinate
					const double *point_3_ptr =
						reinterpret_cast<double *>(data_.data) + point_3_idx * columns;

					// Copy the point coordinates into Eigen vectors
					Eigen::Vector3d point_1_1, point_1_2, point_1_3,
						point_2_1, point_2_2, point_2_3;

					point_1_1 << point_1_ptr[0], point_1_ptr[1], 1;
					point_2_1 << point_1_ptr[2], point_1_ptr[3], 1;
					point_1_2 << point_2_ptr[0], point_2_ptr[1], 1;
					point_2_2 << point_2_ptr[2], point_2_ptr[3], 1;
					point_1_3 << point_3_ptr[0], point_3_ptr[1], 1;
					point_2_3 << point_3_ptr[2], point_3_ptr[3], 1;

					// Calculate the cross-product of the epipole end each point
					Eigen::Vector3d point_1_cross_epipole = point_2_1.cross(epipole);
					Eigen::Vector3d point_2_cross_epipole = point_2_2.cross(epipole);
					Eigen::Vector3d point_3_cross_epipole = point_2_3.cross(epipole);

					Eigen::Vector3d b;
					b << point_2_1.cross(A * point_1_1).transpose() * point_1_cross_epipole / point_1_cross_epipole.squaredNorm(),
						point_2_2.cross(A * point_1_2).transpose() * point_2_cross_epipole / point_2_cross_epipole.squaredNorm(),
						point_2_3.cross(A * point_1_3).transpose() * point_3_cross_epipole / point_3_cross_epipole.squaredNorm();

					Eigen::Matrix3d M;
					M << point_1_1(0), point_1_1(1), point_1_1(2),
						point_1_2(0), point_1_2(1), point_1_2(2),
						point_1_3(0), point_1_3(1), point_1_3(2);
					
					Eigen::Matrix3d homography =
						A - epipole * (M.inverse() * b).transpose();

					// The number of point consistent with the implied homography
					size_t inlier_number = 3;

					// Count the inliers of the homography
					for (size_t i = 0; i < sampleSize(); ++i)
					{
						// Get the point's index from the minimal sample
						size_t idx = minimal_sample_[i];

						// Check if the point is not included in the current triplet
						if (idx == point_1_idx ||
							idx == point_2_idx ||
							idx == point_3_idx)
							continue; // If yes, the error does not have to be calculated
					
						// Calculate the re-projection error
						const double *point_ptr =
							reinterpret_cast<double *>(data_.data) + idx * columns;

						const double &x1 = point_ptr[0], // The x coordinate in the first image
							&y1 = point_ptr[1], // The y coordinate in the first image
							&x2 = point_ptr[2], // The x coordinate in the second image
							&y2 = point_ptr[3]; // The y coordinate in the second image

						// Calculating H * p
						const double t1 = homography(0, 0) * x1 + homography(0, 1) * y1 + homography(0, 2),
							t2 = homography(1, 0) * x1 + homography(1, 1) * y1 + homography(1, 2),
							t3 = homography(2, 0) * x1 + homography(2, 1) * y1 + homography(2, 2);

						// Calculating the difference of the projected and original points
						const double d1 = x2 - (t1 / t3),
							d2 = y2 - (t2 / t3);

						// Calculating the squared re-projection error
						const double squared_residual = d1 * d1 + d2 * d2;

						// If the squared re-projection error is smaller than the threshold, 
						// consider the point inlier.
						if (squared_residual < squared_homography_threshold)
							++inlier_number;
					}

					// If at least 5 points are correspondences are consistent with the homography,
					// consider the sample as H-degenerate.
					if (inlier_number >= 5)
					{
						// Saving the parameters of the homography
						best_homography = homography; 
						// Setting the flag of being a h-degenerate sample
						h_degenerate_sample = true;
						break;
					}
				}

				// If the sample is H-degenerate
				if (h_degenerate_sample)
				{
					// Declare a homography estimator to be able to calculate the residual and the homography from a non-minimal sample
					static const estimator::RobustHomographyEstimator<estimator::solver::HomographyFourPointSolver, // The solver used for fitting a model to a minimal sample
						estimator::solver::HomographyFourPointSolver> homography_estimator;

					// The inliers of the homography
					std::vector<size_t> homography_inliers;
					homography_inliers.reserve(inliers_.size());

					// Iterate through the inliers of the fundamental matrix
					// and select those which are inliers of the homography as well.
					//for (size_t inlier_idx = 0; inlier_idx < data_.rows; ++inlier_idx)
					for (const size_t &inlier_idx : inliers_)
						if (homography_estimator.squaredResidual(data_.row(inlier_idx), best_homography) < squared_homography_threshold)
							homography_inliers.emplace_back(inlier_idx);

					// If the homography does not have enough inliers to be estimated, terminate.
					if (homography_inliers.size() < homography_estimator.nonMinimalSampleSize())
						return false;

					// The set of estimated homographies. For all implemented solvers,
					// this should be of size 1.
					std::vector<Model> homographies;

					// Estimate the homography parameters from the provided inliers.
					homography_estimator.estimateModelNonminimal(data_, // All data points
 						&homography_inliers[0], // The inliers of the homography
						homography_inliers.size(), // The number of inliers
						&homographies); // The estimated homographies

					// If the number of estimated homographies is not 1, there is some problem
					// and, thus, terminate.
					if (homographies.size() != 1)
						return false;

					// Get the reference of the homography fit to the non-minimal sample
					const Eigen::Matrix3d &nonminimal_homography = 
						homographies[0].descriptor;

					// Do a local GC-RANSAC to determine the parameters of the fundamental matrix by
					// the plane-and-parallax algorithm using the determined homography.
					estimator::FundamentalMatrixEstimator<estimator::solver::FundamentalMatrixPlaneParallaxSolver, // The solver used for fitting a model to a minimal sample
						estimator::solver::FundamentalMatrixEightPointSolver> estimator(0.0, false);
					estimator.getMinimalSolver()->setHomography(&nonminimal_homography);

					std::vector<int> inliers;
					Model model;

					GCRANSAC<estimator::FundamentalMatrixEstimator<estimator::solver::FundamentalMatrixPlaneParallaxSolver, // The solver used for fitting a model to a minimal sample
						estimator::solver::FundamentalMatrixEightPointSolver>, neighborhood::GridNeighborhoodGraph<4>> gcransac;
					gcransac.settings.threshold = threshold_; // The inlier-outlier threshold
					gcransac.settings.spatial_coherence_weight = 0; // The weight of the spatial coherence term
					gcransac.settings.confidence = 0.99; // The required confidence in the results
					gcransac.settings.neighborhood_sphere_radius = 8; // The radius of the neighborhood ball

					sampler::UniformSampler sampler(&data_); // The local optimization sampler is used inside the local optimization

					gcransac.run(data_, // All data points
						estimator, // The fundamental matrix estimator
						&sampler, // The main sampler 
						&sampler, // The sampler for local optimization
						nullptr, // There is no neighborhood graph now  
						model); // The estimated model parameters
					
					// The statistics of the inner GC-RANSAC
					const utils::RANSACStatistics &statistics = 
						gcransac.getRansacStatistics();

					// If more inliers are found the what initially was given,
					// update the model parameters.
					if (statistics.inliers.size() > inliers_.size())
					{
						// Consider the model to be updated
						model_updated_ = true;
						// Update the parameters
						model_ = model;
					}
				}

				// If we get to this point, the procedure was successfull
				return true;
			}

			inline bool estimateModelNonminimal(
				const cv::Mat& data_,
				const size_t *sample_,
				const size_t &sample_number_,
				std::vector<Model>* models_,
				const double *weights_ = nullptr) const
			{
				if (sample_number_ < nonMinimalSampleSize())
					return false;

				cv::Mat normalized_points(sample_number_, data_.cols, data_.type()); // The normalized point coordinates
				Eigen::Matrix3d normalizing_transform_source, // The normalizing transformations in the source image
					normalizing_transform_destination; // The normalizing transformations in the destination image

				// Normalize the point coordinates to achieve numerical stability when
				// applying the least-squares model fitting.
				if (!normalizePoints(data_, // The data points
					sample_, // The points to which the model will be fit
					sample_number_, // The number of points
					normalized_points, // The normalized point coordinates
					normalizing_transform_source, // The normalizing transformation in the first image
					normalizing_transform_destination)) // The normalizing transformation in the second image
					return false;

				// The eight point fundamental matrix fitting algorithm
				non_minimal_solver->estimateModel(normalized_points,
					nullptr,
					sample_number_,
					*models_,
					weights_);

				// Denormalizing the estimated fundamental matrices
				const Eigen::Matrix3d normalizing_transform_destination_transpose = normalizing_transform_destination.transpose();
				for (auto &model : *models_)
				{
					// Transform the estimated fundamental matrix back to the not normalized space
					model.descriptor = normalizing_transform_destination_transpose * model.descriptor * normalizing_transform_source;

					// Normalizing the fundamental matrix elements
					model.descriptor.normalize();
					if (model.descriptor(2, 2) < 0)
						model.descriptor = -model.descriptor;
				}
				return true;
			}

			inline void enforceRankTwoConstraint(Model &model_) const
			{
				// Applying SVD decomposition to the estimated fundamental matrix
				Eigen::JacobiSVD<Eigen::MatrixXd> svd(
					model_.descriptor,
					Eigen::ComputeThinU | Eigen::ComputeThinV);

				// Making the last singular value zero
				Eigen::Matrix3d diagonal = svd.singularValues().asDiagonal();
				diagonal(2, 2) = 0.0;

				// Getting back the fundamental matrix from the SVD decomposition
				// using the new singular values
				model_.descriptor =
					svd.matrixU() * diagonal * svd.matrixV().transpose();
			}

			inline bool normalizePoints(
				const cv::Mat& data_, // The data points
				const size_t *sample_, // The points to which the model will be fit
				const size_t &sample_number_,// The number of points
				cv::Mat &normalized_points_, // The normalized point coordinates
				Eigen::Matrix3d &normalizing_transform_source_, // The normalizing transformation in the first image
				Eigen::Matrix3d &normalizing_transform_destination_) const // The normalizing transformation in the second image
			{
				const size_t cols = data_.cols;
				double *normalized_points_ptr = reinterpret_cast<double *>(normalized_points_.data);
				const double *points_ptr = reinterpret_cast<double *>(data_.data);

				double mass_point_src[2], // Mass point in the first image
					mass_point_dst[2]; // Mass point in the second image

				// Initializing the mass point coordinates
				mass_point_src[0] =
					mass_point_src[1] =
					mass_point_dst[0] =
					mass_point_dst[1] =
					0.0;

				// Calculating the mass points in both images
				for (size_t i = 0; i < sample_number_; ++i)
				{
					// Get pointer of the current point
					const double *d_idx = points_ptr + cols * sample_[i];

					// Add the coordinates to that of the mass points
					mass_point_src[0] += *(d_idx);
					mass_point_src[1] += *(d_idx + 1);
					mass_point_dst[0] += *(d_idx + 2);
					mass_point_dst[1] += *(d_idx + 3);
				}

				// Get the average
				mass_point_src[0] /= sample_number_;
				mass_point_src[1] /= sample_number_;
				mass_point_dst[0] /= sample_number_;
				mass_point_dst[1] /= sample_number_;

				// Get the mean distance from the mass points
				double average_distance_src = 0.0,
					average_distance_dst = 0.0;
				for (size_t i = 0; i < sample_number_; ++i)
				{
					const double *d_idx = points_ptr + cols * sample_[i];

					const double &x1 = *(d_idx);
					const double &y1 = *(d_idx + 1);
					const double &x2 = *(d_idx + 2);
					const double &y2 = *(d_idx + 3);

					const double dx1 = mass_point_src[0] - x1;
					const double dy1 = mass_point_src[1] - y1;
					const double dx2 = mass_point_dst[0] - x2;
					const double dy2 = mass_point_dst[1] - y2;

					average_distance_src += sqrt(dx1 * dx1 + dy1 * dy1);
					average_distance_dst += sqrt(dx2 * dx2 + dy2 * dy2);
				}

				average_distance_src /= sample_number_;
				average_distance_dst /= sample_number_;

				// Calculate the sqrt(2) / MeanDistance ratios
				const double ratio_src = M_SQRT2 / average_distance_src;
				const double ratio_dst = M_SQRT2 / average_distance_dst;

				// Compute the normalized coordinates
				for (size_t i = 0; i < sample_number_; ++i)
				{
					const double *d_idx = points_ptr + cols * sample_[i];

					const double &x1 = *(d_idx);
					const double &y1 = *(d_idx + 1);
					const double &x2 = *(d_idx + 2);
					const double &y2 = *(d_idx + 3);

					*normalized_points_ptr++ = (x1 - mass_point_src[0]) * ratio_src;
					*normalized_points_ptr++ = (y1 - mass_point_src[1]) * ratio_src;
					*normalized_points_ptr++ = (x2 - mass_point_dst[0]) * ratio_dst;
					*normalized_points_ptr++ = (y2 - mass_point_dst[1]) * ratio_dst;

					for (int j = 4; j < normalized_points_.cols; ++j)
						*normalized_points_ptr++ = *(d_idx + j);
				}

				// Creating the normalizing transformations
				normalizing_transform_source_ << ratio_src, 0, -ratio_src * mass_point_src[0],
					0, ratio_src, -ratio_src * mass_point_src[1],
					0, 0, 1;

				normalizing_transform_destination_ << ratio_dst, 0, -ratio_dst * mass_point_dst[0],
					0, ratio_dst, -ratio_dst * mass_point_dst[1],
					0, 0, 1;
				return true;
			}

			/************** Oriented epipolar constraints ******************/
			OLGA_INLINE void getEpipole(
				Eigen::Vector3d &epipole_, // The epipole 
				const Eigen::Matrix3d &fundamental_matrix_) const
			{
				constexpr double epsilon = 1.9984e-15;
				epipole_ = fundamental_matrix_.row(0).cross(fundamental_matrix_.row(2));

				for (auto i = 0; i < 3; i++)
					if ((epipole_(i) > epsilon) ||
						(epipole_(i) < -epsilon))
						return;
				epipole_ = fundamental_matrix_.row(1).cross(fundamental_matrix_.row(2));
			}

			OLGA_INLINE double getOrientationSignum(
				const Eigen::Matrix3d &fundamental_matrix_,
				const Eigen::Vector3d &epipole_,
				const cv::Mat &point_) const
			{
				double signum1 = fundamental_matrix_(0, 0) * point_.at<double>(2) + fundamental_matrix_(1, 0) * point_.at<double>(3) + fundamental_matrix_(2, 0),
					signum2 = epipole_(1) - epipole_(2) * point_.at<double>(1);
				return signum1 * signum2;
			}

			OLGA_INLINE int isOrientationValid(
				const Eigen::Matrix3d &fundamental_matrix_, // The fundamental matrix
				const cv::Mat &data_, // The data points
				const size_t *sample_, // The sample used for the estimation
				size_t sample_size_) const // The size of the sample
			{
				Eigen::Vector3d epipole; // The epipole in the second image
				getEpipole(epipole, fundamental_matrix_);

				double signum1, signum2;

				// The sample is null pointer, the method is applied to normalized data_
				if (sample_ == nullptr)
				{
					// Get the sign of orientation of the first point_ in the sample
					signum2 = getOrientationSignum(fundamental_matrix_, epipole, data_.row(0));
					for (size_t i = 1; i < sample_size_; i++)
					{
						// Get the sign of orientation of the i-th point_ in the sample
						signum1 = getOrientationSignum(fundamental_matrix_, epipole, data_.row(i));
						// The signs should be equal, otherwise, the fundamental matrix is invalid
						if (signum2 * signum1 < 0)
							return false;
					}
				}
				else
				{
					// Get the sign of orientation of the first point_ in the sample
					signum2 = getOrientationSignum(fundamental_matrix_, epipole, data_.row(sample_[0]));
					for (size_t i = 1; i < sample_size_; i++)
					{
						// Get the sign of orientation of the i-th point_ in the sample
						signum1 = getOrientationSignum(fundamental_matrix_, epipole, data_.row(sample_[i]));
						// The signs should be equal, otherwise, the fundamental matrix is invalid
						if (signum2 * signum1 < 0)
							return false;
					}
				}
				return true;
			}
		};
	}
}