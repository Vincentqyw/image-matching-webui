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
#include "model.h"

namespace gcransac
{
	namespace estimator
	{
		// This is the estimator class for estimating a essential matrix between two images. A model_ estimation method and error calculation method are implemented
		template<class _MinimalSolverEngine,  // The solver used for estimating the model_ from a minimal sample
			class _NonMinimalSolverEngine> // The solver used for estimating the model_ from a non-minimal sample
			class EssentialMatrixEstimator : public Estimator < cv::Mat, Model >
		{
		protected:
			// Minimal solver engine used for estimating a model_ from a minimal sample
			const std::shared_ptr<const _MinimalSolverEngine> minimal_solver;

			// Non-minimal solver engine used for estimating a model_ from a bigger than minimal sample
			const std::shared_ptr<const _NonMinimalSolverEngine> non_minimal_solver;

			const Eigen::Matrix3d intrinsics_src, // The intrinsic parameters of the source camera
				intrinsics_dst; // The intrinsic parameters of the destination camera

			// The lower bound of the inlier ratio which is required to pass the validity test.
			// The validity test measures what proportion of the inlier (by Sampson distance) is inlier
			// when using symmetric epipolar distance. 
			const double minimum_inlier_ratio_in_validity_check;

			// The ratio of points used when the non-minimal model fitting method returns multiple models.
			// The selected points are not used for the estimation but for selecting the best model
			// from the set of estimated ones. 
			const double point_ratio_for_selecting_from_multiple_models;
			
		public:
			EssentialMatrixEstimator(Eigen::Matrix3d intrinsics_src_, // The intrinsic parameters of the source camera
				Eigen::Matrix3d intrinsics_dst_,  // The intrinsic parameters of the destination camera
				const double minimum_inlier_ratio_in_validity_check_ = 0.1,
				const double point_ratio_for_selecting_from_multiple_models_ = 0.05) :
				// The intrinsic parameters of the source camera
				intrinsics_src(intrinsics_src_),
				// The intrinsic parameters of the destination camera
				intrinsics_dst(intrinsics_dst_),
				// Minimal solver engine used for estimating a model from a minimal sample
				minimal_solver(std::make_shared<const _MinimalSolverEngine>()),
				// Non-minimal solver engine used for estimating a model from a bigger than minimal sample
				non_minimal_solver(std::make_shared<const _NonMinimalSolverEngine>()),
				// The lower bound of the inlier ratio which is required to pass the validity test.
				// It is clamped to be in interval [0, 1].
				minimum_inlier_ratio_in_validity_check(std::clamp(minimum_inlier_ratio_in_validity_check_, 0.0, 1.0)),
				// The ratio of points used when the non-minimal model fitting method returns multiple models.
				// The selected points are not used for the estimation but for selecting the best model
				// from the set of estimated ones. 
				point_ratio_for_selecting_from_multiple_models(std::clamp(point_ratio_for_selecting_from_multiple_models_, 0.0, 1.0))
			{}
			~EssentialMatrixEstimator() {}

			// The size of a non-minimal sample required for the estimation
			static constexpr size_t nonMinimalSampleSize() {
				return _NonMinimalSolverEngine::sampleSize();
			}

			// The size of a minimal sample_ required for the estimation
			static constexpr size_t sampleSize() {
				return _MinimalSolverEngine::sampleSize();
			}

			// The size of a minimal sample_ required for the estimation
			static constexpr size_t maximumMinimalSolutions() {
				return _MinimalSolverEngine::maximumSolutions();
			}

			// A flag deciding if the points can be weighted when the non-minimal fitting is applied 
			static constexpr bool isWeightingApplicable() {
				return true;
			}

			// The size of a sample_ when doing inner RANSAC on a non-minimal sample
			OLGA_INLINE size_t inlierLimit() const {
				return 7 * sampleSize();
			}

			// Estimating the essential matrix from a minimal sample
			OLGA_INLINE bool estimateModel(const cv::Mat& data, // The data_ points
				const size_t *sample, // The selected sample_ which will be used for estimation
				std::vector<Model>* models) const // The estimated model_ parameters
			{
				constexpr size_t sample_size = sampleSize(); // The size of a minimal sample

				// Estimating the model_ parameters by the solver engine
				if (!minimal_solver->estimateModel(data, // The data points
					sample, // The selected sample which will be used for estimation
					sample_size, // The size of a minimal sample required for the estimation
					*models)) // The estimated model_ parameters
					return false;

				/* Orientation constraint check */
				for (short model_idx = models->size() - 1; model_idx >= 0; --model_idx)
					if (!isOrientationValid(models->at(model_idx).descriptor,
						data, // The data points
						sample, // The selected sample which will be used for estimation
						sample_size)) // The size of a minimal sample required for the estimation
						models->erase(models->begin() + model_idx); // Delete the model if the orientation constraint does not hold

				// Return true, if at least one model_ is kept
				return models->size() > 0;
			}

			// The squared sampson distance between a point_ correspondence and an essential matrix
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

			// The squared residual function used for deciding which points are inliers
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
							return true;
				// If the algorithm has not terminated earlier, there are not enough inliers_.
				return false;
			}
			
			// Estimating the model from a non-minimal sample
			OLGA_INLINE bool estimateModelNonminimal(
				const cv::Mat& data_,
				const size_t *sample_,
				const size_t &sample_number_,
				std::vector<Model>* models_,
				const double *weights_ = nullptr) const
			{
				if (sample_number_ < nonMinimalSampleSize())
					return false;

				// Number of points used for selecting the best model out of the estimated ones.
				// In case the solver return a single model, 0 points are not used for the estimation.
				size_t points_not_used = 0; 
				if constexpr (_NonMinimalSolverEngine::returnMultipleModels())
					points_not_used = 
						MAX(1, std::round(sample_number_ * point_ratio_for_selecting_from_multiple_models));

				// Number of points used for the estimation
				const size_t points_used = sample_number_ - points_not_used;

				// The container where the estimated models are stored
				std::vector<Model> temp_models;

				// The eight point fundamental matrix fitting algorithm
				if (!non_minimal_solver->estimateModel(data_,
					sample_,
					points_used,
					temp_models,
					weights_))
					return false;
				
				// Denormalizing the estimated essential matrices and selecting the best
				// if multiple ones have been estimated.
				const size_t &model_number = temp_models.size();
				double best_residual = std::numeric_limits<double>::max();
				size_t best_model_idx = 0;
				for (size_t model_idx = 0; model_idx < model_number; ++model_idx)
				{
					// Get the reference of the current model
					Model &model = temp_models[model_idx];

					// Calculate the sum of squared residuals from the selected point set
					double current_residual = 0.0;
					if (model_number > 1)
						for (size_t sample_idx = points_used; sample_idx < sample_number_; ++sample_idx)
							current_residual += squaredResidual(data_.row(sample_[sample_idx]), model.descriptor);

					// Update the best model index if the sum of squared residuals measured from the correspondences
					// not used in the estimation is smaller than the previous best.
					if (current_residual < best_residual)
					{
						best_residual = current_residual;
						best_model_idx = model_idx;
					}
				}

				// Store the best model
				models_->emplace_back(temp_models[best_model_idx]);
				Model &best_model = models_->back();

				// Normalizing the essential matrix elements
				best_model.descriptor.normalize();
				if (best_model.descriptor(2, 2) < 0)
					best_model.descriptor = -best_model.descriptor;

				return true;
			}
			
			/************** Oriented epipolar constraints ******************/
			OLGA_INLINE void getEpipole(
				Eigen::Vector3d &epipole_, // The epipole 
				const Eigen::Matrix3d &essential_matrix_) const
			{
				constexpr double epsilon = 1.9984e-15;
				epipole_ = essential_matrix_.row(0).cross(essential_matrix_.row(2));

				for (auto i = 0; i < 3; i++)
					if ((epipole_(i) > epsilon) ||
						(epipole_(i) < -epsilon))
						return;
				epipole_ = essential_matrix_.row(1).cross(essential_matrix_.row(2));
			}

			OLGA_INLINE double getOrientationSignum(
				const Eigen::Matrix3d &essential_matrix_,
				const Eigen::Vector3d &epipole_,
				const cv::Mat &point_) const
			{
				double signum1 = essential_matrix_(0, 0) * point_.at<double>(2) + essential_matrix_(1, 0) * point_.at<double>(3) + essential_matrix_(2, 0),
					signum2 = epipole_(1) - epipole_(2) * point_.at<double>(1);
				return signum1 * signum2;
			}

			OLGA_INLINE int isOrientationValid(
				const Eigen::Matrix3d &essential_matrix_, // The fundamental matrix
				const cv::Mat &data_, // The data points
				const size_t *sample_, // The sample used for the estimation
				const size_t &sample_size_) const // The size of the sample
			{
				Eigen::Vector3d epipole; // The epipole in the second image
				getEpipole(epipole, essential_matrix_);

				double signum1, signum2;

				// The sample is null pointer, the method is applied to normalized data_
				if (sample_ == nullptr)
				{
					// Get the sign of orientation of the first point_ in the sample
					signum2 = getOrientationSignum(essential_matrix_, epipole, data_.row(0));
					for (size_t i = 1; i < sample_size_; i++)
					{
						// Get the sign of orientation of the i-th point_ in the sample
						signum1 = getOrientationSignum(essential_matrix_, epipole, data_.row(i));
						// The signs should be equal, otherwise, the fundamental matrix is invalid
						if (signum2 * signum1 < 0)
							return false;
					}
				}
				else
				{
					// Get the sign of orientation of the first point_ in the sample
					signum2 = getOrientationSignum(essential_matrix_, epipole, data_.row(sample_[0]));
					for (size_t i = 1; i < sample_size_; i++)
					{
						// Get the sign of orientation of the i-th point_ in the sample
						signum1 = getOrientationSignum(essential_matrix_, epipole, data_.row(sample_[i]));
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