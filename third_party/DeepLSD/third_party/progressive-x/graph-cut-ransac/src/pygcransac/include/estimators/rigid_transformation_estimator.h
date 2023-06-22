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

#include "solver_rigid_transformation_svd.h"

namespace gcransac
{
	namespace estimator
	{
		// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
		template<class _MinimalSolverEngine,  // The solver used for estimating the model from a minimal sample
			class _NonMinimalSolverEngine> // The solver used for estimating the model from a non-minimal sample
			class RigidTransformationEstimator : public Estimator < cv::Mat, Model >
		{
		protected:
			// Minimal solver engine used for estimating a model from a minimal sample
			const std::shared_ptr<const _MinimalSolverEngine> minimal_solver;

			// Non-minimal solver engine used for estimating a model from a bigger than minimal sample
			const std::shared_ptr<const _NonMinimalSolverEngine> non_minimal_solver;

		public:
			RigidTransformationEstimator() :
				minimal_solver(std::make_shared<const _MinimalSolverEngine>()), // Minimal solver engine used for estimating a model from a minimal sample
				non_minimal_solver(std::make_shared<const _NonMinimalSolverEngine>()) // Non-minimal solver engine used for estimating a model from a bigger than minimal sample
			{}
			~RigidTransformationEstimator() {}

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

			// The size of a minimal sample_ required for the estimation
			static constexpr size_t maximumMinimalSolutions() {
				return _MinimalSolverEngine::maximumSolutions();
			}

			// The size of a sample when doing inner RANSAC on a non-minimal sample
			inline size_t inlierLimit() const {
				return 7 * sampleSize();
			}

			// Estimating the model from a minimal sample
			inline bool estimateModel(
				const cv::Mat& data_, // The data points
				const size_t *sample_, // The sample usd for the estimation
				std::vector<Model>* models_) const // The estimated model parameters
			{
				return minimal_solver->estimateModel(data_, // The data points
					sample_, // The sample used for the estimation
					sampleSize(), // The size of a minimal sample
					*models_); // The estimated model parameters
			}

			// Estimating the model from a non-minimal sample
			inline bool estimateModelNonminimal(const cv::Mat& data_, // The data points
				const size_t *sample_, // The sample used for the estimation
				const size_t &sample_number_, // The size of a minimal sample
				std::vector<Model>* models_,
				const double *weights_ = nullptr) const // The estimated model parameters
			{
				// Return of there are not enough points for the estimation
				if (sample_number_ < nonMinimalSampleSize())
					return false;

				// The four point fundamental matrix fitting algorithm
				if (!non_minimal_solver->estimateModel(data_,
					sample_,
					sample_number_,
					*models_,
					weights_))
					return false;
				return true;
			}

			inline double squaredResidual(const cv::Mat& point_,
				const Model& model_) const
			{
				return squaredResidual(point_, model_.descriptor);
			}

			inline double squaredResidual(const cv::Mat& point_,
				const Eigen::MatrixXd& descriptor_) const
			{
				const double* s = reinterpret_cast<double *>(point_.data);

				const double &x1 = *s;
				const double &y1 = *(s + 1);
				const double &z1 = *(s + 2);
				const double &x2 = *(s + 3);
				const double &y2 = *(s + 4);
				const double &z2 = *(s + 5);

				const double t1 = descriptor_(0, 0) * x1 + descriptor_(1, 0) * y1 + descriptor_(2, 0) * z1 + descriptor_(3, 0);
				const double t2 = descriptor_(0, 1) * x1 + descriptor_(1, 1) * y1 + descriptor_(2, 1) * z1 + descriptor_(3, 1);
				const double t3 = descriptor_(0, 2) * x1 + descriptor_(1, 2) * y1 + descriptor_(2, 2) * z1 + descriptor_(3, 2);
				
				const double dx = x2 - t1;
				const double dy = y2 - t2;
				const double dz = z2 - t3;

				return dx * dx + dy * dy + dz * dz;
			}

			inline double residual(const cv::Mat& point_,
				const Model& model_) const
			{
				return residual(point_, model_.descriptor);
			}

			inline double residual(const cv::Mat& point_,
				const Eigen::MatrixXd& descriptor_) const
			{
				return sqrt(squaredResidual(point_, descriptor_));
			}

			// A function to decide if the selected sample is degenerate or not
			// before calculating the model parameters
			inline bool isValidSample(
				const cv::Mat& data_, // All data points
				const size_t *sample_) const // The indices of the selected points
			{
				return true;
			}

			// Enable a quick check to see if the model is valid. This can be a geometric
			// check or some other verification of the model structure.
			inline bool isValidModel(Model& model,
				const cv::Mat& data_,
				const std::vector<size_t> &inliers_,
				const size_t *minimal_sample_,
				const double threshold_,
				bool &model_updated_) const
			{
				constexpr size_t sample_size = sampleSize();
				const double squared_threshold =
					threshold_ * threshold_;

				// Check the minimal sample if the transformation fits for them well
				for (size_t sample_idx = 0; sample_idx < 3; ++sample_idx)
				{
					const size_t &point_idx = minimal_sample_[sample_idx];
					const double squared_residual =
						squaredResidual(data_.row(point_idx), model);
					if (squared_residual > squared_threshold)
						return false;
				}

				return true;
			}
		};
	}
}