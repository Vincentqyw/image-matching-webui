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

#include "solver_homography_four_point.h"

namespace gcransac
{
	namespace estimator
	{
		// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
		template<class _MinimalSolverEngine,  // The solver used for estimating the model from a minimal sample
			class _NonMinimalSolverEngine> // The solver used for estimating the model from a non-minimal sample
			class RadialHomographyEstimator : public Estimator < cv::Mat, Model >
		{
		protected:
			// Minimal solver engine used for estimating a model from a minimal sample
			const std::shared_ptr<_MinimalSolverEngine> minimal_solver;

			// Non-minimal solver engine used for estimating a model from a bigger than minimal sample
			const std::shared_ptr<_NonMinimalSolverEngine> non_minimal_solver;

		public:
			RadialHomographyEstimator() :
				minimal_solver(std::make_shared<_MinimalSolverEngine>()), // Minimal solver engine used for estimating a model from a minimal sample
				non_minimal_solver(std::make_shared<_NonMinimalSolverEngine>()) // Non-minimal solver engine used for estimating a model from a bigger than minimal sample
			{}
			~RadialHomographyEstimator() {}

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

			static constexpr bool useRadialDistortion()
			{
				return true;
			}

			const _MinimalSolverEngine &getMinimalSolver() const
			{
				return *minimal_solver;
			}

			const _NonMinimalSolverEngine &getNonMinimalSolver() const
			{
				return *non_minimal_solver;
			}

			_MinimalSolverEngine &getMutableMinimalSolver()
			{
				return *minimal_solver;
			}

			_NonMinimalSolverEngine &getMutableNonMinimalSolver()
			{
				return *minimal_solver;
			}

			// The size of a minimal sample_ required for the estimation
			static constexpr size_t maximumMinimalSolutions() {
				return _MinimalSolverEngine::maximumSolutions();
			}

			// The size of a sample when doing inner RANSAC on a non-minimal sample
			OLGA_INLINE size_t inlierLimit() const {
				return 7 * sampleSize();
			}

			// Estimating the model from a minimal sample
			OLGA_INLINE bool estimateModel(
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
			OLGA_INLINE bool estimateModelNonminimal(const cv::Mat& data_, // The data points
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

			OLGA_INLINE double squaredResidual(const cv::Mat& point_,
				const Model& model_) const
			{
				return squaredResidual(point_, model_.descriptor);
			}

			OLGA_INLINE double squaredResidual(const cv::Mat& point_,
				const Eigen::MatrixXd& descriptor_) const
			{
				// The pointer of the current point
				const double* s = reinterpret_cast<double *>(point_.data);

				double l1 = 0, l2 = 0; // The radial distortion parameter of the division model
				// If the descriptor has more than 3 columns, the radial distortion is estimated
				if (descriptor_.cols() > 3)
				{
					l1 = descriptor_(0, 3);
					l2 = descriptor_(1, 3);
				}

				// The coordinates of the point correspondneces
				const double &x1 = *(s + 0),
					&y1 = *(s + 1),
					&x2 = *(s + 2),
					&y2 = *(s + 3);

				// Calculate the homogeneous coordinate implied by the radial distortion
				// Homogeneous coordinate when projecting from the source to the destination image
				const double
					h1 = 1.0 + l1 * (x1 * x1 + y1 * y1);
				// Homogeneous coordinate when projecting from the destinationto the source image
				const double
					h2 = 1.0 + l2 * (x2 * x2 + y2 * y2);

				// Project the point from the source to the destination image
				double t1 = descriptor_(0, 0) * x1 + descriptor_(0, 1) * y1 + descriptor_(0, 2) * h1;
				double t2 = descriptor_(1, 0) * x1 + descriptor_(1, 1) * y1 + descriptor_(1, 2) * h1;
				double t3 = descriptor_(2, 0) * x1 + descriptor_(2, 1) * y1 + descriptor_(2, 2) * h1;

				// Do the homogeneous division
				double d1 = x2 / h2 - (t1 / t3);
				double d2 = y2 / h2 - (t2 / t3);

				// Calculate the re-projection error
				const double kErrorDestinationSource =
					std::sqrt(d1 * d1 + d2 * d2);

				// Get the inverse homography. It is already stored in the descriptor to avoid additional computations
				const Eigen::Matrix3d &kInverseHomography = 
					descriptor_.block<3, 3>(0, 4);

				// Project the point from the destination to the source image
				t1 = kInverseHomography(0, 0) * x2 + kInverseHomography(0, 1) * y2 + kInverseHomography(0, 2) * h2;
				t2 = kInverseHomography(1, 0) * x2 + kInverseHomography(1, 1) * y2 + kInverseHomography(1, 2) * h2;
				t3 = kInverseHomography(2, 0) * x2 + kInverseHomography(2, 1) * y2 + kInverseHomography(2, 2) * h2;

				// Do the homogeneous division
				d1 = x1 / h1 - (t1 / t3);
				d2 = y1 / h1 - (t2 / t3);

				// Calculate the re-projection error
				double kErrorSourceDestination = 
					std::sqrt(d1 * d1 + d2 * d2);

				// Return the squared average symmetric re-projection error
				const double kSymmetricError =
					0.5 * (kErrorDestinationSource + kErrorSourceDestination);

				return kSymmetricError * kSymmetricError;
			}

			OLGA_INLINE double residual(const cv::Mat& point_,
				const Model& model_) const
			{
				return residual(point_, model_.descriptor);
			}

			OLGA_INLINE double residual(const cv::Mat& point_,
				const Eigen::MatrixXd& descriptor_) const
			{
				return sqrt(squaredResidual(point_, descriptor_));
			}

			// A function to decide if the selected sample is degenerate or not
			// before calculating the model parameters
			OLGA_INLINE bool isValidSample(
				const cv::Mat& data_, // All data points
				const size_t *sample_) const // The indices of the selected points
			{
				return true;
			}
		};
	}
}