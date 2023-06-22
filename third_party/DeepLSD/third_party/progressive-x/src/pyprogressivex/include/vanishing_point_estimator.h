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

#include "estimators/estimator.h"
#include "model.h"

#include "GCRANSAC.h"

namespace gcransac
{
	namespace estimator
	{
		// This is the estimator class for estimating a fundamental matrix between two images. A model estimation method and error calculation method are implemented
		template<class _MinimalSolverEngine,  // The solver used for estimating the model from a minimal sample
			class _NonMinimalSolverEngine> // The solver used for estimating the model from a non-minimal sample
			class VanishingPointEstimator : public Estimator < cv::Mat, Model >
		{
		protected:
			// Minimal solver engine used for estimating a model from a minimal sample
			const std::shared_ptr<_MinimalSolverEngine> minimal_solver;

			// Non-minimal solver engine used for estimating a model from a bigger than minimal sample
			const std::shared_ptr<_NonMinimalSolverEngine> non_minimal_solver;

		public:
			VanishingPointEstimator() :
				// Minimal solver engine used for estimating a model from a minimal sample
				minimal_solver(std::make_shared<_MinimalSolverEngine>()),
				// Non-minimal solver engine used for estimating a model from a bigger than minimal sample
				non_minimal_solver(std::make_shared<_NonMinimalSolverEngine>())
			{}

			~VanishingPointEstimator() {}

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

				// The estimation was successfull if at least one model is kept
				return models->size() > 0;
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
				const double residualValue = 
					residual(point_, descriptor_);
				return residualValue * residualValue;
			}

			// The residual function used for deciding which points are inliers
			OLGA_INLINE double residual(const cv::Mat& point_,
				const Model& model_) const
			{
				return residual(point_, model_.descriptor);
			}

			OLGA_INLINE void vec_cross(
				const double &a1, 
				const double &b1, 
				const double &c1,
				const double &a2, 
				const double &b2, 
				const double &c2,
				double& a3, 
				double& b3, 
				double& c3) const
			{
				a3 = b1*c2 - c1*b2;
				b3 = -(a1*c2 - c1*a2);
				c3 = a1*b2 - b1*a2;
			}

			// The residual function used for deciding which points are inliers
			OLGA_INLINE double residual(const cv::Mat& point_,
				const Eigen::MatrixXd& descriptor_) const
			{
				const double* s = 
					reinterpret_cast<double *>(point_.data);
				const double 
					&xs = *s,
					&ys = *(s + 1),
					&xe = *(s + 2),
					&ye = *(s + 3);

				double lx, ly, lz,
					mx = (xs+xe) / 2.0,
					my = (ys+ye) / 2.0;
				
				// Cross product
				lx = my * descriptor_(2) - descriptor_(1);
				ly = -(mx * descriptor_(2) - descriptor_(0));
				lz = mx * descriptor_(1) - my * descriptor_(0);

				const double dist = fabs(lx*xs + ly*ys + lz) / 
					sqrt(lx * lx + ly * ly);
				return dist;
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

				// The eight point fundamental matrix fitting algorithm
				non_minimal_solver->estimateModel(data_,
					sample_,
					sample_number_,
					*models_,
					weights_);

				return true;
			}
		};
	}
}