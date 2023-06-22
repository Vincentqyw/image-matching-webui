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
#include "solver_linear_model.h"
#include "model.h"
#include "../neighborhood/grid_neighborhood_graph.h"
#include "../samplers/uniform_sampler.h"

#include "GCRANSAC.h"


namespace gcransac
{
	namespace estimator
	{
		// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
		template<class _MinimalSolverEngine,  // The solver used for estimating the model from a minimal sample
			class _NonMinimalSolverEngine, // The solver used for estimating the model from a non-minimal sample
			size_t _DimensionNumber> // The dimension of the space where the linear model is fitted
			class LinearModelEstimator : public Estimator < cv::Mat, Model >
		{
		protected:
			// Minimal solver engine used for estimating a model from a minimal sample
			const std::shared_ptr<_MinimalSolverEngine> minimal_solver;

			// Non-minimal solver engine used for estimating a model from a bigger than minimal sample
			const std::shared_ptr<_NonMinimalSolverEngine> non_minimal_solver;

		public:
			LinearModelEstimator() :
				// Minimal solver engine used for estimating a model from a minimal sample
				minimal_solver(std::make_shared<_MinimalSolverEngine>()),
				// Non-minimal solver engine used for estimating a model from a bigger than minimal sample
				non_minimal_solver(std::make_shared<_NonMinimalSolverEngine>())
			{}

			~LinearModelEstimator() {}

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
				const double* pointPtr = reinterpret_cast<double*>(point_.data);
				double residual = 0;
				for (size_t coordinateIdx = 0; coordinateIdx < _DimensionNumber; ++coordinateIdx)
					residual += pointPtr[coordinateIdx] * descriptor_(coordinateIdx);
				residual += descriptor_(_DimensionNumber);
				return residual * residual;
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
				const double* pointPtr = reinterpret_cast<double*>(point_.data);
				double residual = 0;
				for (size_t coordinateIdx = 0; coordinateIdx < _DimensionNumber; ++coordinateIdx)
					residual += pointPtr[coordinateIdx] * descriptor_(coordinateIdx);
				residual += descriptor_(_DimensionNumber);
				return std::abs(residual);
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

				cv::Mat normalized_points(sample_number_, data_.cols, data_.type()); // The normalized point coordinates
				Eigen::Matrix<double, _DimensionNumber, 1> mass_point; // The normalizing transformation

				// Normalize the point coordinates to achieve numerical stability when
				// applying the least-squares model fitting.
				if (!normalizePoints(data_, // The data points
					sample_, // The points to which the model will be fit
					sample_number_, // The number of points
					normalized_points, // The normalized point coordinates
					mass_point)) // The normalizing transformation
					return false;

				// The eight point fundamental matrix fitting algorithm
				non_minimal_solver->estimateModel(
					normalized_points,
					nullptr,
					sample_number_,
					*models_,
					weights_);

				for (auto &model : *models_)
				{
					// Denormalizing the estimated linear model.
					// It is represented by its implicit form a x + by + ... + w = 0, where w represents the offset from the origin.
					// Parameters [a, b, ...] are the coordinates of the normal. Only w has to be recalculated in the unnormalized space.
					// It is calculated from the mass point.
					auto& descriptor = model.descriptor;
					double w = 0;
					for (size_t coordinateIdx = 0; coordinateIdx < _DimensionNumber; ++coordinateIdx)
						w -= mass_point(coordinateIdx) * descriptor(coordinateIdx);

					// Replace the offset parameter in the descriptor
					descriptor(_DimensionNumber) = w;
				}

				return true;
			}

			inline bool normalizePoints(
				const cv::Mat& data_, // The data points
				const size_t *sample_, // The points to which the model will be fit
				const size_t &sampleNumber_,// The number of points
				cv::Mat &normalizedPoints_, // The normalized point coordinates
				Eigen::Matrix<double, _DimensionNumber, 1> &massPoint_) const // The normalizing transformation in the second image
			{
				const size_t kColumns = data_.cols;
				double *normalizedPointsPtr = reinterpret_cast<double *>(normalizedPoints_.data);
				const double *kPointsPtr = reinterpret_cast<double *>(data_.data);

				// Reseting the mass point's coordinates to zero
				massPoint_.setZero();

				// Calculating the mass points in both images
				for (size_t i = 0; i < sampleNumber_; ++i)
				{
					size_t sampleIdx = sample_ == nullptr ?
						i : sample_[i];

					// Get pointer of the current point
					const double *coordinateIdx = kPointsPtr + kColumns * sampleIdx;

					// Add the coordinates to that of the mass points
					for (size_t col = 0; col < _DimensionNumber; ++col)
					{
						massPoint_(col) += *(coordinateIdx);
						*(normalizedPointsPtr++) = *(coordinateIdx);
						++coordinateIdx;
					}

					// Copy the rest of the matrix if there is
					for (size_t col = _DimensionNumber; col < kColumns; ++col)
					{
						*(normalizedPointsPtr++) = *(coordinateIdx);
						++coordinateIdx;
					}
				}

				// Get the average
				massPoint_ /= sampleNumber_;
				
				// Get the mean distance from the mass points
				normalizedPointsPtr = reinterpret_cast<double*>(normalizedPoints_.data);
				double averageDistance = 0.0, squaredDistance;
				for (size_t i = 0; i < sampleNumber_; ++i)
				{
					double *coordinateIdx = normalizedPointsPtr + kColumns * i;

					squaredDistance = 0.0;
					for (size_t col = 0; col < _DimensionNumber; ++col)
					{
						*(coordinateIdx) -= massPoint_(col);
						squaredDistance += *(coordinateIdx) * *(coordinateIdx);
						*(++coordinateIdx);
					}

					averageDistance += std::sqrt(squaredDistance);
				}

				averageDistance /= sampleNumber_;

				// Calculate the sqrt(DimensionNumber) / MeanDistance ratios
				static const double sqrtDimension = std::sqrt(_DimensionNumber);
				const double ratio = sqrtDimension / averageDistance;

				// Compute the normalized coordinates
				for (size_t i = 0; i < sampleNumber_; ++i)
					for (size_t i = 0; i < normalizedPoints_.cols; ++i)
						*normalizedPointsPtr++ *= ratio;

				return true;
			}

		};
	}
}