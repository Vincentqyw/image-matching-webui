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

#include "solver_engine.h"
#include "fundamental_estimator.h"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class FundamentalMatrixEightPointSolver : public SolverEngine
			{
			public:
				FundamentalMatrixEightPointSolver()
				{
				}

				~FundamentalMatrixEightPointSolver()
				{
				}

				// Determines if there is a chance of returning multiple models
				// the function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return maximumSolutions() > 1;
				}

				// The maximum number of solutions returned by the estimator
				static constexpr size_t maximumSolutions()
				{
					return 1;
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 8;
				}

				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				OLGA_INLINE bool estimateModel(
					const cv::Mat& data_, // The set of data points
					const size_t *sample_, // The sample used for the estimation
					size_t sample_number_, // The size of the sample
					std::vector<Model> &models_, // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point
			};

			OLGA_INLINE bool FundamentalMatrixEightPointSolver::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				if (sample_ == nullptr)
					sample_number_ = data_.rows;

				Eigen::MatrixXd coefficients(sample_number_, 9);
				const double *data_ptr = reinterpret_cast<double *>(data_.data);
				const int cols = data_.cols;

				// form a linear system: i-th row of A(=a) represents
				// the equation: (m2[i], 1)'*F*(m1[i], 1) = 0
				double weight = 1.0;
				size_t offset;
				for (size_t i = 0; i < sample_number_; i++)
				{
					if (sample_ == nullptr)
					{
						offset = cols * i;
						if (weights_ != nullptr)
							weight = weights_[i];
					} 
					else
					{
						offset = cols * sample_[i];
						if (weights_ != nullptr)
							weight = weights_[sample_[i]];
					}

					const double
						x0 = data_ptr[offset],
						y0 = data_ptr[offset + 1],
						x1 = data_ptr[offset + 2],
						y1 = data_ptr[offset + 3];

					// If not weighted least-squares is applied
					if (weights_ == nullptr)
					{
						coefficients(i, 0) = x1 * x0;
						coefficients(i, 1) = x1 * y0;
						coefficients(i, 2) = x1;
						coefficients(i, 3) = y1 * x0;
						coefficients(i, 4) = y1 * y0;
						coefficients(i, 5) = y1;
						coefficients(i, 6) = x0;
						coefficients(i, 7) = y0;
						coefficients(i, 8) = 1;
					}
					else
					{
						// Precalculate these values to avoid calculating them multiple times
						const double
							weight_times_x0 = weight * x0,
							weight_times_y0 = weight * y0,
							weight_times_x1 = weight * x1,
							weight_times_y1 = weight * y1;

						coefficients(i, 0) = weight_times_x1 * x0;
						coefficients(i, 1) = weight_times_x1 * y0;
						coefficients(i, 2) = weight_times_x1;
						coefficients(i, 3) = weight_times_y1 * x0;
						coefficients(i, 4) = weight_times_y1 * y0;
						coefficients(i, 5) = weight_times_y1;
						coefficients(i, 6) = weight_times_x0;
						coefficients(i, 7) = weight_times_y0;
						coefficients(i, 8) = weight;
					}
				}

				// A*(f11 f12 ... f33)' = 0 is singular (8 equations for 9 variables), so
				// the solution is linear subspace of dimensionality 1.
				// => use the last two singular std::vectors as a basis of the space
				// (according to SVD properties)
				const Eigen::FullPivHouseholderQR<Eigen::MatrixXd> qr(
					coefficients.transpose() * coefficients);
				const Eigen::MatrixXd& Q = qr.matrixQ();
				const Eigen::Matrix<double, 9, 1>& null_space =
					Q.rightCols<1>();

				FundamentalMatrix model;
				model.descriptor << null_space(0), null_space(1), null_space(2),
					null_space(3), null_space(4), null_space(5),
					null_space(6), null_space(7), null_space(8);
				models_.push_back(model);
				return true;
			}
		}
	}
}