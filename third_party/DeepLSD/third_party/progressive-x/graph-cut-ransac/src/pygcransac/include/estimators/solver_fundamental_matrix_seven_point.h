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
			class FundamentalMatrixSevenPointSolver : public SolverEngine
			{
			public:
				FundamentalMatrixSevenPointSolver()
				{
				}

				~FundamentalMatrixSevenPointSolver()
				{
				}

				// Determines if there is a chance of returning multiple models
				// when function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return maximumSolutions() > 1;
				}

				// The maximum number of solutions returned by the estimator
				static constexpr size_t maximumSolutions()
				{
					return 3;
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 7;
				}

				OLGA_INLINE bool estimateModel(
					const cv::Mat& data_, // The set of data points
					const size_t *sample_, // The sample used for the estimation
					size_t sample_number_, // The size of the sample
					std::vector<Model> &models_, // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point
			};

			OLGA_INLINE bool FundamentalMatrixSevenPointSolver::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				Eigen::MatrixXd coefficients(sample_number_, 9);
				const double *data_ptr = reinterpret_cast<double *>(data_.data);
				const int cols = data_.cols;
				double c[4];
				double t0, t1, t2;
				int i, n;

				// Form a linear system: i-th row of A(=a) represents
				// the equation: (m2[i], 1)'*F*(m1[i], 1) = 0
				double weight = 1.0;
				for (i = 0; i < 7; i++)
				{
					const int sample_idx = sample_[i];
					const int offset = cols * sample_idx;

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
						weight = weights_[sample_idx];

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
									
				Eigen::Matrix<double, 9, 1> f1, f2;

				// For the minimal problem, the matrix is small and, thus, fullPivLu decomposition is significantly
				// faster than both JacobiSVD and BDCSVD methods.
				// https://eigen.tuxfamily.org/dox/group__DenseDecompositionBenchmark.html
				if (sample_number_ == sampleSize())
				{
					const Eigen::FullPivLU<Eigen::MatrixXd> lu(coefficients.transpose() * coefficients);
					if (lu.dimensionOfKernel() != 2) 
						return false;

					const Eigen::Matrix<double, 9, 2> null_space = 
						lu.kernel();

					f1 = null_space.col(0);
					f2 = null_space.col(1);
				}
				else
				{
					// A*(f11 f12 ... f33)' = 0 is singular (7 equations for 9 variables), so
					// the solution is linear subspace of dimensionality 2.
					// => use the last two singular std::vectors as a basis of the space
					// (according to SVD properties)
					Eigen::JacobiSVD<Eigen::MatrixXd> svd(
						// Theoretically, it would be faster to apply SVD only to matrix coefficients, but
						// multiplication is faster than SVD in the Eigen library. Therefore, it is faster
						// to apply SVD to a smaller matrix.
						coefficients.transpose() * coefficients,
						Eigen::ComputeFullV);
					f1 = svd.matrixV().block<9, 1>(0, 7);
					f2 = svd.matrixV().block<9, 1>(0, 8);
				}

				// f1, f2 is a basis => lambda*f1 + mu*f2 is an arbitrary f. matrix.
				// as it is determined up to a scale, normalize lambda & mu (lambda + mu = 1),
				// so f ~ lambda*f1 + (1 - lambda)*f2.
				// use the additional constraint det(f) = det(lambda*f1 + (1-lambda)*f2) to find lambda.
				// it will be a cubic equation.
				// find c - polynomial coefficients.
				for (i = 0; i < 9; i++)
					f1[i] -= f2[i];

				t0 = f2[4] * f2[8] - f2[5] * f2[7];
				t1 = f2[3] * f2[8] - f2[5] * f2[6];
				t2 = f2[3] * f2[7] - f2[4] * f2[6];

				c[0] = f2[0] * t0 - f2[1] * t1 + f2[2] * t2;

				c[1] = f1[0] * t0 - f1[1] * t1 + f1[2] * t2 -
					f1[3] * (f2[1] * f2[8] - f2[2] * f2[7]) +
					f1[4] * (f2[0] * f2[8] - f2[2] * f2[6]) -
					f1[5] * (f2[0] * f2[7] - f2[1] * f2[6]) +
					f1[6] * (f2[1] * f2[5] - f2[2] * f2[4]) -
					f1[7] * (f2[0] * f2[5] - f2[2] * f2[3]) +
					f1[8] * (f2[0] * f2[4] - f2[1] * f2[3]);

				t0 = f1[4] * f1[8] - f1[5] * f1[7];
				t1 = f1[3] * f1[8] - f1[5] * f1[6];
				t2 = f1[3] * f1[7] - f1[4] * f1[6];

				c[2] = f2[0] * t0 - f2[1] * t1 + f2[2] * t2 -
					f2[3] * (f1[1] * f1[8] - f1[2] * f1[7]) +
					f2[4] * (f1[0] * f1[8] - f1[2] * f1[6]) -
					f2[5] * (f1[0] * f1[7] - f1[1] * f1[6]) +
					f2[6] * (f1[1] * f1[5] - f1[2] * f1[4]) -
					f2[7] * (f1[0] * f1[5] - f1[2] * f1[3]) +
					f2[8] * (f1[0] * f1[4] - f1[1] * f1[3]);

				c[3] = f1[0] * t0 - f1[1] * t1 + f1[2] * t2;
				
				// Check if the sum of the polynomical coefficients is close to zero. 
				// In this case "psolve.realRoots(real_roots)" gets into an infinite loop.
				if (fabs(c[0]+c[1]+c[2]+c[3]) < 1e-9) 
					return false;

				// solve the cubic equation; there can be 1 to 3 roots ...
				Eigen::Matrix<double, 4, 1> polynomial;
				for (auto i = 0; i < 4; ++i)
					polynomial(i) = c[i];
				Eigen::PolynomialSolver<double, 3> psolve(polynomial);

				std::vector<double> real_roots;
				psolve.realRoots(real_roots);

				n = real_roots.size();
				if (n < 1 || n > 3)
					return false;

				double f[8];
				for (const double &root : real_roots)
				{
					// for each root form the fundamental matrix
					double lambda = root, 
						mu = 1.;
					double s = f1[8] * root + f2[8];

					// normalize each matrix, so that F(3,3) (~fmatrix[8]) == 1
					if (fabs(s) > std::numeric_limits<double>::epsilon())
					{
						mu = 1.0f / s;
						lambda *= mu;

						for (auto i = 0; i < 8; ++i)
							f[i] = f1[i] * lambda + f2[i] * mu;

						FundamentalMatrix model;
						model.descriptor << f[0], f[1], f[2],
							f[3], f[4], f[5],
							f[6], f[7], 1.0;
						models_.push_back(model);
					}
				}

				return true;
			}
		}
	}
}