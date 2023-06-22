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

#include "estimators/solver_engine.h"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class VanishingPointTwoLineSolver : public SolverEngine
			{
			public:
				VanishingPointTwoLineSolver()
				{
				}

				~VanishingPointTwoLineSolver()
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
					return 2;
				}

				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				OLGA_INLINE bool estimateModel(
					const cv::Mat& data_, // The set of data points
					const size_t *sample_, // The sample used for the estimation
					size_t sample_number_, // The size of the sample
					std::vector<Model> &models_, // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point
					
				OLGA_INLINE void vec_cross(
					const double &a1, 
					const double &b1, 
					const double &c1,
					const double &a2, 
					const double &b2, 
					const double &c2,
					double& a3, 
					double& b3, 
					double& c3) const;

				OLGA_INLINE void vec_norm(
					double& a, 
					double& b, 
					double& c) const;
			};

			OLGA_INLINE void VanishingPointTwoLineSolver::vec_cross(
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

			OLGA_INLINE void VanishingPointTwoLineSolver::vec_norm(
				double& a, 
				double& b, 
				double& c) const
			{
				double len = sqrt(a*a+b*b+c*c);
				a/=len; 
				b/=len; 
				c/=len;
			}

			OLGA_INLINE bool VanishingPointTwoLineSolver::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				if (sample_ == nullptr)
					sample_number_ = data_.rows;

				const double *data_ptr = reinterpret_cast<double *>(data_.data);
				const int cols = data_.cols;
				double weight = 1.0;
				size_t offset;
				Model model;

				// Solving the problem from a minimal sample requires
				// calculating the intersection of two 2D lines.
				// This can be easily done in closed form.
				if (sample_number_ == sampleSize())
				{
					size_t offset1, offset2;

					if (sample_ == nullptr)
					{
						offset1 = 0;
						offset2 = cols;
					} else
					{
						offset1 = cols * sample_[0];
						offset2 = cols * sample_[1];
					}

					const double 
						&xs0 = data_ptr[offset1],
						&ys0 = data_ptr[offset1 + 1],
						&xe0 = data_ptr[offset1 + 2],
						&ye0 = data_ptr[offset1 + 3],
						&xs1 = data_ptr[offset2],
						&ys1 = data_ptr[offset2 + 1],
						&xe1 = data_ptr[offset2 + 2],
						&ye1 = data_ptr[offset2 + 3];

					double l0[3],l1[3],v[3];
					vec_cross(xs0, ys0, 1,
						xe0, ye0, 1,
						l0[0], l0[1], l0[2]);
					vec_cross(xs1, ys1, 1,
						xe1, ye1, 1,
						l1[0], l1[1], l1[2]);
					vec_cross(l0[0], l0[1], l0[2],
						l1[0], l1[1], l1[2],
						v[0], v[1], v[2]);
					vec_norm(v[0], v[1], v[2]);

					model.descriptor.resize(3, 1);
					model.descriptor << v[0], v[1], v[2];
				} else
				{
					Eigen::MatrixXd A(sample_number_, 3);

					for (size_t sampleIdx = 0; sampleIdx < sample_number_; ++sampleIdx)
					{
						if (sample_ == nullptr)
						{
							offset = cols * sampleIdx;							
							if (weights_ != nullptr)
								weight = weights_[sampleIdx];
						} else
						{
							offset = cols * sample_[sampleIdx];							
							if (weights_ != nullptr)
								weight = weights_[sample_[sampleIdx]];
						}

						const double
							&x0 = data_ptr[offset],
							&y0 = data_ptr[offset + 1],
							&x1 = data_ptr[offset + 2],
							&y1 = data_ptr[offset + 3];

						const double
							mx = (x0 + x1) / 2.0,
							my = (y0 + y1) / 2.0,
							mz = 1.0;

						A.row(sampleIdx) << y0 * mz - my, mx - x0 * mz, x0 * my - y0 * mx;
						A.row(sampleIdx) *= weight;
					}

					// Estimating the vanishing point from the overdetermined linear system
					/*const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(
						A.transpose() * A);
					const Eigen::MatrixXd &Q = qr.matrixQ();
					model.descriptor = Q.rightCols<1>();*/

					Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(A.transpose() * A);
					const Eigen::ArrayXcd &singularVals = eigensolver.eigenvalues();

					Eigen::MatrixXd::Index minRow;
					const double &min =
						singularVals.real().minCoeff(&minRow); // smallest one of the three eigenvalues

					model.descriptor = eigensolver.eigenvectors().col(minRow);
					model.descriptor.normalize();
				}

				models_.push_back(model);
				return true;
			}
		}
	}
}