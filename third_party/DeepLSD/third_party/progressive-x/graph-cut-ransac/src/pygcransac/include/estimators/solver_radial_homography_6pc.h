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

#include <Eigen/Eigen>
#include "solver_engine.h"
#include "fundamental_estimator.h"
#include "unsupported/Eigen/Polynomials"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class RadialHomography6PC : public SolverEngine
			{
			public:
				RadialHomography6PC()
				{
				}

				~RadialHomography6PC()
				{
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 6;
				}
				
				// Determines if there is a chance of returning multiple models
				// the function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return true;
				}

				static constexpr char * getName()
				{
					return "H6l1l2";
				}

				// The maximum number of solutions returned by the estimator
				static constexpr size_t maximumSolutions()
				{
					return 2;
				}

				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				OLGA_INLINE bool estimateModel(
					const cv::Mat &data_,					 // The set of data points
					const size_t *sample_,					 // The sample used for the estimation
					size_t sample_number_,					 // The size of the sample
					std::vector<Model> &models_,			 // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point

				void setIntrinsics(
					const Eigen::Matrix3d& K1_,
					const Eigen::Matrix3d& K2_)
				{
					K1 = K1_;
					K2 = K2_;
				}

			protected:
				Eigen::Matrix3d K1, K2;
			};

			OLGA_INLINE bool RadialHomography6PC::estimateModel(
				const cv::Mat &data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				using namespace Eigen;
				
				Eigen::MatrixXd X(sample_number_, 2),
					U(sample_number_, 2);

				for (size_t sampleIdx = 0; sampleIdx < sample_number_; ++sampleIdx)
				{
					const size_t &pointIdx = sample_[sampleIdx];

					X(sampleIdx, 0) = data_.at<double>(pointIdx, 0);
					X(sampleIdx, 1) = data_.at<double>(pointIdx, 1);
					U(sampleIdx, 0) = data_.at<double>(pointIdx, 2);
					U(sampleIdx, 1) = data_.at<double>(pointIdx, 3);
				}

				Eigen::MatrixXd M(sample_number_, 8);
				Eigen::MatrixXd u2 = U.col(0).array().square() + U.col(1).array().square();
				
				M.col(0) = -X.col(1).array() * U.col(0).array();
				M.col(1) = -X.col(1).array() * U.col(1).array();
				M.col(2) = -X.col(1);
				M.col(3) = X.col(0).array() * U.col(0).array();
				M.col(4) = X.col(0).array() * U.col(1).array();
				M.col(5) = X.col(0);
				M.col(6) = -X.col(1).array() * u2.array();
				M.col(7) = X.col(0).array() * u2.array();

				Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::FullPivHouseholderQRPreconditioner> Svd1(M, Eigen::ComputeFullV);
				const Eigen::Matrix<double, 8, 8> &V1 = Svd1.matrixV();

				double a = -V1(2, 6) * V1(7, 6) + V1(5, 6) * V1(6, 6);
				double b = -V1(2, 6) * V1(7, 7) - V1(2, 7) * V1(7, 6) + V1(5, 6) * V1(6, 7) + V1(5, 7) * V1(6, 6);
				double c = -V1(2, 7) * V1(7, 7) + V1(5, 7) * V1(6, 7);
				double d = b * b - 4.0 * a * c;

				int nsols = 0;
				Eigen::Matrix<double, 2, 1> rs;

				if (abs(d) < std::numeric_limits<double>::epsilon())
				{
					nsols = 1;
					rs(0) = (-b) / (2.0 * a);
				}
				else if (d > 0.0)
				{
					nsols = 2;
					double d2 = std::sqrt(d);
					rs(0) = (-b + d2) / (2.0 * a);
					rs(1) = (-b - d2) / (2.0 * a);
				}
				else
				{
					return false;
				}

				Eigen::MatrixXd x2 = X.col(0).array().square() + X.col(1).array().square();
				Eigen::MatrixXd u3(sample_number_, 1), r(sample_number_, 1);
				Eigen::Matrix<double, 8, 1> n;
				Eigen::MatrixXd T(sample_number_, 5);
				T.col(0) = -M.col(3);
				T.col(1) = -M.col(4);

				for (int i = 0; i < nsols; i++)
				{
					n = rs(i) * V1.col(6) + V1.col(7);

					RadialHomography model;
					double l2 = n(6) / n(2);
					
					u3 = u3.Ones(sample_number_, 1) + l2 * u2;
					r = n(0) * U.col(0) + n(1) * U.col(1) + n(2) * u3;

					T.col(2) = -X.col(0).array() * u3.array();
					T.col(3) = x2.array() * r.array();
					T.col(4) = r;

					Eigen::JacobiSVD<Eigen::MatrixXd> Svd2(T, Eigen::ComputeFullV);
					Eigen::MatrixXd v2 = Svd2.matrixV().col(4);

					v2.block<4,1>(0,0) /= v2(4,0);
					double l1 = v2(3);

					if (l1 < -10 || l1 > 2)
						continue;
					if (l2 < -10 || l2 > 2)
						continue;

					model.descriptor(0, 0) = n(0);
					model.descriptor(0, 1) = n(3);
					model.descriptor(0, 2) = v2(0);
					model.descriptor(1, 0) = n(1);
					model.descriptor(1, 1) = n(4);
					model.descriptor(1, 2) = v2(1);
					model.descriptor(2, 0) = n(2);
					model.descriptor(2, 1) = n(5);
					model.descriptor(2, 2) = v2(2);

					model.descriptor(0, 3) = l1;
					model.descriptor(1, 3) = l2;

					model.descriptor.block<3, 3>(0, 0).transposeInPlace();
					model.descriptor.block<3, 3>(0, 4) = model.descriptor.block<3, 3>(0, 0);
					model.descriptor.block<3, 3>(0, 0) = model.descriptor.block<3, 3>(0, 0).inverse().eval();

					models_.push_back(model);
				}
				
				return models_.size() > 0;
			}

		} // namespace solver
	}	  // namespace estimator
} // namespace gcransac
