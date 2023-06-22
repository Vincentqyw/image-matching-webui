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
#include "linear_model_estimator.h"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			template <size_t _DimensionNumber>
			class LinearModelSolver : public SolverEngine
			{
			public:
				LinearModelSolver()
				{
				}

				~LinearModelSolver()
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
					return 1;
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return _DimensionNumber;
				}

				OLGA_INLINE bool estimateModel(
					const cv::Mat& data_, // The set of data points
					const size_t *sample_, // The sample used for the estimation
					size_t sample_number_, // The size of the sample
					std::vector<Model> &models_, // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point

			protected:
				OLGA_INLINE bool estimate2DLine(
					const cv::Mat& data_, // The set of data points
					const size_t* sample_, // The sample used for the estimation
					size_t sample_number_, // The size of the sample
					std::vector<Model>& models_, // The estimated model parameters
					const double* weights_ = nullptr) const; // The weight for each point

				OLGA_INLINE bool estimate3DPlane(
					const cv::Mat& data_, // The set of data points
					const size_t* sample_, // The sample used for the estimation
					size_t sample_number_, // The size of the sample
					std::vector<Model>& models_, // The estimated model parameters
					const double* weights_ = nullptr) const; // The weight for each point

			};

			template <size_t _DimensionNumber>
			OLGA_INLINE bool LinearModelSolver<_DimensionNumber>::estimate3DPlane(
				const cv::Mat& data_,
				const size_t* sample_,
				size_t sampleNumber_,
				std::vector<Model>& models_,
				const double* weights_) const
			{
				const double* dataPtr = reinterpret_cast<double*>(data_.data);
				const int kColumns = data_.cols;

				// The pointer to the first point
				const double* point1Ptr = dataPtr + kColumns * sample_[0];
				// The pointer to the second point
				const double* point2Ptr = dataPtr + kColumns * sample_[1];
				// The pointer to the second point
				const double* point3Ptr = dataPtr + kColumns * sample_[2];

				// The tangent directions of the plane
				Eigen::Vector3d tangent1, tangent2;
				tangent1 <<
					point2Ptr[0] - point1Ptr[0],
					point2Ptr[1] - point1Ptr[1],
					point2Ptr[2] - point1Ptr[2];
				tangent2 <<
					point3Ptr[0] - point1Ptr[0],
					point3Ptr[1] - point1Ptr[1],
					point3Ptr[2] - point1Ptr[2];

				// The plane normal is calculated as the cross-product of the tangent directions
				Eigen::Vector3d normal =
					tangent1.cross(tangent2);
				// Normalizing the plane normal is important to keep the residual calculation as fast as possible
				normal.normalize();

				// The offset from the origin
				double d = -normal(0) * point1Ptr[0] - normal(1) * point1Ptr[1] - normal(2) * point1Ptr[2];
				// Adding the estimated line to the vector models
				models_.resize(models_.size() + 1);
				models_.back().descriptor.resize(4, 1);
				models_.back().descriptor << normal, d;
				return true;
			}

			template <size_t _DimensionNumber>
			OLGA_INLINE bool LinearModelSolver<_DimensionNumber>::estimate2DLine(
				const cv::Mat& data_,
				const size_t* sample_,
				size_t sampleNumber_,
				std::vector<Model>& models_,
				const double* weights_) const
			{
				const double* dataPtr = reinterpret_cast<double*>(data_.data);
				const int kColumns = data_.cols;

				// The pointer to the first point
				const double* point1Ptr = dataPtr + kColumns * sample_[0];
				// The pointer to the second point
				const double* point2Ptr = dataPtr + kColumns * sample_[1];

				// The coordinates of the points
				const double& x1 = point1Ptr[0],
					& y1 = point1Ptr[1],
					& x2 = point2Ptr[0],
					& y2 = point2Ptr[1];

				// The line's normal
				double nx = y1 - x2,
					ny = x2 - x1;
				// Normalizing the line number is important to keep the residual calculation as fast as possible
				double magnitude = std::sqrt(nx * nx + ny * ny);
				nx /= magnitude;
				ny /= magnitude;
				// The offset from the origin
				double c = -nx * x1 - ny * y1;
				// Adding the estimated line to the vector models
				models_.resize(models_.size() + 1);
				models_.back().descriptor.resize(3, 1);
				models_.back().descriptor << nx, ny, c;
				return true;
			}

			template <size_t _DimensionNumber>
			OLGA_INLINE bool LinearModelSolver<_DimensionNumber>::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sampleNumber_,
				std::vector<Model> &models_,
				const double *weights_) const
			{				
				// Line and plane fitting is particularly easy from a minimal sample, thus
				// we address them accordingly.
				if constexpr (_DimensionNumber == 2) 
					// 2D line fitting from a minimal sample, i.e., 2 points
					if (sampleNumber_ == sampleSize())
						return estimate2DLine(
							data_,
							sample_,
							sampleNumber_,
							models_,
							weights_);

				if constexpr (_DimensionNumber == 3)
					// 3D plane fitting from a minimal sample, i.e., 3 points
					if (sampleNumber_ == sampleSize())
						return estimate3DPlane(data_,
							sample_,
							sampleNumber_,
							models_,
							weights_);

				const double* dataPtr = reinterpret_cast<double*>(data_.data);
				const int kColumns = data_.cols;

				// The coefficient matrix containing the point coordinates
				Eigen::MatrixXd coefficients(sampleNumber_, _DimensionNumber);

				// We assume that the points are normalized and, therefore, the mass point is in the origin
				for (size_t pointIdx = 0; pointIdx < sampleNumber_; pointIdx++)
				{
					const int sampleIdx = sample_ == nullptr ?
						pointIdx : 
						sample_[pointIdx];
					const double* pointPtr = dataPtr + kColumns * sampleIdx;

					for (size_t column = 0; column < _DimensionNumber; ++column)
						coefficients(pointIdx, column) = *(pointPtr++);
				}

				Eigen::Matrix<double, _DimensionNumber, 1> hyperPlane;
				// If we are given a minimal sample, the problem is not overdetermined and, thus, can be solved by
				// LU decomposition which is significantly faster than QR or SVD
				if (sampleNumber_ == sampleSize())
				{
					const Eigen::FullPivLU<Eigen::MatrixXd> lu(coefficients.transpose() * coefficients);
					if (lu.dimensionOfKernel() != 1)
						return false;
					hyperPlane = lu.kernel();
				}
				// If the problem overdetermined, we are applying QR decomposition to find the null-space.
				else 
				{
					const Eigen::FullPivHouseholderQR<Eigen::MatrixXd> qr(
						coefficients.transpose() * coefficients);
					const Eigen::MatrixXd& Q = qr.matrixQ();
					hyperPlane = Q.rightCols<1>();
				}

				// Normalizing the normal to have unit length
				hyperPlane.normalize();

				models_.resize(models_.size() + 1);
				models_.back().descriptor.resize(_DimensionNumber + 1, 1);
				models_.back().descriptor << hyperPlane, 0;
				return true;
			}
		}
	}
}