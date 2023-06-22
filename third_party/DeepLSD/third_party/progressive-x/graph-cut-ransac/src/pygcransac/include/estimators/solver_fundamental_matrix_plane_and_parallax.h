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

#include <iostream>

#include "solver_engine.h"
#include "fundamental_estimator.h"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class FundamentalMatrixPlaneParallaxSolver : public SolverEngine
			{
			protected:
				const Eigen::Matrix3d *homography;
				bool is_homography_set;

			public:
				FundamentalMatrixPlaneParallaxSolver(const Eigen::Matrix3d *homography_ = nullptr) :
					is_homography_set(false),
					homography(homography_)
				{
				}

				~FundamentalMatrixPlaneParallaxSolver()
				{
				}

				void setHomography(const Eigen::Matrix3d *homography_)
				{
					is_homography_set = true;
					homography = homography_;
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
					return 2;
				}

				OLGA_INLINE bool estimateModel(
					const cv::Mat& data_, // The set of data points
					const size_t *sample_, // The sample used for the estimation
					size_t sample_number_, // The size of the sample
					std::vector<Model> &models_, // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point
			};

			OLGA_INLINE bool FundamentalMatrixPlaneParallaxSolver::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				// Check if the required homography has been set
				if (!is_homography_set)
				{
					fprintf(stderr, "The homography has not been set when applying the plane-and-parallax fundamental matrix estimation.\n");
					return false;
				}

				Eigen::Vector3d source_point_1, // First point in the first image
					source_point_2, // The second point in the first image
					destination_point_1, // The first point in the second image
					destination_point_2; // The second point in the second image

				// The indices of the points
				const size_t &point_1_idx = sample_[0],
					&point_2_idx = sample_[1];

				// The pointers of the points
				const double * point_1_ptr = reinterpret_cast<double *>(data_.data) + point_1_idx * data_.cols;
				const double * point_2_ptr = reinterpret_cast<double *>(data_.data) + point_2_idx * data_.cols;

				source_point_1 << point_1_ptr[0], point_1_ptr[1], 1;
				destination_point_1 << point_1_ptr[2], point_1_ptr[3], 1;
				source_point_2 << point_2_ptr[0], point_2_ptr[1], 1;
				destination_point_2 << point_2_ptr[2], point_2_ptr[3], 1;

				// Projecting the points by the homography matrix
				const Eigen::Vector3d projected_point_1 = *homography * source_point_1,
					projected_point_2 = *homography * source_point_2;

				// Calculating the parameters of the lines between the projected and original points
				const Eigen::Vector3d line_1 = projected_point_1.cross(destination_point_1),
					line_2 = projected_point_2.cross(destination_point_2);

				// Estimating the epipole
				const Eigen::Vector3d epipole = line_1.cross(line_2);

				// There is no intersection
				if (std::abs(epipole(2)) < std::numeric_limits<double>::epsilon())
					return false;

				// Calculate the cross-product matrix of the epipole
				Eigen::Matrix3d epipolar_cross;
				epipolar_cross << 0, -epipole(2), epipole(1),
					epipole(2), 0, -epipole(0),
					-epipole(1), epipole(0), 0;

				// Calculate the fundamental matrix
				FundamentalMatrix model;
				model.descriptor = epipolar_cross * *homography;
				models_.push_back(model);
				return true;
			}
		}
	}
}