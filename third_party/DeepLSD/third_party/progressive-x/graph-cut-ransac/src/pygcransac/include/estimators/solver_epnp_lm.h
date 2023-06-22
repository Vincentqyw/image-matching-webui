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
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class EPnPLM : public SolverEngine
			{
			public:
				EPnPLM()
				{
				}

				~EPnPLM()
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
					return 3;
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


			// Estimate the model parameters from the given point sample
			// using weighted fitting if possible.
			OLGA_INLINE bool EPnPLM::estimateModel(
				const cv::Mat& data_, // The set of data points
				const size_t *sample_, // The sample used for the estimation
				size_t sample_number_, // The size of the sample
				std::vector<Model> &models_, // The estimated model parameters
				const double *weights_) const // The weight for each point
			{
				if (sample_ == nullptr)
					sample_number_ = data_.rows;

				if (sample_number_ < sampleSize())
					return false;

				if (sample_ == nullptr)
					sample_number_ = data_.rows;

				if (sample_number_ < sampleSize())
					return false;

				const double * data_ptr = reinterpret_cast<double *>(data_.data);
				const size_t columns = data_.cols;

				cv::Mat inlier_image_points(sample_number_, 2, CV_64F),
					inlier_object_points(sample_number_, 3, CV_64F);

				for (size_t i = 0; i < sample_number_; ++i)
				{
					const size_t idx = 
						sample_ == nullptr ? i : sample_[i];
					inlier_image_points.at<double>(i, 0) = data_.at<double>(idx, 0);
					inlier_image_points.at<double>(i, 1) = data_.at<double>(idx, 1);
					inlier_object_points.at<double>(i, 0) = data_.at<double>(idx, 2);
					inlier_object_points.at<double>(i, 1) = data_.at<double>(idx, 3);
					inlier_object_points.at<double>(i, 2) = data_.at<double>(idx, 4);
				}

				// Converting the estimated pose parameters OpenCV format
				Eigen::Matrix3d rotation;
				Eigen::Vector3d translation;

				cv::Mat cv_rotation(3, 3, CV_64F, rotation.data()), // The estimated rotation matrix converted to OpenCV format
					cv_translation(3, 1, CV_64F, translation.data()); // The estimated translation converted to OpenCV format

				// Convert the rotation matrix by the rodrigues formula
				cv::Mat cv_rodrigues(3, 1, CV_64F);
				cv::Rodrigues(cv_rotation.t(), cv_rodrigues);

				// Applying numerical optimization to the estimated pose parameters
				cv::solvePnP(inlier_object_points, // The object points
					inlier_image_points, // The image points
					cv::Mat::eye(3, 3, CV_64F), // The camera's intrinsic parameters 
					cv::Mat(), // An empty vector since the radial distortion is not known
					cv_rodrigues, // The initial rotation
					cv_translation, // The initial translation
					false, // Use the initial values
					cv::SOLVEPNP_ITERATIVE); // Apply numerical refinement
				
				// Convert the rotation vector back to a rotation matrix
				cv::Rodrigues(cv_rodrigues, cv_rotation);

				// Transpose the rotation matrix back
				cv_rotation = cv_rotation.t();

				Model model;
				model.descriptor = Eigen::Matrix<double, 3, 4>();
				model.descriptor.block<3, 3>(0, 0) = rotation;
				model.descriptor.block<3, 1>(0, 3) = translation;
				models_.emplace_back(model);

				return true;
			}
		
		}
	}
}