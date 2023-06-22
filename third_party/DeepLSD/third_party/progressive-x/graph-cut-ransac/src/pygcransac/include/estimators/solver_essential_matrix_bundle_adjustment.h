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
#include "../relative_pose/bundle.h"
#include "../relative_pose/essential.h"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating the essential matrix between two images from a larger-than-minimal
			// sample using bundle adjustment. 
			class EssentialMatrixBundleAdjustmentSolver : public SolverEngine
			{
			protected:
				// The options for the bundle adjustment
				pose_lib::BundleOptions bundle_options;

			public:
				EssentialMatrixBundleAdjustmentSolver(
					const pose_lib::BundleOptions::LossType &loss_type_ = pose_lib::BundleOptions::LossType::TRUNCATED,
					const size_t &maximum_iterations_ = 25)
				{
					bundle_options.loss_type = loss_type_;
					bundle_options.max_iterations = maximum_iterations_;
				}

				~EssentialMatrixBundleAdjustmentSolver()
				{
				}

				pose_lib::BundleOptions& getMutableOptions()
				{
					return bundle_options;
				}

				const pose_lib::BundleOptions& getOptions() const
				{
					return bundle_options;
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
					return maximumSolutions() > 1;
				}

				// The maximum number of solutions returned by the solver
				static constexpr size_t maximumSolutions()
				{
					return 1;
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

			OLGA_INLINE bool EssentialMatrixBundleAdjustmentSolver::estimateModel(
				const cv::Mat& data_, // All point correspondences
				const size_t *sample_, // The sample, i.e., indices of points to be used
				size_t sample_number_, // The size of the sample
				std::vector<Model> &models_, // The estimated model parameters
				const double *weights_) const // The weights used for the estimation
			{
				// Check if we have enough points for the bundle adjustment
				if (sample_number_ < sampleSize())
					return false;

				// If no sample is provided use all points
				if (sample_ == nullptr)
					sample_number_ = data_.rows;

				// If there is no initial model provided estimate one
				std::vector<Model> temp_models;
				if (models_.size() == 0)
				{
					// If we are given enough points use the eight-point solver since that is fast and accurate enough for initializing the BA.
					if (sample_number_ >= 8)
					{
						estimator::solver::FundamentalMatrixEightPointSolver eight_point_solver;
						eight_point_solver.estimateModel(data_, // All point correspondences
							sample_, // The sample, i.e., indices of points to be used
							sample_number_, // The size of the sample
							temp_models, // The estimated model parameters
							weights_); // The weights used for the estimation
					}
					// Otherwise, use the five-point solver.
					else
					{
						estimator::solver::EssentialMatrixFivePointSteweniusSolver five_point_solver;
						five_point_solver.estimateModel(data_, // All point correspondences
							sample_, // The sample, i.e., indices of points to be used
							sample_number_, // The size of the sample
							temp_models, // The estimated model parameters
							weights_); // The weights used for the estimation
					}
				}
				else
					temp_models = models_;
				models_.clear();
				// Select the first point in the sample to be used for the cheirality check
				const size_t& point_idx = sample_[0];
				Eigen::Vector3d pt1, pt2;
				pt1 << data_.at<double>(point_idx, 0), data_.at<double>(point_idx, 1), 1;
				pt2 << data_.at<double>(point_idx, 2), data_.at<double>(point_idx, 3), 1;

				// Iterating through the possible models.
				// This is 1 if the eight-point solver is used.
				// Otherwise, it is up to 3. 
				for (auto& model : temp_models)
				{					
					// Decompose the essential matrix to camera poses
					pose_lib::CameraPoseVector poses;

					motion_from_essential(
						model.descriptor, // The essential matrix
						pt1, pt2, // The point correspondence used for the cheirality check
						&poses); // The decomposed poses

					// Iterating through the possible poses and optimizing each
					for (auto& pose : poses)
					{
						// Apply bundle adjustment
						pose_lib::refine_relpose(
							data_, // All point correspondences
							sample_, // The sample, i.e., indices of points to be used
							sample_number_, // The size of the sample
							&pose, // The optimized pose
							bundle_options, // The bundle adjustment options
							weights_); // The weights for the weighted LSQ fitting

						// Composing the essential matrix from the pose
						Eigen::Matrix3d E;
						pose_lib::essential_from_motion(pose, &E);

						// Adding the essential matrix as the estimated models.
						Model model;
						model.descriptor = E;
						models_.emplace_back(model);
					}
				}

				return models_.size() > 0;
			}

		}
	}
}