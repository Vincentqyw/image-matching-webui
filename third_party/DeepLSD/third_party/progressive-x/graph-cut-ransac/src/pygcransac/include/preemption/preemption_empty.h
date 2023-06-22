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

#include "model.h"
#include <opencv2/core.hpp>
#include <Eigen/Eigen>

namespace gcransac
{
	namespace preemption
	{
		template <typename _ModelEstimator>
		class EmptyPreemptiveVerfication
		{
		public:
			static constexpr bool providesScore() { return false; }
			static constexpr const char *getName() { return "empty"; }

			bool verifyModel(
				const gcransac::Model &model_, // The current model
				const _ModelEstimator &estimator_, // The model estimator
				const double &threshold_, // The truncated threshold
				const size_t &iteration_number_, // The current iteration number
				const Score &best_score_, // The current best score
				const cv::Mat &points_, // The data points
				const size_t *minimal_sample_, // The current minimal sample
				const size_t sample_number_, // The number of samples used
				std::vector<size_t> &inliers_,// The current inlier set
				Score &score_, // The score of the model
				const std::vector<const std::vector<size_t>*> *index_sets_ = nullptr) // Sets of pre-selected point indices
			{
				return true;
			}
		};
	}
}