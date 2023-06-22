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

#include <vector>
#include <queue>
#include <opencv2/core/core.hpp>
#include "uniform_random_generator.h"
#include "sampler.h"

namespace gcransac
{
	namespace sampler
	{
		class AdaptiveReorderingSampler : public Sampler < cv::Mat, size_t >
		{
		protected:
			std::vector<std::tuple<double, size_t, size_t, double, double>> probabilities;
			double estimator_variance;
			double randomness,
				randomness_2,
				randomness_rand_max;
			
			std::priority_queue<std::pair<double, size_t>, 
				std::vector<std::pair<double, size_t>> > processing_queue;

		public:
			explicit AdaptiveReorderingSampler(const cv::Mat * const container_,
				const std::vector<double> &inlier_probabilities_,
				const size_t sample_size_,
				const double estimator_variance_ = 0.9765, //0.12,
				const double randomness_ = 0.01,
				const bool ordered_ = false)
				: Sampler(container_),
					randomness(randomness_),
					randomness_2(randomness_ / 2.0),
					randomness_rand_max(randomness_ / static_cast<double>(RAND_MAX)),
					estimator_variance(estimator_variance_)
			{
				if (inlier_probabilities_.size() != container_->rows)
				{
					fprintf(stderr, "The number of correspondences (%d) and the number of provided probabilities (%d) do not match.",
						container_->rows, 
						inlier_probabilities_.size());
					return;
				}

				// Saving the probabilities
				probabilities.reserve(inlier_probabilities_.size());
				for (size_t pointIdx = 0; pointIdx < inlier_probabilities_.size(); ++pointIdx)
				{
					double probability = inlier_probabilities_[pointIdx];
					if (probability == 1.0)
						probability -= 1e-6; 

					double a = probability * probability * (1.0 - probability) / estimator_variance - probability;
					double b = a * (1.0 - probability) / probability;
					
					probabilities.emplace_back(std::make_tuple(probability, pointIdx, 0, a, b));
					processing_queue.emplace(std::make_pair(probability, pointIdx));
				}

				// Initializing the base class
				initialized = initialize(container_);
			}

			~AdaptiveReorderingSampler() {}

			const std::string getName() const { return "Probabilistic Reordering Sampler"; }

			// Initializes any non-trivial variables and sets up sampler if
			// necessary. Must be called before sample is called.
			bool initialize(const cv::Mat * const container_)
			{
				return true;
			}

			void reset()
			{
			}

			// Samples the input variable data and fills the std::vector subset with the
			// samples.
			OLGA_INLINE bool sample(const std::vector<size_t> &pool_,
				size_t * const subset_,
				size_t sample_size_);

			OLGA_INLINE void update(
				const size_t* const subset_,
				const size_t &sample_size_,
				const size_t& iteration_number_,
				const double &inlier_ratio_);
		};

		OLGA_INLINE void AdaptiveReorderingSampler::update(
			const size_t* const subset_,
			const size_t& sample_size_,
			const size_t& iteration_number_,
			const double& inlier_ratio_)
		{
			for (size_t i = 0; i < sample_size_; ++i)
			{
				const size_t& sample_idx = subset_[i];
				size_t& appearance_number = std::get<2>(probabilities[sample_idx]);
				++appearance_number;

				const double &a = std::get<3>(probabilities[sample_idx]); 
				const double &b = std::get<4>(probabilities[sample_idx]); 

				double& updated_inlier_ratio = std::get<0>(probabilities[sample_idx]);

				updated_inlier_ratio = 
					abs(a / (a + b + appearance_number)) + 
					randomness_rand_max * static_cast<double>(rand()) - randomness_2;
					
				updated_inlier_ratio = 
					MAX(0.0, MIN(0.999, updated_inlier_ratio));

				processing_queue.emplace(std::make_pair(updated_inlier_ratio, sample_idx));
			}
		}

		OLGA_INLINE bool AdaptiveReorderingSampler::sample(
			const std::vector<size_t>& pool_,
			size_t* const subset_,
			size_t sample_size_)
		{
			for (size_t i = 0; i < sample_size_; ++i)
			{
				const auto& item = processing_queue.top();
				subset_[i] = item.second;
				processing_queue.pop();
			}
			return true;
		}
	}
}