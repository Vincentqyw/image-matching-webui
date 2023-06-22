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
#include <opencv2/core/core.hpp>
#include "uniform_random_generator.h"
#include "sampler.h"

namespace gcransac
{
	namespace sampler
	{
		// This sampler deterministically selects points one after another. 
		// It acts like a quasi-random sampler: it partitions the point number into segments
		// and selects the indices from the consecutive segments.
		class SinglePointSampler : public Sampler < cv::Mat, size_t >
		{
		protected:
			const size_t step_size,
				point_number;
			size_t current_offset,
				current_index;

			// A flag to see if the sampler used all possible samples
			bool finished;

		public:
			explicit SinglePointSampler(const cv::Mat * const container_,
				const size_t step_size_ = 10)
				: Sampler(container_),
				current_index(0),
				current_offset(0),
				step_size(step_size_),
				finished(false),
				point_number(container_->rows)
			{
				initialized = initialize(container_);
			}

			~SinglePointSampler() {}

			const std::string getName() const { return "Single Point Sampler"; }

			// Initializes any non-trivial variables and sets up sampler if
			// necessary. Must be called before sample is called.
			bool initialize(const cv::Mat * const container_)
			{
				return true;
			}

			void reset()
			{
				current_offset = 0;
				current_index = 0;
				finished = false;
			}

			// Samples the input variable data and fills the std::vector subset with the
			// samples.
			OLGA_INLINE bool sample(const std::vector<size_t> &pool_,
				size_t * const subset_,
				size_t sample_size_);
		};

		OLGA_INLINE bool SinglePointSampler::sample(
			const std::vector<size_t> &pool_,
			size_t * const subset_,
			size_t sample_size_)
		{
			if (finished)
				return false;

			// If there are not enough points in the pool, interrupt the procedure.
			if (sample_size_ > pool_.size())
				return false;

			if (sample_size_ < 1)
			{
				fprintf(stderr, "An error has occured. The single point sampler can only sample, wait..., a single point.\n");
				return false;
			}

			if (pool_.size() != point_number)
			{
				fprintf(stderr, "An error has occured in the single point sampler. The pool should be of the same size as the initial point number.\n");
				return false;
			}

			// Select the point to which the current index points
			subset_[0] = pool_[current_index];
			// Increase the index by step size
			current_index += step_size;
			// If the current index exceeds the point number
			if (current_index >= point_number)
			{
				// Increase the offset to don't start from the previously first point
				++current_offset;
				// Initialize the current index to the beginning and start again the increment
				current_index = current_offset;

				if (current_offset == step_size)
				{
					finished = true;
					return false;
				}
			}
			return true;
		}
	}
}