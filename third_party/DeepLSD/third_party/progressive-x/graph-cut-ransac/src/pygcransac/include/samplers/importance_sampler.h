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

#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>

#include "sampler.h"
#include "uniform_random_generator.h"

namespace gcransac
{
	namespace sampler
	{
		// Prosac sampler used for PROSAC implemented according to "cv::Matching with PROSAC
		// - Progressive Sampling Consensus" by Chum and cv::Matas.
		class ImportanceSampler : public Sampler < cv::Mat, size_t >
		{
		protected:
			size_t sample_size,
				point_number,
				ransac_convergence_iterations, // Number of iterations of PROSAC before it just acts like ransac.
				kth_sample_number, // The kth sample of prosac sampling.
				largest_sample_size,
				subset_size;

			std::vector<size_t> growth_function;

			std::unique_ptr<utils::UniformRandomGenerator<size_t>> random_generator; // The random number generator
			std::discrete_distribution<int> multinomial_distribution;

		public:
			explicit ImportanceSampler(const cv::Mat * const container_,
				const std::vector<double> &probabilities_,
				const size_t sample_size_) :
				sample_size(sample_size_),
				point_number(container_->rows),
				Sampler(container_)
			{
				if (probabilities_.size() != container_->rows)
					return;

				// Initialize the distribution from the point probabilities
				multinomial_distribution = std::discrete_distribution<int>(
					std::begin(probabilities_), 
					std::end(probabilities_));

				initialized = initialize(container);
			}

			~ImportanceSampler() 
			{
				utils::UniformRandomGenerator<size_t> *generator_ptr = random_generator.release();
				delete generator_ptr;
			}

			const std::string getName() const { return "Importance Sampler"; }

			void reset()
			{
				subset_size = sample_size; // The size of the current sampling pool		
				random_generator->resetGenerator(0,
					subset_size - 1);
			}

			bool initialize(const cv::Mat * const container_)
			{
				// Initialize the random generator
				random_generator = std::make_unique<utils::UniformRandomGenerator<size_t>>();
				random_generator->resetGenerator(0,
					subset_size - 1);
				return true;
			}

			// Samples the input variable data and fills the std::vector subset with the prosac
			// samples.
			// NOTE: This assumes that data is in sorted order by quality where data[i] is
			// of higher quality than data[j] for all i < j.
			OLGA_INLINE bool sample(const std::vector<size_t> &pool_,
				size_t * const subset_,
				size_t sample_size_)
			{
				if (sample_size_ != sample_size)
				{
					fprintf(stderr, "An error occured when sampling.\n");
					return false;
				}

				for (size_t sample_idx = 0; sample_idx < sample_size_; ++sample_idx)
					subset_[sample_idx] = multinomial_distribution(random_generator->getGenerator());
				return true;
			}
		};
	}
}
