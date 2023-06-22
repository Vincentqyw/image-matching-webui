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
		template<typename _NeighborhoodGraph>
		class NapsacSampler : public Sampler < cv::Mat, size_t >
		{
		protected:
			std::unique_ptr<utils::UniformRandomGenerator<size_t>> random_generator;
			const _NeighborhoodGraph *neighborhood;
			const size_t maximum_iterations;

		public:
			explicit NapsacSampler(const cv::Mat * const container_,
				const _NeighborhoodGraph *neighborhood_)
				: Sampler(container_),
				neighborhood(neighborhood_),
				maximum_iterations(100)
			{
				initialized = initialize(container_);
			}

			~NapsacSampler() 
			{
				utils::UniformRandomGenerator<size_t> *generator_ptr = random_generator.release();
				delete generator_ptr;
			}

			const std::string getName() const { return "NAPSAC Sampler"; }

			// Initializes any non-trivial variables and sets up sampler if
			// necessary. Must be called before sample is called.
			bool initialize(const cv::Mat * const container_)
			{
				random_generator = std::make_unique<utils::UniformRandomGenerator<size_t>>();
				random_generator->resetGenerator(0,
					static_cast<size_t>(container_->rows));
				return true;
			}

			void reset()
			{
				random_generator->resetGenerator(0,
					static_cast<size_t>(container->rows));
			}

			// Samples the input variable data and fills the std::vector subset with the
			// samples.
			OLGA_INLINE bool sample(const std::vector<size_t> &pool_,
				size_t * const subset_,
				size_t sample_size_);
		};

		template<typename _NeighborhoodGraph>
		OLGA_INLINE bool NapsacSampler<_NeighborhoodGraph>::sample(
			const std::vector<size_t> &pool_,
			size_t * const subset_,
			size_t sample_size_)
		{
			// If there are not enough points in the pool, interrupt the procedure.
			if (sample_size_ > pool_.size())
				return false;

			size_t attempts = 0;
			while (attempts++ < maximum_iterations)
			{
				// Select a point randomly
				random_generator->generateUniqueRandomSet(subset_, // The sample to be selected
					1, // Only a single point is selected to be the center
					pool_.size() - 1); // The index upper bound

				// The indices of the points which are in the same cell as the
				// initially selected one.
				const std::vector<size_t> &neighbors =
					neighborhood->getNeighbors(subset_[0]);

				// Try again with another first point since the current one does not have enough neighbors
				if (neighbors.size() < sample_size_)
					continue;

				// If the selected point has just enough neighbors use them all.
				if (neighbors.size() == sample_size_)
				{
					for (size_t i = 0; i < sample_size_; ++i)
						subset_[i] = neighbors[i];
					break;	
				}

				// If the selected point has more neighbors than required select randomly.
				random_generator->generateUniqueRandomSet(subset_ + 1, // The sample to be selected
					sample_size_ - 1, // Only a single point is selected to be the center
					neighbors.size() - 1, // The index upper bound
					subset_[0]); // Index to skip

				// Replace the indices reffering to neighbor to the ones that refer to points
				for (size_t i = 1; i < sample_size_; ++i)
					subset_[i] = neighbors[subset_[i]];
				break;
			}
			// Return true only if the iteration was interrupted due to finding a good sample
			return attempts < maximum_iterations;
		}
	}
}