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
		class ProsacSampler : public Sampler < cv::Mat, size_t >
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

			inline void incrementIterationNumber()
			{
				// Increase the iteration number
				++kth_sample_number;

				// If the method should act exactly like RANSAC, set the random generator to
				// generate values from all possible indices.
				if (kth_sample_number > ransac_convergence_iterations)
					random_generator->resetGenerator(0,
						point_number - 1);
				else // Increment the size of the sampling pool if required			
					if (kth_sample_number > growth_function[subset_size - 1]) {
						++subset_size; // n = n + 1
						if (subset_size > point_number)
							subset_size = point_number;
						if (largest_sample_size < subset_size)
							largest_sample_size = subset_size;

						// Reset the random generator to generate values from the current subset of points,
						// except the last one since it will always be used. 
						random_generator->resetGenerator(0,
							subset_size - 2);
					}
			}

		public:
			explicit ProsacSampler(const cv::Mat * const container_,
				const size_t sample_size_,
				const size_t ransac_convergence_iterations_ = 100000) :
				sample_size(sample_size_),
				ransac_convergence_iterations(ransac_convergence_iterations_),
				point_number(container_->rows),
				kth_sample_number(1),
				Sampler(container_)
			{
				initialized = initialize(container);
			}

			~ProsacSampler() 
			{
				utils::UniformRandomGenerator<size_t> *generator_ptr = random_generator.release();
				delete generator_ptr;
			}

			const std::string getName() const { return "PROSAC Sampler"; }

			void reset()
			{
				kth_sample_number = 1;
				largest_sample_size = sample_size; // largest set sampled in PROSAC
				subset_size = sample_size; // The size of the current sampling pool		
				random_generator->resetGenerator(0,
					subset_size - 1);
			}

			bool initialize(const cv::Mat * const container_)
			{
				// Set T_n according to the PROSAC paper's recommendation.
				growth_function.resize(point_number, 0);

				// Tq.he data points in U_N are sorted in descending order w.r.t. the quality function 
				// Let {Mi}i = 1...T_N denote the sequence of samples Mi c U_N that are uniformly drawn by Ransac.

				// Let T_n be an average number of samples from {Mi}i=1...T_N that contain data points from U_n only.
				// compute initial value for T_n
				//                                  n - i
				// T_n = T_N * Product i = 0...m-1 -------, n >= sample size, N = points size
				//                                  N - i
				double T_n = ransac_convergence_iterations;
				for (size_t i = 0; i < sample_size; i++)
					T_n *= static_cast<double>(sample_size - i) / (point_number - i);

				size_t T_n_prime = 1;
				// compute values using recurrent relation
				//             n + 1
				// T(n+1) = --------- T(n), m is sample size.
				//           n + 1 - m

				// growth function is defined as
				// g(t) = min {n, T'_(n) >= t}
				// T'_(n+1) = T'_(n) + (T_(n+1) - T_(n))
				for (size_t i = 0; i < point_number; ++i) {
					if (i + 1 <= sample_size) {
						growth_function[i] = T_n_prime;
						continue;
					}
					double Tn_plus1 = static_cast<double>(i + 1) * T_n / (i + 1 - sample_size);
					growth_function[i] = T_n_prime + (unsigned int)ceil(Tn_plus1 - T_n);
					T_n = Tn_plus1;
					T_n_prime = growth_function[i];
				}

				largest_sample_size = sample_size; // largest set sampled in PROSAC
				subset_size = sample_size; // The size of the current sampling pool		

				// Initialize the random generator
				random_generator = std::make_unique<utils::UniformRandomGenerator<size_t>>();
				random_generator->resetGenerator(0,
					subset_size - 1);
				return true;
			}

			// Set the sample such that you are sampling the kth prosac sample (Eq. 6).
			void setSampleNumber(int k)
			{
				kth_sample_number = k;

				// If the method should act exactly like RANSAC, set the random generator to
				// generate values from all possible indices.
				if (kth_sample_number > ransac_convergence_iterations)
					random_generator->resetGenerator(0,
						point_number - 1);
				else // Increment the size of the sampling pool while required			
					while (kth_sample_number > growth_function[subset_size - 1] && 
						subset_size != point_number)
					{
						++subset_size; // n = n + 1
						if (subset_size > point_number)
							subset_size = point_number;
						if (largest_sample_size < subset_size)
							largest_sample_size = subset_size;

						// Reset the random generator to generate values from the current subset of points,
						// except the last one since it will always be used. 
						random_generator->resetGenerator(0,
							subset_size - 2);
					}
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
					fprintf(stderr, "An error occured when sampling. PROSAC is not yet implemented to change the sample size after being initialized.\n");
					incrementIterationNumber(); // Increase the iteration number
					return false;
				}

				// If the method should act exactly like RANSAC, sample from all points.
				// From this point, function 'incrementIterationNumber()' is not called
				// since it is not important to increase the iteration number.
				if (kth_sample_number > ransac_convergence_iterations) {
					random_generator->generateUniqueRandomSet(
						subset_, // The set of points' indices to be selected
						sample_size_); // The number of points to be selected
					return true;
				}

				// Generate PROSAC sample in range [0, subset_size-2]
				random_generator->generateUniqueRandomSet(
					subset_, // The set of points' indices to be selected
					sample_size - 1); // The number of points to be selected
				subset_[sample_size - 1] = subset_size - 1; // The last index is that of the point at the end of the current subset used.

				incrementIterationNumber(); // Increase the iteration number
				return true;
			}
		};
	}
}
