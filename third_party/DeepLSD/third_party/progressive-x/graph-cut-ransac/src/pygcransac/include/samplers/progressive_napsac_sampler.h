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
#include "neighborhood/grid_neighborhood_graph.h"
#include "prosac_sampler.h"
#include "sampler.h"

namespace gcransac
{
	namespace sampler
	{
		// The paper in which the Progressive NAPSAC sampler is described is
		// at https://arxiv.org/abs/1906.02295
		template <size_t _DimensionNumber>
		class ProgressiveNapsacSampler : public Sampler < cv::Mat, size_t >
		{
		protected:
			std::unique_ptr<utils::UniformRandomGenerator<size_t>> random_generator; // The random number generator
			const size_t layer_number; // The number of overlapping neighborhood grids
			const double sampler_length; // The length of fully blending to global sampling 
			const std::vector<double> sizes; // The sizes along each axis

			std::vector<neighborhood::GridNeighborhoodGraph<_DimensionNumber>> grid_layers; // The overlapping neighborhood grids
			std::vector<size_t> layer_data; // The sizes of the grids

			std::vector<size_t> current_layer_per_point, // It stores the index of the layer which is used for each point
				hits_per_point, // It stores how many times each point has been selected
				subset_size_per_point, // It stores what is the size of subset (~the size of the neighborhood ball) for each point
				growth_function_progressive_napsac; // The P-NAPSAC growth function.

			ProsacSampler one_point_prosac_sampler, // The PROSAC sampler used for selecting the initial points, i.e., the center of the hypersphere.
				prosac_sampler; // The PROSAC sampler used when the sampling has been fully blended to be global sampling.

			size_t kth_sample_number, // The kth sample of prosac sampling.
				max_progressive_napsac_iterations, // The maximum number of local sampling iterations before applying global sampling
				sample_size, // The size of a minimal sample to fit a model.
				point_number; // The number of data points.

		public:
			explicit ProgressiveNapsacSampler(
				const cv::Mat *container_, // The pointer pointing to the data points
				const std::vector<size_t> layer_data_, // The number of cells for each neighborhood grid. This must be in descending order.
				const size_t sample_size_, // The size of a minimal sample.
				const std::vector<double> &sizes_, // The width of the source image
				const double sampler_length_ = 20) // The length of fully blending to global sampling 
				: Sampler(container_),
				layer_data(layer_data_),
				sample_size(sample_size_),
				layer_number(layer_data_.size()),
				point_number(container_->rows),
				current_layer_per_point(container_->rows, 0),
				hits_per_point(container_->rows, 0),
				subset_size_per_point(container_->rows, sample_size_),
				sizes(sizes_),
				kth_sample_number(0),
				one_point_prosac_sampler(container_, 1, container_->rows),
				prosac_sampler(container_, sample_size_, container_->rows),
				sampler_length(sampler_length_)
			{
				initialized = initialize(container_);
			}

			~ProgressiveNapsacSampler() 
			{
				utils::UniformRandomGenerator<size_t> *generator_ptr = random_generator.release();
				delete generator_ptr;
			}

			// Initializes any non-trivial variables and sets up sampler if
			// necessary. Must be called before sample is called.
			bool initialize(const cv::Mat *container_);

			void reset();

			const std::string getName() const { return "Progressive NAPSAC Sampler"; }

			// Samples the input variable data and fills the std::vector subset with the
			// samples.
			OLGA_INLINE bool sample(const std::vector<size_t> &pool_,
				size_t * const subset_,
				size_t sample_size_);
		};

		template <size_t _DimensionNumber>
		void ProgressiveNapsacSampler<_DimensionNumber>::reset()
		{
			random_generator->resetGenerator(0,
				static_cast<size_t>(point_number));
			kth_sample_number = 0;
			hits_per_point = std::vector<size_t>(point_number, 0);
			current_layer_per_point = std::vector<size_t>(point_number, 0);
			subset_size_per_point = std::vector<size_t>(point_number, sample_size);
			one_point_prosac_sampler.reset();
			prosac_sampler.reset();
		}

		template <size_t _DimensionNumber>
		bool ProgressiveNapsacSampler<_DimensionNumber>::initialize(const cv::Mat *container_)
		{
			// Initialize the random generator
			random_generator = std::make_unique<utils::UniformRandomGenerator<size_t>>();
			random_generator->resetGenerator(0,
				static_cast<size_t>(point_number));

			max_progressive_napsac_iterations =
				static_cast<size_t>(sampler_length * container_->rows);

			// Initialize the grid layers. We do not need to add the last layer
			// since it contains the whole image.
			grid_layers.reserve(layer_number);

			// Check layer data
			for (size_t layer_idx = 0; layer_idx < layer_number; ++layer_idx)
			{
				if (layer_idx > 0 &&
					layer_data[layer_idx - 1] <= layer_data[layer_idx])
				{
					fprintf(stderr, "Error when initializing the Progressive NAPSAC sampler. The layers must be in descending order. The current order is \"");
					for (size_t layer_idx = 0; layer_idx < layer_number - 1; ++layer_idx)
						fprintf(stderr, "%d ", layer_data[layer_idx]);
					fprintf(stderr, "%d\"\n", layer_data.back());
					return false;
				}

				const size_t cell_number_in_grid = layer_data[layer_idx];

				std::vector<double> cell_sizes(_DimensionNumber);
				for (size_t dimensionIdx = 0; dimensionIdx < _DimensionNumber; ++dimensionIdx)
					cell_sizes[dimensionIdx] = sizes[dimensionIdx] / cell_number_in_grid;
 
				grid_layers.emplace_back(neighborhood::GridNeighborhoodGraph<_DimensionNumber>(container_,
					cell_sizes,
					cell_number_in_grid));
			}

			// Inititalize the P-NAPSAC growth function
			growth_function_progressive_napsac.resize(point_number);
			size_t local_sample_size = sample_size - 1;
			double T_n = max_progressive_napsac_iterations;
			for (size_t i = 0; i < local_sample_size; ++i) {
				T_n *= static_cast<double>(local_sample_size - i) /
					(point_number - i);
			}

			unsigned int T_n_prime = 1;
			for (size_t i = 0; i < point_number; ++i) {
				if (i + 1 <= local_sample_size) {
					growth_function_progressive_napsac[i] = T_n_prime;
					continue;
				}
				double Tn_plus1 = static_cast<double>(i + 1) * T_n /
					(i + 1 - local_sample_size);
				growth_function_progressive_napsac[i] = T_n_prime + static_cast<size_t>(ceil(Tn_plus1 - T_n));
				T_n = Tn_plus1;
				T_n_prime = growth_function_progressive_napsac[i];
			}

			return true;
		}

		template <size_t _DimensionNumber>
		OLGA_INLINE bool ProgressiveNapsacSampler<_DimensionNumber>::sample(
			const std::vector<size_t> &pool_,
			size_t * const subset_,
			size_t sample_size_)
		{
			++kth_sample_number;

			if (sample_size_ != sample_size)
			{
				fprintf(stderr, "An error occured when sampling. Progressive NAPSAC is not yet implemented to change the sample size after being initialized.\n");
				return false;
			}

			// If there are not enough points in the pool, interrupt the procedure.
			if (sample_size_ > pool_.size())
				return false;

			// Do completely global sampling (PROSAC is used now), instead of Progressive NAPSAC,
			// if the maximum iterations has been done without finding the sought model.
			if (kth_sample_number > max_progressive_napsac_iterations)
			{
				prosac_sampler.setSampleNumber(kth_sample_number);
				return prosac_sampler.sample(pool_,
					subset_,
					sample_size_);
			}

			// Select the first point used as the center of the
			// hypersphere in the local sampling.
			const bool success = one_point_prosac_sampler.sample(pool_, // The pool from which the indices are chosen
				subset_, // The sample to be selected
				1); // Only a single point is selected to be the center
			if (!success) // Return false, if the selection of the initial point was not successfull.
				return false;

			// The index of the selected center
			const size_t initial_point = subset_[0];

			// Increase the number of hits of the selected point
			size_t &hits = ++hits_per_point[initial_point];

			// Get the subset size (i.e., the size of the neighborhood sphere) of the 
			// selected initial point.
			size_t &subset_size_progressive_napsac = subset_size_per_point[initial_point];
			while (hits > growth_function_progressive_napsac[subset_size_progressive_napsac - 1] &&
				subset_size_progressive_napsac < point_number)
				subset_size_progressive_napsac = MIN(subset_size_progressive_napsac + 1, point_number);

			// Get the neighborhood from the grids
			size_t &current_layer = current_layer_per_point[initial_point];
			bool is_last_layer = false;
			do // Try to find the grid which contains enough points
			{
				// In the case when the grid with a single cell is used,
				// apply PROSAC.
				if (current_layer >= layer_number)
				{
					is_last_layer = true;
					break;
				}

				// Get the points in the cell where the selected initial
				// points is located.
				const std::vector<size_t> &neighbors =
					grid_layers[current_layer].getNeighbors(initial_point);

				// If there are not enough points in the cell, start using a 
				// less fine grid.
				if (neighbors.size() < subset_size_progressive_napsac)
				{
					++current_layer; // Jump to the next layer with bigger cells.
					continue;
				}
				// If the procedure got to this point, there is no reason to choose a different layer of grids
				// since the current one has enough points. 
				break;
			} while (1);

			// If not the last layer has been chosen, sample from the neighbors of the initially selected point.
			if (!is_last_layer)
			{
				// The indices of the points which are in the same cell as the
				// initially selected one.
				const std::vector<size_t> &neighbors =
					grid_layers[current_layer].getNeighbors(initial_point);

				// Put the selected point to the end of the sample array to avoid
				// being overwritten when sampling the remaining points.
				subset_[sample_size - 1] = initial_point;

				// The next point should be the farthest one from the initial point. Note that the points in the grid cell are
				// not ordered w.r.t. to their distances from the initial point. However, they are ordered as in PROSAC.
				subset_[sample_size - 2] = neighbors[subset_size_progressive_napsac - 1];

				// Select n - 2 points randomly  
				random_generator->generateUniqueRandomSet(subset_,
					sample_size_ - 2,
					subset_size_progressive_napsac - 2,
					initial_point);

				for (size_t i = 0; i < sample_size_ - 2; ++i)
				{
					subset_[i] = neighbors[subset_[i]];  // Replace the neighbor index by the index of the point
					++hits_per_point[subset_[i]]; // Increase the hit number of each selected point
				}
				++hits_per_point[subset_[sample_size_ - 2]]; // Increase the hit number of each selected point
			}
			// If the last layer (i.e., the layer with a single cell) has been chosen, do global sampling
			// by PROSAC sampler. 
			else
			{
				// If local sampling
				prosac_sampler.setSampleNumber(kth_sample_number);
				const bool success = prosac_sampler.sample(pool_,
					subset_,
					sample_size_);
				subset_[sample_size - 1] = initial_point;
				return success;
			}
			return true;
		}
	}
}
