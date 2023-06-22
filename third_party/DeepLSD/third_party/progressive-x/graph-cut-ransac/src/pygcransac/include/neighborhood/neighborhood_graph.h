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
#include <Eigen/Eigen>
#include "../estimators/estimator.h"

namespace gcransac
{
	namespace neighborhood
	{
		template <typename _DataContainer>
		class NeighborhoodGraph
		{
		protected:
			// The pointer of the container consisting of the data points from which
			// the neighborhood graph is constructed.
			const _DataContainer  * const container;

			// The number of neighbors, i.e., edges in the neighborhood graph.
			size_t neighbor_number;

			// A flag indicating if the initialization was successfull.
			bool initialized;

		public:
			NeighborhoodGraph() : initialized(false), 
				container(nullptr) 
			{
			}

			NeighborhoodGraph(const _DataContainer * const container_) :
				neighbor_number(0),
				container(container_)
			{
			}

			virtual ~NeighborhoodGraph()
			{

			}

			// A function to initialize and create the neighbordhood graph.
			virtual bool initialize(const _DataContainer * const container_) = 0;

			// Returns the neighbors of the current point in the graph.
			inline virtual const std::vector<size_t> &getNeighbors(size_t point_idx_) const = 0;

			// Returns the number of edges in the neighborhood graph.
			size_t getNeighborNumber() const { return neighbor_number; }

			// Returns if the initialization was successfull.
			bool isInitialized() const { return initialized; }

			// A function returning the cell sizes
			virtual const std::vector<double> &getCellSizes() const = 0;

			// A function returning all cells in the graph
			virtual const std::unordered_map<size_t, std::pair<std::vector<size_t>, std::vector<size_t>>>& getCells() const = 0;

			// The number of divisions/cells along an axis
			virtual size_t getDivisionNumber() const = 0;

			// A function returning the number of cells filled
			virtual size_t filledCellNumber() const = 0;
		};
	}
}