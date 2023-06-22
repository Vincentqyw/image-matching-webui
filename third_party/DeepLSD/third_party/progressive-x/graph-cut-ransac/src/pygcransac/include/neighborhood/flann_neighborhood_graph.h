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

#include "neighborhood_graph.h"
#include <vector>
#include <opencv2/core/core.hpp>
#include <Eigen/Eigen>
#include "../estimators/estimator.h"

namespace gcransac
{
	namespace neighborhood 
	{
		class FlannNeighborhoodGraph : public NeighborhoodGraph<cv::Mat>
		{
		public:
			// The possible methods for building the neighborhood graph.
			// "SearchType::RadiusSearch" uses a hypersphere around each point with a manually set radius.
			// "SearchType::KNN" applies the k-nearest-neighbors algorithm with a manually set k value.
			enum SearchType { RadiusSearch, KNN };

		protected:
			// The radius used as the radius of the neighborhood ball.
			double matching_radius;

			// The k value used in the k-nearest-neighbors algorithm.
			size_t knn;

			// The container consisting of the found neighbors for each point.
			std::vector<std::vector<size_t>> neighbours;

			// The method used for building the neighborhood graph.
			SearchType search_type;

		public:
			FlannNeighborhoodGraph() : NeighborhoodGraph() {}

			FlannNeighborhoodGraph(const cv::Mat * const container_, // The pointer of the container consisting of the data points.
				double matching_radius_, // The radius used as the radius of the neighborhood ball.
				size_t knn_ = 0, // The k value used in the k-nearest-neighbors algorithm.
				SearchType search_type_ = SearchType::RadiusSearch) : // The method used for building the neighborhood graph.
				matching_radius(matching_radius_),
				search_type(search_type_),
				knn(knn_),
				NeighborhoodGraph(container_)
			{
				initialized = initialize(container);
			}

			bool initialize(const cv::Mat * const container_);
			inline const std::vector<size_t> &getNeighbors(size_t point_idx_) const;

			// A function returning the cell sizes
			const std::vector<double> &getCellSizes() const { return std::vector<double>(); }

			// A function returning all cells in the graph
			const std::unordered_map<size_t, std::pair<std::vector<size_t>, std::vector<size_t>>>& getCells() const {
				return std::unordered_map<size_t, std::pair<std::vector<size_t>, std::vector<size_t>>>();
			}

			// The number of divisions/cells along an axis
			size_t getDivisionNumber() const { return 0; }

			// A function returning the number of cells filled
			size_t filledCellNumber() const { return neighbours.size(); }
		};

		bool FlannNeighborhoodGraph::initialize(const cv::Mat * const container_)
		{
			// Compute the neighborhood graph
			// TODO: replace by nanoflann
			std::vector<std::vector<cv::DMatch>> tmp_neighbours;
			cv::FlannBasedMatcher flann(new cv::flann::KDTreeIndexParams(4), new cv::flann::SearchParams(6));

			if (container_->type() == CV_32F)
			{
				flann.radiusMatch(*container_, // The point set 
					*container_, // The point set 
					tmp_neighbours, // The estimated neighborhood graph
					static_cast<float>(matching_radius)); // The radius of the neighborhood ball
			}
			else
			{
				cv::Mat tmp_points;
				container_->convertTo(tmp_points, CV_32F); // OpenCV's FLANN dies if the points are doubles
				flann.radiusMatch(tmp_points, // The point set converted to floats
					tmp_points, // The point set converted to floats
					tmp_neighbours, // The estimated neighborhood graph
					static_cast<float>(matching_radius)); // The radius of the neighborhood ball
			}

			// Count the edges in the neighborhood graph
			neighbours.resize(tmp_neighbours.size());
			for (size_t i = 0; i < tmp_neighbours.size(); ++i)
			{
				if (tmp_neighbours[i].size() == 0)
					continue;
				const size_t n = tmp_neighbours[i].size() - 1;
				neighbor_number += static_cast<int>(n);

				neighbours[i].resize(n);
				for (size_t j = 0; j < n; ++j)
					neighbours[i][j] = tmp_neighbours[i][j + 1].trainIdx;
			}

			return neighbor_number > 0;
		}

		inline const std::vector<size_t> &FlannNeighborhoodGraph::getNeighbors(size_t point_idx_) const
		{
			return neighbours[point_idx_];
		}
	}
}