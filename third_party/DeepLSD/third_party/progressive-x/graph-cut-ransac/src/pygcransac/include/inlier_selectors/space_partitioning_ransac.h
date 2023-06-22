// Copyright (C) 2021 ETH Zurich.
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
//     * Neither the name of 2021 ETH Zurich nor the
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
#include <opencv2/core.hpp>
#include "model.h"
#include "inlier_selector.h"
#include "neighborhood/grid_neighborhood_graph.h"

namespace gcransac
{
	namespace inlier_selector
	{
        template <
            typename _Estimator,
            typename _NeighborhoodStructure>
        class SpacePartitioningRANSAC : public AbstractInlierSelector<_Estimator, _NeighborhoodStructure>
        {
        protected:
            std::vector<bool> gridCornerMask;
            std::vector<std::tuple<int, int, double, double>> gridCornerCoordinatesH;
            std::vector<double> additionalParameters;

        public:
            static constexpr bool doesSomething() { return true; }

            explicit SpacePartitioningRANSAC(const _NeighborhoodStructure *kNeighborhoodGraph_) : 
                AbstractInlierSelector<_Estimator, _NeighborhoodStructure>(kNeighborhoodGraph_)
            {
                // The number cells filled in the grid
                const size_t &kCellNumber = kNeighborhoodGraph_->filledCellNumber();
                const size_t &kDivisionNumber = kNeighborhoodGraph_->getDivisionNumber();
                const size_t kMaximumCellNumber = std::pow(kDivisionNumber, 4);

                // Initialize the structures speeding up the selection by caching data
                gridCornerMask.resize(kMaximumCellNumber, false);
                gridCornerCoordinatesH.resize(kMaximumCellNumber);

                // Save additional info needed for the selection
                const auto &sizes = kNeighborhoodGraph_->getCellSizes();

                additionalParameters.resize(5);
                additionalParameters[0] = sizes[0]; // The width of the source image
                additionalParameters[1] = sizes[1]; // The height of the source image
                additionalParameters[2] = sizes[2]; // The width of the destination image
                additionalParameters[3] = sizes[3]; // The height of the destination image
                additionalParameters[4] = kDivisionNumber; // The number of cells along an axis   
            }

            // The function that runs the model-based inlier selector
            void run(
                const cv::Mat& kCorrespondences_, // All point correspondences
                const gcransac::Model& kModel_, // The model parameters
                const _NeighborhoodStructure& kNeighborhood_, // The neighborhood structure. This probably will be a GridNeighborhood currently.
                const double& inlierOutlierThreshold_,
                std::vector<const std::vector<size_t>*>& selectedCells_, // The indices of the points selected
                size_t& pointNumber_); 

        protected:
            void runHomography(
                const cv::Mat& kCorrespondences_,
                const gcransac::Model& kModel_,
                const _NeighborhoodStructure& kNeighborhood_,
                std::vector<const std::vector<size_t>*>& selectedCells_,
                size_t& pointNumber_,
                const double& inlierOutlierThreshold_);
        };

        // The function that runs the model-based inlier selector
        template <
            typename _Estimator,
            typename _NeighborhoodStructure>
        void SpacePartitioningRANSAC<_Estimator, _NeighborhoodStructure>::run(
            const cv::Mat& kCorrespondences_, // All point correspondences
            const gcransac::Model& kModel_, // The model parameters
            const _NeighborhoodStructure& kNeighborhood_, // The neighborhood structure. This probably will be a GridNeighborhood currently.
            const double& inlierOutlierThreshold_,
            std::vector<const std::vector<size_t>*>& selectedCells_, // The indices of the points selected
            size_t& pointNumber_)
        {
            // Initializing the selected point number to zero
            pointNumber_ = 0;

            if constexpr (std::is_same<_Estimator, gcransac::utils::DefaultHomographyEstimator>())
                runHomography(
                    kCorrespondences_,
                    kModel_,
                    kNeighborhood_,
                    selectedCells_,
                    pointNumber_,
                    inlierOutlierThreshold_);
            else  // Other then homography estimation is not implemented yet. 
            {
                // Return all points
                const auto& cellMap = kNeighborhood_.getCells();
                selectedCells_.reserve(cellMap.size());
                for (const auto& [cell, value] : cellMap)
                {
                    const auto &points = value.first;
                    selectedCells_.emplace_back(&points);
                    pointNumber_ += points.size();
                }
            }
        }
        
        template <typename _Estimator,
            typename _NeighborhoodStructure>
        OLGA_INLINE void SpacePartitioningRANSAC<_Estimator, _NeighborhoodStructure>::runHomography(
            const cv::Mat& kCorrespondences_,
            const gcransac::Model& kModel_,
            const _NeighborhoodStructure& kNeighborhood_,
            std::vector<const std::vector<size_t>*>& selectedCells_,
            size_t& pointNumber_,
            const double& inlierOutlierThreshold_)
        {
            /* 
                Selecting cells based on mutual visibility
            */
            constexpr double kDeterminantEpsilon = 1e-3;
            const Eigen::Matrix3d &descriptor = kModel_.descriptor;
            const double kDeterminant = descriptor.determinant();
            if (abs(kDeterminant) < kDeterminantEpsilon)
                return;

            const double& kCellSize1 = additionalParameters[0],
                & kCellSize2 = additionalParameters[1],
                & kCellSize3 = additionalParameters[2],
                & kCellSize4 = additionalParameters[3],
                & kPartitionNumber = additionalParameters[4];

            // Iterate through all cells and project their corners to the second image
            const static std::vector<int> steps = { 0, 0,
                0, 1,
                1, 0,
                1, 1 };

            std::fill(std::begin(gridCornerMask), std::end(gridCornerMask), 0);

            // Iterating through all cells in the neighborhood graph
            for (const auto& [cell, value] : kNeighborhood_.getCells())
            {
                // The points in the cell
                const auto &points = value.first;

                // Checking if there are enough points in the cell to make the cell selection worth it
                if (points.size() < 4)
                {
                    // If not, simply test all points from the cell and continue
                    selectedCells_.emplace_back(&points);
                    pointNumber_ += points.size();
                    continue;
                }

                const auto& kCornerIndices = value.second;
                bool overlaps = false;

                // Iterate through the corners of the current cell
                for (size_t stepIdx = 0; stepIdx < 8; stepIdx += 2)
                {
                    // The index of the currently projected corner
                    const size_t kCornerHorizontalIndex = kCornerIndices[0] + steps[stepIdx];
                    if (kCornerHorizontalIndex >= kPartitionNumber)
                        continue;

                    const size_t kCornerVerticalIndex = kCornerIndices[1] + steps[stepIdx + 1];
                    if (kCornerVerticalIndex >= kPartitionNumber)
                        continue;

                    // Get the index of the corner's projection in the destination image
                    const size_t kIdx2d = kCornerHorizontalIndex * kPartitionNumber + kCornerVerticalIndex;

                    // This is already or will be the horizontal and vertical indices in the destination image 
                    auto& indexPair = gridCornerCoordinatesH[kIdx2d];

                    // If the corner hasn't yet been projected to the destination image
                    if (!gridCornerMask[kIdx2d])
                    {
                        // Get the coordinates of the corner
                        const double kX1 = kCornerHorizontalIndex * kCellSize1,
                            kY1 = kCornerVerticalIndex * kCellSize2;

                        // Project them by the estimated homography matrix
                        double x2p = kX1 * descriptor(0, 0) + kY1 * descriptor(0, 1) + descriptor(0, 2),
                            y2p = kX1 * descriptor(1, 0) + kY1 * descriptor(1, 1) + descriptor(1, 2),
                            h2p = kX1 * descriptor(2, 0) + kY1 * descriptor(2, 1) + descriptor(2, 2);

                        x2p /= h2p;
                        y2p /= h2p;

                        // Store the projected corner's cell indices
                        std::get<0>(indexPair) = x2p / kCellSize3;
                        std::get<1>(indexPair) = y2p / kCellSize4;
                        std::get<2>(indexPair) = x2p;
                        std::get<3>(indexPair) = y2p;
                        
                        // Note that the corner has been already projected
                        gridCornerMask[kIdx2d] = true;
                    }

                    // Check if the projected corner is equal to the correspondence's destination point's grid cell.
                    // This works due to the coordinate truncation.
                    if (std::get<0>(indexPair) == kCornerIndices[2] &&
                        std::get<1>(indexPair) == kCornerIndices[3])
                    {
                        // Store the points in the cell to be tested
                        selectedCells_.emplace_back(&points);
                        pointNumber_ += points.size();
                        overlaps = true;
                        break;
                    }
                }

                // Check if there is an overlap
                if (!overlaps)
                {
                    // The horizontal index of the bottom-right corner
                    const size_t kCornerHorizontalIndex11 = kCornerIndices[0] + steps[6];
                    // The vertical index of the bottom-right corner
                    const size_t kCornerVerticalIndex11 = kCornerIndices[1] + steps[7];

                    // Calculating the index of the top-left corner
                    const size_t kIdx2d00 = kCornerIndices[0] * kPartitionNumber + kCornerIndices[1];
                    // Calculating the index of the bottom-right corner
                    const size_t kIdx2d11 = kCornerHorizontalIndex11 * kPartitionNumber + kCornerVerticalIndex11;

                    // Coordinates of the top-left and bottom-right corners in the destination image
                    auto& indexPair00 = gridCornerCoordinatesH[kIdx2d00];

                    std::tuple<int, int, double, double> indexPair11;
                    if (kCornerVerticalIndex11 >= kPartitionNumber ||
                        kCornerHorizontalIndex11 >= kPartitionNumber)
                    {
                        // Get the coordinates of the corner
                        const double kX11 = kCornerHorizontalIndex11 * kCellSize1,
                            kY11 = kCornerVerticalIndex11 * kCellSize2;

                        // Project them by the estimated homography matrix
                        double x2p = kX11 * descriptor(0, 0) + kY11 * descriptor(0, 1) + descriptor(0, 2),
                            y2p = kX11 * descriptor(1, 0) + kY11 * descriptor(1, 1) + descriptor(1, 2),
                            h2p = kX11 * descriptor(2, 0) + kY11 * descriptor(2, 1) + descriptor(2, 2);

                        x2p /= h2p;
                        y2p /= h2p;

                        indexPair11 = std::tuple<int, int, double, double>(x2p / kCellSize3, y2p / kCellSize4, x2p, y2p);
                    } 
                    else
                        indexPair11 = gridCornerCoordinatesH[kIdx2d11];

                    const double &l1x = std::get<2>(indexPair00) - inlierOutlierThreshold_;
                    const double &l1y = std::get<3>(indexPair00) - inlierOutlierThreshold_;
                    const double &r1x = std::get<2>(indexPair11) + inlierOutlierThreshold_;
                    const double &r1y = std::get<3>(indexPair11) + inlierOutlierThreshold_;

                    const double l2x = kCellSize3 * kCornerIndices[2];
                    const double l2y = kCellSize4 * kCornerIndices[3];
                    const double r2x = l2x + kCellSize3;
                    const double r2y = l2y + kCellSize4;

                    // If one rectangle is on left side of other
                    if (l1x <= r2x && l2x <= r1x ||
                        // If one rectangle is above other
                        r1y <= l2y && r2y <= l1y)
                    {
                        // Store the points in the cell to be tested
                        selectedCells_.emplace_back(&points);
                        pointNumber_ += points.size();
                        overlaps = true;
                    }
                }
            }
        }
    }
}