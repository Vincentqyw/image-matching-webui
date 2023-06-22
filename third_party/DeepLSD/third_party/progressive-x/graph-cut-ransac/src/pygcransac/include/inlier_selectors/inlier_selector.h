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

namespace gcransac
{
	namespace inlier_selector
	{
        template <
            typename _Estimator,
            typename _NeighborhoodStructure>
        class AbstractInlierSelector
        {
        public:
            explicit AbstractInlierSelector(const _NeighborhoodStructure *kNeighborhoodGraph_)
            {

            }

			virtual ~AbstractInlierSelector() {}

            static constexpr bool doesSomething() { return false; }

            // The function that runs the model-based inlier selector
            virtual void run(
                const cv::Mat& kCorrespondences_, // All point correspondences
                const gcransac::Model& kModel_, // The model parameters
                const _NeighborhoodStructure& kNeighborhood_, // The neighborhood structure. This probably will be a GridNeighborhood currently.
                const double& inlierOutlierThreshold_,
                std::vector<const std::vector<size_t>*>& selectedCells_, // The indices of the points selected
                size_t& pointNumber_) = 0; 
        };
    }
}