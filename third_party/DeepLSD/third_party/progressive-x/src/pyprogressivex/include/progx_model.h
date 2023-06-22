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

#include <opencv2/core/core.hpp>
#include <vector>
#include <Eigen/Eigen>
#include "model.h"

namespace progx
{
	template<class _ModelEstimator>
	class Model : public gcransac::Model
	{
	public:
		const _ModelEstimator * estimator;
		Eigen::VectorXd preference_vector;

		double probability,
			preference_vector_length,
			tanimoto_distance,
			eucledian_distance;

		const _ModelEstimator * const getEstimator() const
		{
			return estimator;
		}

		void setEstimator(const _ModelEstimator * estimator_)
		{
			estimator = estimator_;
		}

		void setDescriptor(const Eigen::MatrixXd &descriptor_)
		{
			descriptor = descriptor_;
		}

		void setPreferenceVector(const cv::Mat &data_,
			const double &truncated_squared_threshold_)
		{
			const size_t point_number = data_.rows;
			preference_vector.resize(point_number);

			double squared_residual;
			for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
			{
				// The point-to-model residual
				squared_residual = estimator->squaredResidual(data_.row(point_idx), *this);

				// Update the preference vector of the current model since it might be changed
				// due to the optimization.
				preference_vector(point_idx) =
					MAX(0, 1.0 - squared_residual / truncated_squared_threshold_);
			}
		}

		Model(const Eigen::MatrixXd &descriptor_) :
			gcransac::Model(descriptor_)
		{

		}

		Model()
		{

		}
	};
}