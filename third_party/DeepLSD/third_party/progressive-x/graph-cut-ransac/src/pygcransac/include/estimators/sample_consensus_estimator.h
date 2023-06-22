// Copyright (C) 2013 The Regents of the University of California (Regents).
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
//     * Neither the name of The Regents or University of California nor the
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
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

#ifndef THEIA_SOLVERS_SAMPLE_CONSENSUS_ESTIMATOR_H_
#define THEIA_SOLVERS_SAMPLE_CONSENSUS_ESTIMATOR_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>
#include <iostream>
#include "estimator.h"
#include "quality_measurement.h"
#include "sampler.h"

namespace theia
{

	// Helper struct to hold parameters to various RANSAC schemes. error_thresh is
	// the threshold for consider data points to be inliers to a model. This is the
	// only variable that must be explitly set, and the rest can be used with the
	// default values unless other values are desired.
	struct RansacParameters
	{
		RansacParameters()
			: error_thresh(-1),
			failure_probability(0.01),
			min_inlier_ratio(0.1),
			max_iterations(std::numeric_limits<int>::max()),
			use_Tdd_test(false) {}

		// residual threshold to determin inliers for RANSAC (e.g., squared reprojection
		// error). This is what will be used by the estimator to determine inliers.
		double error_thresh;

		// The failure probability of RANSAC. Set to 0.01 means that RANSAC has a 1%
		// chance of missing the correct pose.
		double failure_probability;

		// The minimal assumed inlier ratio, i.e., it is assumed that the given set
		// of correspondences has an inlier ratio of at least min_inlier_ratio.
		// This is required to limit the number of RANSAC iteratios.
		double min_inlier_ratio;

		// Another way to specify the maximal number of RANSAC iterations. In effect,
		// the maximal number of iterations is set to min(max_ransac_iterations, T),
		// where T is the number of iterations corresponding to min_inlier_ratio.
		// This variable is useful if RANSAC is to be applied iteratively, i.e.,
		// first applying RANSAC with an min_inlier_ratio of x, then with one
		// of x-y and so on, and we want to avoid repeating RANSAC iterations.
		// However, the preferable way to limit the number of RANSAC iterations is
		// to set min_inlier_ratio and leave max_ransac_iterations to its default
		// value.
		// Per default, this variable is set to std::numeric_limits<int>::max().
		int max_iterations;

		// Whether to use the T_{d,d}, with d=1, test proposed in
		// Chum, O. and cv::Matas, J.: Randomized RANSAC and T(d,d) test, BMVC 2002.
		// After computing the pose, RANSAC selects one match at random and evaluates
		// all poses. If the point is an outlier to one pose, the corresponding pose
		// is rejected. Notice that if the pose solver returns multiple poses, then
		// at most one pose is correct. If the selected match is correct, then only
		// the correct pose will pass the test. Per default, the test is disabled.
		//
		// NOTE: Not currently implemented!
		bool use_Tdd_test;
	};

	// A struct to hold useful outputs of Ransac-like methods.
	struct RansacSummary
	{
		// Contains the indices of all inliers.
		std::vector<int> inliers;

		// The number of iterations performed before stopping RANSAC.
		int num_iterations;

		// The confidence in the solution.
		double confidence;
	};

	template <class _ModelEstimator> class SampleConsensusEstimator
	{
	public:
		typedef typename _ModelEstimator::Datum Datum;
		typedef typename _ModelEstimator::Model Model;

		SampleConsensusEstimator(const RansacParameters& ransac_params,
			const _ModelEstimator& estimator);

		virtual bool initialize()
		{
			return true;
		}

		virtual ~SampleConsensusEstimator() {}

		// Computes the best-fitting model using RANSAC. Returns false if RANSAC
		// calculation fails and true (with the best_model output) if successful.
		// Params:
		//   data: the set from which to sample
		//   estimator: The estimator used to estimate the model based on the Datum
		//     and Model type
		//   best_model: The output parameter that will be filled with the best model
		//     estimated from RANSAC
		virtual bool Estimate(const std::vector<Datum>& data,
			Model* best_model,
			RansacSummary* summary);

		std::vector<double> GetOutputValues() { return output_values_; }

	protected:
		// This method is called from derived classes to set up the sampling scheme
		// and the method for computing inliers. It must be called by derived classes
		// unless they override the Estimate(...) method.
		//
		// sampler: The class that instantiates the sampling strategy for this
		//   particular type of sampling consensus.
		// quality_measurement: class that instantiates the quality measurement of
		//   the data. This determines the stopping criterion.
		bool initialize(Sampler<Datum>* sampler,
			QualityMeasurement* quality_measurement);

		// Computes the maximum number of iterations required to ensure the inlier
		// ratio is the best with a probability corresponding to log_failure_prob.
		int ComputeMaxIterations(const double min_sample_size,
			const double inlier_ratio,
			const double log_failure_prob) const;

		// The sampling strategy.
		std::unique_ptr<Sampler<Datum> > sampler_;

		// The quality metric for the estimated model and data.
		std::unique_ptr<QualityMeasurement> quality_measurement_;

		// Ransac parameters (see above struct).
		const RansacParameters& ransac_params_;

		// Estimator to use for generating models.
		const _ModelEstimator& estimator_;

		std::vector<double> output_values_;
	};

	// --------------------------- Implementation --------------------------------//

	template <class _ModelEstimator>
	SampleConsensusEstimator<_ModelEstimator>::SampleConsensusEstimator(
		const RansacParameters& ransac_params, const _ModelEstimator& estimator)
		: ransac_params_(ransac_params), estimator_(estimator)
	{
		if (ransac_params.error_thresh <= 0)
			std::cout << "Error threshold must be set to greater than zero" << " @" << __LINE__ << std::endl;
		if (ransac_params.min_inlier_ratio > 1.0)
			std::cout << "Error min_inlier_ratio must be set less than or equal to 1.0" << " @" << __LINE__ << std::endl;
		if (ransac_params.min_inlier_ratio < 0.0)
			std::cout << "Error min_inlier_ratio must be set greater than or equal to 0.0" << " @" << __LINE__ << std::endl;
		if (ransac_params.failure_probability >= 1.0)
			std::cout << "Error failure_probability must be set less than 1.0" << " @" << __LINE__ << std::endl;
		if (ransac_params.failure_probability <= 0.0)
			std::cout << "Error failure_probability must be set greater than 0.0" << " @" << __LINE__ << std::endl;
	}

	template <class _ModelEstimator>
	bool SampleConsensusEstimator<_ModelEstimator>::initialize(
		Sampler<Datum>* sampler,
		QualityMeasurement* quality_measurement)
	{
		if (sampler == NULL)
			std::cout << "Error sampler must not be null" << " @" << __LINE__ << std::endl;
		if (quality_measurement == NULL)
			std::cout << "Error quality_measurement must not be null" << " @" << __LINE__ << std::endl;

		sampler_.reset(sampler);
		if (!sampler_->initialize())
		{
			return false;
		}

		quality_measurement_.reset(quality_measurement);
		return quality_measurement_->initialize();
	}

	template <class _ModelEstimator>
	int SampleConsensusEstimator<_ModelEstimator>::ComputeMaxIterations(
		const double min_sample_size,
		const double inlier_ratio,
		const double log_failure_prob) const
	{
		if (inlier_ratio <= 0.0)
			std::cout << "Error inlier_ratio must be greater than 0.0" << " @" << __LINE__ << std::endl;

		int num_iterations = 1;
		if (inlier_ratio < 1.0)
		{
			double num_samples = min_sample_size;
			if (ransac_params_.use_Tdd_test)
			{
				// If we use the T_{1,1} test, we have to adapt the number of samples
				// that needs to be generated accordingly since we use another
				// match for verification and a correct match is selected with probability
				// inlier_ratio.
				num_samples += 1.0;
			}
			double log_prob = log(1.0 - pow(inlier_ratio, num_samples));
			num_iterations = static_cast<int>(std::floor(
				log_failure_prob / log_prob) + 1.0);
		}
		return std::min(num_iterations, ransac_params_.max_iterations);
	}

	template <class _ModelEstimator>
	bool SampleConsensusEstimator<_ModelEstimator>::Estimate(
		const std::vector<Datum>& data,
		Model* best_model,
		RansacSummary* summary)
	{
		if (data.size() <= 0)
			std::cout << "Cannot perform estimation with 0 data measurements!" << " @" << __LINE__ << std::endl;
		if (sampler_.get() == NULL)
			std::cout << "Error sampler_.get() must not be null" << std::endl;
		if (quality_measurement_.get() == NULL)
			std::cout << "Error quality_measurement_.get() must not be null" << " @" << __LINE__ << std::endl;
		if (summary == NULL)
			std::cout << "Error summary must not be null" << " @" << __LINE__ << std::endl;

		double best_quality = static_cast<double>(QualityMeasurement::INVALID);
		int max_iterations = ransac_params_.max_iterations;
		const double log_failure_prob = log(ransac_params_.failure_probability);

		for (summary->num_iterations = 0;
			summary->num_iterations < max_iterations;
			summary->num_iterations++)
		{

			// sample subset. Proceed if successfully sampled.
			std::vector<Datum> data_subset;
			if (!sampler_->sample(data, &data_subset))
			{
				continue;
			}

			// Estimate model from subset. Skip to next iteration if the model fails to
			// estimate.
			std::vector<Model> temp_models;
			if (!estimator_.estimateModel(data_subset, &temp_models))
			{
				continue;
			}

			// Calculate residuals from estimated model.
			for (const Model& temp_model : temp_models)
			{
				std::vector<double> residuals = estimator_.residuals(data, temp_model);

				// Determine quality of the generated model.
				double sample_quality =
					quality_measurement_->Calculate(residuals);

				// Update best model if error is the best we have seen.
				if (quality_measurement_->Compare(sample_quality, best_quality) ||
					best_quality == static_cast<double>(QualityMeasurement::INVALID))
				{
					*best_model = temp_model;
					best_quality = sample_quality;
					max_iterations = ComputeMaxIterations(
						estimator_.sampleSize(),
						std::max(quality_measurement_->GetInlierRatio(),
						ransac_params_.min_inlier_ratio),
						log_failure_prob);
				}
			}
		}

		summary->inliers =
			estimator_.getInliers(data, *best_model, ransac_params_.error_thresh);
		const double inlier_ratio =
			static_cast<double>(summary->inliers.size()) / data.size();
		summary->confidence =
			1.0 - pow(1.0 - pow(inlier_ratio, estimator_.sampleSize()),
			summary->num_iterations);

		return true;
	}

}  // namespace theia

#endif  // THEIA_SOLVERS_SAMPLE_CONSENSUS_ESTIMATOR_H_
