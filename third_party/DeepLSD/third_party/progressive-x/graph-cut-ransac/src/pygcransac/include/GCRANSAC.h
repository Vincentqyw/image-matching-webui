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

#include "GCoptimization.h"
#include "model.h"
#include "settings.h"
#include "statistics.h"
#include "scoring_function.h"
#include "samplers/sampler.h"
#include "preemption/preemption_empty.h"
#include "inlier_selectors/empty_inlier_selector.h"


namespace gcransac
{
	template <class _ModelEstimator,
		class _NeighborhoodGraph,
		class _ScoringFunction = MSACScoringFunction<_ModelEstimator>,
		class _PreemptiveModelVerification = preemption::EmptyPreemptiveVerfication<_ModelEstimator>,
		class _FastInlierSelector = inlier_selector::EmptyInlierSelector<_ModelEstimator, _NeighborhoodGraph>>
		class GCRANSAC
	{
	public:
		utils::Settings settings;

		GCRANSAC() :
			time_limit(std::numeric_limits<double>::max()),
			scoring_function(std::make_unique<_ScoringFunction>())
		{
		}
		~GCRANSAC() 
		{ 
			_ScoringFunction * scoring_function_ptr = scoring_function.release();
			delete scoring_function_ptr;
		}

		// The main method applying Graph-Cut RANSAC to the input data points
		void run(const cv::Mat &points_,  // Data points
			const _ModelEstimator &estimator_, // The model estimator
			sampler::Sampler<cv::Mat, size_t> *main_sampler_, // The main sampler is used outside the local optimization
			sampler::Sampler<cv::Mat, size_t> *local_optimization_sampler_, // The local optimization sampler is used inside the local optimization
			const _NeighborhoodGraph *neighborhood_graph_, // The initialized neighborhood graph
			Model &obtained_model_, // The output model
			_PreemptiveModelVerification &preemptive_verification_, // The preemptive verification strategy used
			_FastInlierSelector &fast_inlier_selector_); // The fast inlier selector used

		// The main method applying Graph-Cut RANSAC to the input data points
		void run(const cv::Mat &points_,  // Data points
			const _ModelEstimator &estimator_, // The model estimator
			sampler::Sampler<cv::Mat, size_t> *main_sampler_, // The main sampler is used outside the local optimization
			sampler::Sampler<cv::Mat, size_t> *local_optimization_sampler_, // The local optimization sampler is used inside the local optimization
			const _NeighborhoodGraph *neighborhood_graph_, // The initialized neighborhood graph
			Model &obtained_model_); // The output model

		void setFPS(double fps_) { settings.desired_fps = fps_; time_limit = 1.0 / fps_; } // Set a desired FPS value

		// Return the constant reference of the scoring function
		const utils::RANSACStatistics &getRansacStatistics() const { return statistics; }

		// Return the reference of the scoring function
		utils::RANSACStatistics &getMutableRansacStatistics() { return statistics; }

		// Return the constant reference of the scoring function
		const _ScoringFunction &getScoringFunction() const { return *scoring_function; }

		// Return the reference of the scoring function
		_ScoringFunction &getMutableScoringFunction() { return *scoring_function; }

	protected:
		double time_limit; // The desired time limit
		std::vector<std::vector<cv::DMatch>> neighbours; // The neighborhood structure
		utils::RANSACStatistics statistics; // RANSAC statistics
		int point_number; // The point number
		double truncated_threshold; // 3 / 2 * threshold_
		double squared_truncated_threshold; // 9 / 4 * threshold_^2
		int step_size; // Step size per processes
		double log_probability; // The logarithm of 1 - confidence
		const _NeighborhoodGraph *neighborhood_graph;
		sampler::Sampler<cv::Mat, size_t> *main_sampler; // The main sampler is used outside the local optimization
		sampler::Sampler<cv::Mat, size_t> *local_optimization_sampler; // The local optimization sampler is used inside the local optimization
		std::unique_ptr<_ScoringFunction> scoring_function; // The scoring function used to measure the quality of a model

		Graph<double, double, double> *graph; // The graph for graph-cut

		template<size_t _CalledFromLocalOptimization>
		OLGA_INLINE bool sample(const std::vector<size_t> &pool_,
			size_t sample_number_,
			size_t *sample_);

		// Computes the desired iteration number for RANSAC w.r.t. to the current inlier number
		size_t getIterationNumber(size_t inlier_number_, // The inlier number
			size_t point_number_, // The point number
			size_t sample_size_, // The current_sample size
			double log_probability_); // The logarithm of the desired probability

		// Returns a labeling w.r.t. the current model and point set
		void labeling(const cv::Mat &points_, // The input data points
			size_t neighbor_number_, // The neighbor number in the graph
			const std::vector<std::vector<cv::DMatch>> &neighbors_, // The neighborhood
			Model &model_, // The current model_
			_ModelEstimator estimator_, // The model estimator
			double lambda_, // The weight for the spatial coherence term
			double threshold_, // The threshold_ for the inlier-outlier decision
			std::vector<size_t> &inliers_, // The resulting inlier set
			double &energy_); // The resulting energy

		// Apply the graph-cut optimization for GC-RANSAC
		bool graphCutLocalOptimization(const cv::Mat &points_, // The input data points
			std::vector<size_t> &so_far_the_best_inliers_, // The input, than the resulting inlier set
			Model &so_far_the_best_model_, // The current model
			Score &so_far_the_best_score_, // The current score
			const _ModelEstimator &estimator_, // The model estimator
			const size_t trial_number_); // The max trial number

		// Model fitting by the iterated least squares method
		bool iteratedLeastSquaresFitting(
			const cv::Mat &points_, // The input data points
			const _ModelEstimator &estimator_, // The model estimator
			const double threshold_, // The inlier-outlier threshold
			std::vector<size_t> &inliers_, // The resulting inlier set
			Model &model_, // The estimated model
			const bool use_weighting_ = true); // Use iteratively re-weighted least-squares
	};

	// Computes the desired iteration number for RANSAC w.r.t. to the current inlier number
	template <class _ModelEstimator, class _NeighborhoodGraph, class _ScoringFunction, class _PreemptiveModelVerification, class _FastInlierSelector>
	size_t GCRANSAC<_ModelEstimator, _NeighborhoodGraph, _ScoringFunction, _PreemptiveModelVerification, _FastInlierSelector>::getIterationNumber(
		size_t inlier_number_,
		size_t point_number_,
		size_t sample_size_,
		double log_probability_)
	{
		const double q = pow(static_cast<double>(inlier_number_) / point_number_, sample_size_);
		const double log2 = log(1 - q);

		if (abs(log2) < std::numeric_limits<double>::epsilon())
			return std::numeric_limits<size_t>::max();

		const double iter = log_probability_ / log2;
		return static_cast<size_t>(iter) + 1;
	}

	// The main method applying Graph-Cut RANSAC to the input data points
	template <class _ModelEstimator, class _NeighborhoodGraph, class _ScoringFunction, class _PreemptiveModelVerification, class _FastInlierSelector>
	void GCRANSAC<_ModelEstimator, _NeighborhoodGraph, _ScoringFunction, _PreemptiveModelVerification, _FastInlierSelector>::run(const cv::Mat &points_,  // Data points
		const _ModelEstimator &estimator_, // The model estimator
		sampler::Sampler<cv::Mat, size_t> *main_sampler_, // The main sampler is used outside the local optimization
		sampler::Sampler<cv::Mat, size_t> *local_optimization_sampler_, // The local optimization sampler is used inside the local optimization
		const _NeighborhoodGraph *neighborhood_graph_, // The initialized neighborhood graph
		Model &obtained_model_) // The output model
	{
		// Instantiate the preemptive model verification strategy
		_PreemptiveModelVerification preemptive_verification;
		
		// Instantiate the fast inlier selector object
		_FastInlierSelector fast_inlier_selector(neighborhood_graph_);

		// Running GC-RANSAC by using the specified preemptive verification
		run(points_,
			estimator_,
			main_sampler_,
			local_optimization_sampler_,
			neighborhood_graph_,
			obtained_model_,
			preemptive_verification,
			fast_inlier_selector);
	}

	// The main method applying Graph-Cut RANSAC to the input data points_
	template <class _ModelEstimator, class _NeighborhoodGraph, class _ScoringFunction, class _PreemptiveModelVerification, class _FastInlierSelector>
	void GCRANSAC<_ModelEstimator, _NeighborhoodGraph, _ScoringFunction, _PreemptiveModelVerification, _FastInlierSelector>::run(
		const cv::Mat &points_,  // Data points
		const _ModelEstimator &estimator_, // The model estimator
		sampler::Sampler<cv::Mat, size_t> *main_sampler_, // The main sampler is used outside the local optimization
		sampler::Sampler<cv::Mat, size_t> *local_optimization_sampler_, // The local optimization sampler is used inside the local optimization
		const _NeighborhoodGraph *neighborhood_graph_, // The initialized neighborhood graph
		Model &obtained_model_,  // The output model 
		_PreemptiveModelVerification &preemptive_verification_, // The preemptive verification strategy used
		_FastInlierSelector &fast_inlier_selector_) // The fast inlier selector used
	{
		/*
			Initialization
		*/
		// Variables for measuring the processing time
		std::chrono::time_point<std::chrono::system_clock> start, end;
		std::chrono::duration<double> elapsed_seconds;

		statistics.main_sampler_name = main_sampler_->getName();
		statistics.local_optimizer_sampler_name = local_optimization_sampler_->getName();
		statistics.iteration_number = 0;
		statistics.graph_cut_number = 0;
		statistics.local_optimization_number = 0;
		statistics.neighbor_number = 0;
		statistics.processing_time = 0.0;

		main_sampler = main_sampler_;
		local_optimization_sampler = local_optimization_sampler_;

		// The size of a minimal sample used for the estimation
		constexpr size_t sample_number = _ModelEstimator::sampleSize();

		// log(1 - confidence) used for determining the required number of iterations
		log_probability = log(1.0 - settings.confidence);
		// Maximum number of iterations
		auto max_iteration =
			getIterationNumber(1, points_.rows, sample_number, log_probability);

		std::unique_ptr<size_t[]> current_sample(new size_t[sample_number]); // Minimal sample for model fitting
		bool do_local_optimization = false; // Flag to show if local optimization should be applied
		point_number = points_.rows; // Number of points in the dataset

		size_t inlier_container_offset = 0; // Index to show which inlier vector is currently in use
		Model so_far_the_best_model; // The current so-far-the-best model parameters
		Score so_far_the_best_score; // The score of the current so-far-the-best model
		std::vector<std::vector<size_t>> temp_inner_inliers(2); // The inliers of the current and previous best models 
		temp_inner_inliers[0].reserve(point_number);
		temp_inner_inliers[1].reserve(point_number);

		neighborhood_graph = neighborhood_graph_; // The neighborhood graph used for the graph-cut local optimization
		truncated_threshold = 3.0 / 2.0 * settings.threshold; // The truncated least-squares threshold
		squared_truncated_threshold = truncated_threshold * truncated_threshold; // The squared least-squares threshold
		scoring_function->initialize(squared_truncated_threshold, point_number); // Initializing the scoring function

		// Initialize the pool for sampling
		std::vector<size_t> pool(point_number);
		for (size_t i = 0; i < point_number; ++i)
			pool[i] = i;

		// Initialize the starting time if there is a desired FPS set
		start = std::chrono::system_clock::now();

		// The model estimated from a minimal subset
		std::vector<Model> models;
		models.reserve(estimator_.maximumMinimalSolutions());
		
		// Variables used for the fast inlier selection if needed
		std::vector<const std::vector<size_t>*> preselected_index_sets;  // The indices of the points selected by the proposed approach
		size_t selected_point_number;
		// Initializing the variables if the inlier selection is used
		if constexpr (_FastInlierSelector::doesSomething())
		{
			const auto &cell_number = neighborhood_graph->filledCellNumber();
			preselected_index_sets.reserve(cell_number * cell_number); // Occupying the required memory
		}

		/*
			The main RANSAC iteration
		*/
		while (settings.min_iteration_number > statistics.iteration_number ||
			statistics.iteration_number < MIN(max_iteration, settings.max_iteration_number))
		{
			// Do not apply local optimization if not needed
			do_local_optimization = false;

			// Increase the iteration number counter
			++statistics.iteration_number;

			// Resize the vector containing the models
			models.resize(0);

			int unsuccessful_model_generations = -1;
			// Select a minimal sample and estimate the implied model parameters if possible.
			// If, after a certain number of sample selections, there is no success, terminate.
			while (++unsuccessful_model_generations < settings.max_unsuccessful_model_generations)
			{
				// If the sampling is not successful, try again.
				if (!sample<0>(pool, // The current pool from which the points are chosen
					sample_number, // Number of points to select
					current_sample.get())) // The current sample
					continue;

				// Check if the selected sample is valid before estimating the model
				// parameters which usually takes more time. 
				if (!estimator_.isValidSample(points_, // All points
					current_sample.get())) // The current sample
					continue;

				// Estimate the model parameters using the current sample
				if (estimator_.estimateModel(points_,  // All points
					current_sample.get(), // The current sample
					&models)) // The estimated model parameters
					break;
			}

			// Increase the iteration number by the number of unsuccessful model generations as well.
			statistics.iteration_number += unsuccessful_model_generations;

			// Select the so-far-the-best from the estimated models
			for (auto &model : models)
			{
				// The score of the current model
				Score score;

				// Do point pre-filtering if needed.
				if constexpr (_FastInlierSelector::doesSomething())
				{
					// Remove the previous selection
					preselected_index_sets.clear();
					selected_point_number = 0;

					// Get the indices of the points using the proposed grid-based selection
					fast_inlier_selector_.run(
						points_, // The point correspondences
						model, // The model to be verified
						*neighborhood_graph_, // The neighborhood structure
						truncated_threshold, // The inlier-outlier threshold
						preselected_index_sets, // The 4D cells selected by the algorithm
						selected_point_number); // The total number of points in the cells

					// Check if the inlier upper bound is lower than the inlier number of 
					// the so-far-the-best model. 
					if (selected_point_number < so_far_the_best_score.inlier_number)
					{
						++statistics.rejected_models;
						continue;
					}
					++statistics.accepted_models;
				} 

				// Check if the model should be rejected by the used preemptive verification strategy.
				// If there is no preemption, i.e. EmptyPreemptiveVerfication is used, this should be skipped.
				if constexpr (!std::is_same<preemption::EmptyPreemptiveVerfication<_ModelEstimator>, _PreemptiveModelVerification>())
				{
					bool should_reject = false;
					// Use the pre-selected inlier indices if the pre-selection is applied
					if constexpr (_FastInlierSelector::doesSomething())
					{
						if (!preemptive_verification_.verifyModel(model, // The current model
							estimator_, // The model estimation object
							truncated_threshold, // The truncated threshold
							statistics.iteration_number, // The current iteration number
							so_far_the_best_score, // The current best score
							points_, // The data points
							current_sample.get(), // The current minimal sample
							sample_number, // The number of samples used
							temp_inner_inliers[inlier_container_offset], // The current inlier set
							score,
							&preselected_index_sets))
							should_reject = true;
					}
					else
					{
						if (!preemptive_verification_.verifyModel(model, // The current model
							estimator_, // The model estimation object
							truncated_threshold, // The truncated threshold
							statistics.iteration_number, // The current iteration number
							so_far_the_best_score, // The current best score
							points_, // The data points
							current_sample.get(), // The current minimal sample
							sample_number, // The number of samples used
							temp_inner_inliers[inlier_container_offset], // The current inlier set
							score))
							should_reject = true;
					}

					if (should_reject)
					{
						++statistics.rejected_models;
						continue;
					}
					++statistics.accepted_models;
				}

				// Get the inliers and the score of the non-optimized model
				if constexpr (!_PreemptiveModelVerification::providesScore())
				{
					// Use the pre-selected inlier indices if the pre-selection is applied
					if constexpr (_FastInlierSelector::doesSomething())
						score = scoring_function->getScore(points_, // All points
							model, // The current model parameters
							estimator_, // The estimator 
							settings.threshold, // The current threshold
							temp_inner_inliers[inlier_container_offset], // The current inlier set
							so_far_the_best_score, // The score of the current so-far-the-best model
							true, // Flag to decide if the inliers are needed
							&preselected_index_sets); // The point index set consisting of the pre-selected points' indices
					else // Otherwise, run on all points
						score = scoring_function->getScore(points_, // All points
							model, // The current model parameters
							estimator_, // The estimator 
							settings.threshold, // The current threshold
							temp_inner_inliers[inlier_container_offset], // The current inlier set
							so_far_the_best_score, // The score of the current so-far-the-best model
							true); // Flag to decide if the inliers are needed
				}

				bool is_model_updated = false;

				// Store the model of its score is higher than that of the previous best
				if (so_far_the_best_score < score && // Comparing the so-far-the-best model's score and current model's score
					estimator_.isValidModel(model, // The current model parameters
						points_, // All input points
						temp_inner_inliers[inlier_container_offset], // The inliers of the current model
						current_sample.get(), // The minimal sample initializing the model
						truncated_threshold, // The truncated inlier-outlier threshold
						is_model_updated))
				{
					// Get the inliers and the score of the non-optimized model
					if (is_model_updated)
						score = scoring_function->getScore(points_, // All points
							model, // The current model parameters
							estimator_, // The estimator 
							settings.threshold, // The current threshold
							temp_inner_inliers[inlier_container_offset], // The current inlier set
							so_far_the_best_score, // The score of the current so-far-the-best model
							true); // Flag to decide if the inliers are needed

					inlier_container_offset = (inlier_container_offset + 1) % 2;
					so_far_the_best_model = model; // The new so-far-the-best model
					so_far_the_best_score = score; // The new so-far-the-best model's score
					// Decide if local optimization is needed. The current criterion requires a minimum number of iterations
					// and number of inliers before applying GC.
					do_local_optimization = statistics.iteration_number > settings.min_iteration_number_before_lo &&
						so_far_the_best_score.inlier_number > sample_number;

					// Update the number of maximum iterations
					max_iteration = getIterationNumber(so_far_the_best_score.inlier_number, // The inlier number of the current best model
						point_number, // The number of points
						sample_number, // The sample size
						log_probability); // The logarithm of 1 - confidence
				}
			}

			// Apply local optimziation
			if (settings.do_local_optimization && // Input flag to decide if local optimization is needed
				do_local_optimization) // A flag to decide if all the criteria meet to apply local optimization
			{
				// Increase the number of local optimizations applied
				++statistics.local_optimization_number;

				// Graph-cut-based local optimization 
				graphCutLocalOptimization(points_, // All points
					temp_inner_inliers[inlier_container_offset], // Inlier set of the current so-far-the-best model
					so_far_the_best_model, // Best model parameters
					so_far_the_best_score, // Best model score
					estimator_, // Estimator
					settings.max_local_optimization_number); // Maximum local optimization steps

				// Update the maximum number of iterations variable
				max_iteration =
					getIterationNumber(so_far_the_best_score.inlier_number, // The current inlier number
						point_number, // The number of points
						sample_number, // The sample size
						log_probability); // log(1 - confidence)
			}

			// Apply time limit if there is a required FPS set
			if (settings.desired_fps > -1)
			{
				end = std::chrono::system_clock::now(); // The current time
				elapsed_seconds = end - start; // Time elapsed since the algorithm started

				// Interrupt the algorithm if the time limit is exceeded
				if (elapsed_seconds.count() > time_limit)
				{
					settings.min_iteration_number = 0;
					max_iteration = 0;
					//break;
				}
			}
		}

		// If the best model has only minimal number of points, the model
		// is not considered to be found. 
		if (so_far_the_best_score.inlier_number <= sample_number)
		{
			end = std::chrono::system_clock::now(); // The current time
			elapsed_seconds = end - start; // Time elapsed since the algorithm started
			statistics.processing_time = elapsed_seconds.count();
			return;
		}

		// Apply a final local optimization if it hasn't been applied yet
		if (settings.do_local_optimization &&
			statistics.local_optimization_number == 0)
		{
			// Increase the number of local optimizations applied
			++statistics.local_optimization_number;

			// Graph-cut-based local optimization 
			graphCutLocalOptimization(points_, // All points
				temp_inner_inliers[inlier_container_offset], // Inlier set of the current so-far-the-best model
				so_far_the_best_model, // Best model parameters
				so_far_the_best_score, // Best model score
				estimator_, // Estimator
				settings.max_local_optimization_number); // Maximum local optimization steps
		}

		// Recalculate the score if needed (i.e. there is some inconstistency in
		// in the number of inliers stored and calculated).
		if (temp_inner_inliers[inlier_container_offset].size() != so_far_the_best_score.inlier_number)
			inlier_container_offset = (inlier_container_offset + 1) % 2;

		if (temp_inner_inliers[inlier_container_offset].size() != so_far_the_best_score.inlier_number)
			so_far_the_best_score = scoring_function->getScore(points_, // All points
				so_far_the_best_model, // Best model parameters
				estimator_, // The estimator
				settings.threshold, // The inlier-outlier threshold
				temp_inner_inliers[inlier_container_offset]); // The current inliers

		// Apply iteration least-squares fitting to get the final model parameters if needed
		bool iterative_refitting_applied = false;
		if (settings.do_final_iterated_least_squares)
		{
			Model model = so_far_the_best_model; // The model which is re-estimated by iteratively re-weighted least-squares
			bool success = iteratedLeastSquaresFitting(
				points_, // The input data points
				estimator_, // The model estimator
				settings.threshold, // The inlier-outlier threshold
				temp_inner_inliers[inlier_container_offset], // The resulting inlier set
				model); // The estimated model

			if (success)
			{
				size_t inlier_container_idx = (inlier_container_offset + 1) % 2;
				temp_inner_inliers[inlier_container_idx].clear();
				Score score = scoring_function->getScore(points_, // All points
					model, // Best model parameters
					estimator_, // The estimator
					settings.threshold, // The inlier-outlier threshold
					temp_inner_inliers[inlier_container_idx]); // The current inliers

				if (so_far_the_best_score < score)
				{
					iterative_refitting_applied = true;
					so_far_the_best_model.descriptor = model.descriptor;
					inlier_container_offset = inlier_container_idx;
				}
			}
		}
		
		if (!iterative_refitting_applied) // Otherwise, do only one least-squares fitting on all of the inliers
		{
			// Estimate the final model using the full inlier set
			models.clear();
			estimator_.estimateModelNonminimal(points_,
				&(temp_inner_inliers[inlier_container_offset])[0],
				so_far_the_best_score.inlier_number,
				&models);

			if (models.size() > 0)
				so_far_the_best_model.descriptor = models[0].descriptor;
		}

		// Return the inlier set and the estimated model parameters
		statistics.inliers.swap(temp_inner_inliers[inlier_container_offset]);
		obtained_model_ = so_far_the_best_model;

		end = std::chrono::system_clock::now(); // The current time
		elapsed_seconds = end - start; // Time elapsed since the algorithm started
		statistics.processing_time = elapsed_seconds.count();
	}

	template <class _ModelEstimator, class _NeighborhoodGraph, class _ScoringFunction, class _PreemptiveModelVerification, class _FastInlierSelector>
	bool GCRANSAC<_ModelEstimator, _NeighborhoodGraph, _ScoringFunction, _PreemptiveModelVerification, _FastInlierSelector>::iteratedLeastSquaresFitting(
		const cv::Mat &points_,
		const _ModelEstimator &estimator_,
		const double threshold_,
		std::vector<size_t> &inliers_,
		Model &model_,
		const bool use_weighting_)
	{
		const size_t sample_size = estimator_.sampleSize(); // The minimal sample size
		if (inliers_.size() <= sample_size) // Return if there are not enough points
			return false;

		size_t iterations = 0; // Number of least-squares iterations
		std::vector<size_t> tmp_inliers; // Inliers of the current model

		// Iterated least-squares model fitting
		std::unique_ptr<double []> weights = std::make_unique<double []>(points_.rows);
		Score best_score; // The score of the best estimated model
		while (++iterations < settings.max_least_squares_iterations)
		{
			std::vector<Model> models; // Estimated models

			// Calculate the weights if iteratively re-weighted least-squares is used			
			if (_ModelEstimator::isWeightingApplicable() && // The weighted is applied only if the model estimator can handles it, and
				use_weighting_) // the user wants to apply it.
			{
				// Calculate Tukey bisquare weights of the inliers
				for (size_t inlier_idx = 0; inlier_idx < inliers_.size(); ++inlier_idx)
				{
					// The real index of the current inlier
					const size_t &point_idx = inliers_[inlier_idx];

					// The squares residual of the current inlier
					const double squared_residual = estimator_.squaredResidual(points_.row(point_idx), model_);

					// Calculate the Tukey bisquare weights
					const double weight = MAX(0.0, 1.0 - squared_residual / squared_truncated_threshold);
					weights[point_idx] = weight * weight;
				}

				// Estimate the model from the current inlier set
				estimator_.estimateModelNonminimal(points_,
					&(inliers_)[0], // The current inliers
					inliers_.size(), // The number of inliers
					&models, // The estimated model parameters
					weights.get()); // The weights used in the weighted least-squares fitting

				// Setting the weights back to 0.
				for (const size_t &inlier_idx : inliers_)
					weights[inlier_idx] = 0;
			}
			else
			{
				// Estimate the model from the current inlier set
				estimator_.estimateModelNonminimal(points_,
					&(inliers_)[0], // The current inliers
					inliers_.size(), // The number of inliers
					&models); // The estimated model parameters
			}

			if (models.size() == 0) // If there is no model estimated, interrupt the procedure
				break;
			if (models.size() == 1) // If a single model is estimated we do not have to care about selecting the best
			{
				// Calculate the score of the current model
				tmp_inliers.resize(0);
				Score score = scoring_function->getScore(points_, // All points
					models[0], // The current model parameters
					estimator_, // The estimator 
					threshold_, // The current threshold
					tmp_inliers); // The current inlier set

				// Break if the are not enough inliers
				if (tmp_inliers.size() < sample_size)
					break;

				// Interrupt the procedure if the inlier number has not changed.
				// Therefore, the previous and current model parameters are likely the same.
				if (score.inlier_number <= inliers_.size())
					break;

				// Update the output model
				model_ = models[0];
				// Store the inliers of the new model
				inliers_.swap(tmp_inliers);
			}
			else // If multiple models are estimated select the best (i.e. the one having the highest score) one
			{
				bool updated = false; // A flag determining if the model is updated

				// Evaluate all the estimated models to find the best
				for (auto &model : models)
				{
					// Calculate the score of the current model
					tmp_inliers.resize(0);
					Score score = scoring_function->getScore(points_, // The input data points
						model, // The model parameters
						estimator_, // The estimator
						threshold_, // The inlier-outlier threshold
						tmp_inliers); // The inliers of the current model

					// Continue if the are not enough inliers
					if (tmp_inliers.size() < sample_size)
						continue;

					// Do not test the model if the inlier number has not changed.
					// Therefore, the previous and current model parameters are likely the same.
					if (score.inlier_number <= inliers_.size())
						continue;

					// Update the model if its score is higher than that of the current best
					if (score.inlier_number >= best_score.inlier_number)
					{
						updated = true; // Set a flag saying that the model is updated, so the process should continue
						best_score = score; // Store the new score
						model_ = model; // Store the new model
						inliers_.swap(tmp_inliers); // Store the inliers of the new model
					}
				}

				// If the model has not been updated, interrupt the procedure
				if (!updated)
					break;
			}
		}

		// If there were more than one iterations, the procedure is considered successfull
		return iterations > 1;
	}

	template <class _ModelEstimator, class _NeighborhoodGraph, class _ScoringFunction, class _PreemptiveModelVerification, class _FastInlierSelector>
	template <size_t _CalledFromLocalOptimization>
	OLGA_INLINE bool GCRANSAC<_ModelEstimator, _NeighborhoodGraph, _ScoringFunction, _PreemptiveModelVerification, _FastInlierSelector>::sample(
		const std::vector<size_t> &pool_, // The pool if indices determining which point can be selected
		size_t sample_number_,
		size_t *sample_)
	{
		// Use a different sampler when the algorithm is inside the local optimization
		if constexpr (_CalledFromLocalOptimization)
			return local_optimization_sampler->sample(pool_, // The pool of indices
				sample_, // The selected sample
				sample_number_); // The number of points to be selected
		else
			// Apply the main sampler if the algorithm is not inside the local optimization
			return main_sampler->sample(pool_, // The pool of indices
				sample_, // The selected sample
				sample_number_); // The number of points to be selected
	}

	template <class _ModelEstimator, class _NeighborhoodGraph, class _ScoringFunction, class _PreemptiveModelVerification, class _FastInlierSelector>
	bool GCRANSAC<_ModelEstimator, _NeighborhoodGraph, _ScoringFunction, _PreemptiveModelVerification, _FastInlierSelector>::graphCutLocalOptimization(
		const cv::Mat &points_,
		std::vector<size_t> &so_far_the_best_inliers_,
		Model &so_far_the_best_model_,
		Score &so_far_the_best_score_,
		const _ModelEstimator &estimator_,
		const size_t trial_number_)
	{
		const auto &inlier_limit = estimator_.inlierLimit(); // Number of points used in the inner RANSAC
		Score max_score = so_far_the_best_score_; // The current best score
		Model best_model = so_far_the_best_model_; // The current best model
		std::vector<Model> models; // The estimated models' parameters
		std::vector<size_t> best_inliers, // Inliers of the best model
			inliers,
			tmp_inliers; // The inliers of the current model
		bool updated; // A flag to see if the model is updated
		double energy; // The energy after applying the graph-cut algorithm

		// Occupy the memory for the inlier arrays
		inliers.reserve(points_.rows);
		best_inliers.reserve(points_.rows);
		tmp_inliers.reserve(points_.rows);		
		models.reserve(_ModelEstimator::maximumMinimalSolutions());

		// Increase the number of the local optimizations applied
		++statistics.local_optimization_number;

		// The current sample used in the inner RANSAC
		std::unique_ptr<size_t[]> current_sample(new size_t[inlier_limit]);

		// Apply the graph-cut-based local optimization
		while (++statistics.graph_cut_number < settings.max_graph_cut_number)
		{
			// In the beginning, the best model is not updated
			updated = false;

			// Clear the inliers
			inliers.clear();

			// Apply the graph-cut-based inlier/outlier labeling.
			// The inlier set will contain the points closer than the threshold and
			// their neighbors depending on the weight of the spatial coherence term.
			labeling(
				points_, // The input points
				statistics.neighbor_number, // The number of neighbors, i.e. the edge number of the graph 
				neighbours, // The neighborhood graph
				best_model, // The best model parameters
				estimator_, // The model estimator
				settings.spatial_coherence_weight, // The weight of the spatial coherence term
				settings.threshold, // The inlier-outlier threshold
				inliers, // The selected inliers
				energy); // The energy after the procedure

			// Number of points (i.e. the sample size) used in the inner RANSAC
			const size_t sample_size = 
				static_cast<size_t>(MIN(inlier_limit, inliers.size()));

			// Run an inner RANSAC on the inliers coming from the graph-cut algorithm
			for (auto trial = 0; trial < trial_number_; ++trial)
			{
				// Reset the model vector
				models.clear();
				if (sample_size < inliers.size()) // If there are more inliers available than the minimum number, sample randomly.
				{
					sample<1>(inliers, // The inliers used for the selection
						sample_size, // The size of the minimal sample
						current_sample.get()); // The selected sample

					// Apply least-squares model fitting to the selected points.
					// If it fails, continue the for cycle and, thus, the sampling.
					if (!estimator_.estimateModelNonminimal(points_,  // The input data points
						current_sample.get(),  // The selected sample
						sample_size, // The size of the sample
						&models)) // The estimated model parameter
						continue;
				}
				else if (_ModelEstimator::sampleSize() < inliers.size()) // If there are enough inliers to estimate the model, use all of them
				{
					// Apply least-squares model fitting to the selected points.
					// If it fails, break the for cycle since we have used all inliers for this step.
					if (!estimator_.estimateModelNonminimal(points_, // The input data points
						&inliers[0],  // The selected sample
						inliers.size(), // The size of the sample
						&models)) // The estimated model parameter
						break;
				}
				else // Otherwise, break the for cycle.
					break;

				// Select the best model from the estimated set
				for (auto &model : models)
				{
					tmp_inliers.clear();

					// Calculate the score of the current model
					Score score = scoring_function->getScore(points_, // The input data points
						model, // The estimated model parameters
						estimator_, // The model estimator
						settings.threshold, // The inlier-outlier threshold
						tmp_inliers, // The inliers of the estimated model
						max_score, // The current best model
						true); // Flag saying that we do not need the inlier set

					// If this model is better than the previous best, update.
					if (max_score < score) // Comparing the so-far-the-best model's score and current model's score
					{
						updated = true; // Flag saying that we have updated the model parameters
						max_score = score; // Store the new best score
						best_model = model; // Store the new best model parameters
						best_inliers.swap(tmp_inliers);
						tmp_inliers.clear();
					}
				}
			}

			// If the model is not updated, interrupt the procedure
			if (!updated)
				break;
		}

		// If the new best score is better than the original one, update the model parameters.
		if (so_far_the_best_score_ < max_score) // Comparing the original best score and best score of the local optimization
		{
			so_far_the_best_score_ = max_score; // Store the new best score
			so_far_the_best_model_ = best_model;
			so_far_the_best_inliers_.swap(best_inliers);
			best_inliers.clear();
			return true;
		}
		return false;
	}

	template <class _ModelEstimator, class _NeighborhoodGraph, class _ScoringFunction, class _PreemptiveModelVerification, class _FastInlierSelector>
	void GCRANSAC<_ModelEstimator, _NeighborhoodGraph, _ScoringFunction, _PreemptiveModelVerification, _FastInlierSelector>::labeling(
		const cv::Mat &points_,
		size_t neighbor_number_,
		const std::vector<std::vector<cv::DMatch>> &neighbors_,
		Model &model_,
		_ModelEstimator estimator_,
		double lambda_,
		double threshold_,
		std::vector<size_t> &inliers_,
		double &energy_)
	{
		const int &point_number = points_.rows;

		// Initializing the problem graph for the graph-cut algorithm.
		Energy<double, double, double> *problem_graph =
			new Energy<double, double, double>(point_number, // The number of vertices
				neighbor_number_, // The number of edges
				NULL);

		// Add a vertex for each point
		for (auto i = 0; i < point_number; ++i)
			problem_graph->add_node();

		// The distance and energy for each point
		std::vector<double> distance_per_threshold;
		distance_per_threshold.reserve(point_number);
		double tmp_squared_distance,
			tmp_energy;
		const double squared_truncated_threshold = threshold_ * threshold_ * 9 / 4;
		const double one_minus_lambda = 1.0 - lambda_;

		// Estimate the vertex capacities
		for (size_t i = 0; i < point_number; ++i)
		{
			// Calculating the point-to-model squared residual
			tmp_squared_distance = estimator_.squaredResidual(points_.row(i),
				model_.descriptor);
			// Storing the residual divided by the squared threshold 
			distance_per_threshold.emplace_back(
				std::clamp(tmp_squared_distance / squared_truncated_threshold, 0.0, 1.0));
			// Calculating the implied unary energy
			tmp_energy = 1 - distance_per_threshold.back();

			// Adding the unary energy to the graph
			if (tmp_squared_distance <= squared_truncated_threshold)
				problem_graph->add_term1(i, one_minus_lambda * tmp_energy, 0);
			else
				problem_graph->add_term1(i, 0, one_minus_lambda * (1 - tmp_energy));
		}

		std::vector<std::vector<int>> used_edges(point_number, std::vector<int>(point_number, 0));

		if (lambda_ > 0)
		{
			double energy1, energy2, energy_sum;
			double e00, e11 = 0; // Unused: e01 = 1.0, e10 = 1.0,

			// Iterate through all points and set their edges
			for (auto point_idx = 0; point_idx < point_number; ++point_idx)
			{
				energy1 = distance_per_threshold[point_idx]; // Truncated quadratic cost

				// Iterate through  all neighbors
				for (const size_t &actual_neighbor_idx : neighborhood_graph->getNeighbors(point_idx))
				{
					if (actual_neighbor_idx == point_idx)
						continue;

					if (actual_neighbor_idx == point_idx || actual_neighbor_idx < 0)
						continue;

					if (used_edges[actual_neighbor_idx][point_idx] == 1 ||
						used_edges[point_idx][actual_neighbor_idx] == 1)
						continue;

					used_edges[actual_neighbor_idx][point_idx] = 1;
					used_edges[point_idx][actual_neighbor_idx] = 1;

					energy2 = distance_per_threshold[actual_neighbor_idx]; // Truncated quadratic cost
					energy_sum = energy1 + energy2;

					e00 = 0.5 * energy_sum;

					constexpr double e01_plus_e10 = 2.0; // e01 + e10 = 2
					if (e00 + e11 > e01_plus_e10)
						printf("Non-submodular expansion term detected; smooth costs must be a metric for expansion\n");

					problem_graph->add_term2(point_idx, // The current point's index
						actual_neighbor_idx, // The current neighbor's index
						e00 * lambda_,
						lambda_, // = e01 * lambda
						lambda_, // = e10 * lambda
						e11 * lambda_);
				}
			}
		}

		// Run the standard st-graph-cut algorithm
		problem_graph->minimize();

		// Select the inliers, i.e., the points labeled as SINK.
		inliers_.reserve(points_.rows);
		for (auto point_idx = 0; point_idx < points_.rows; ++point_idx)
			if (problem_graph->what_segment(point_idx) == Graph<double, double, double>::SINK)
				inliers_.emplace_back(point_idx);

		// Clean the memory
		delete problem_graph;
	}
}