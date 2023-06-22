#pragma once

#include <math.h>
#include <random>
#include <vector>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <Eigen/Eigen>
#include <glog/logging.h>

#include "GCoptimization.h"
#include "progx_model.h"

namespace pearl
{ 
	template<class _ModelEstimator> // The type of the model estimator used for estimating the model from a set of data points
	struct EnergyDataStructure
	{
		// All data points
		const cv::Mat &points; 
		// All model instances stored in the compound model
		const std::vector<progx::Model<_ModelEstimator>> * const model_instances;
		
		// The weight of the spatial coherence term. The terms are connected by linear interpolation.
		// Therefore Energy = (1 - spatial_coherence_weight) * Energy_{data} + spatial_coherence_weight * Energy_{spatial coherence}
		const double spatial_coherence_weight, 
			// 1.0 - spatial_coherence_weight
 			one_minus_spatial_coherence_weight, 
			// The inlier-outlier threshold
			inlier_outlier_threshold,
			// The squared inlier-outlier threshold
			squared_threshold, 
			// The truncated squared inlier-outlier threshold
			truncated_squared_threshold; 
		
		// The residuals of the point and model instances. This is required to speed up the procedure
		// by preventing the calculate the point-to-model distance multiple times.
		std::vector<std::vector<double>> squared_residuals;

		EnergyDataStructure(const cv::Mat &points_, // All data points
			const std::vector<progx::Model<_ModelEstimator>> * const model_instances_, // All model instances stored in the compound model
			const double spatial_coherence_weight_, // The weight of the spatial coherence term.
			const double inlier_outlier_threshold_) : // The inlier-outlier threshold
			points(points_),
			model_instances(model_instances_),
			spatial_coherence_weight(spatial_coherence_weight_),
			one_minus_spatial_coherence_weight(1.0 - spatial_coherence_weight_),
			inlier_outlier_threshold(inlier_outlier_threshold_),
			squared_threshold(inlier_outlier_threshold_ * inlier_outlier_threshold_),
			truncated_squared_threshold(9.0 / 4.0 * inlier_outlier_threshold_ * inlier_outlier_threshold_),
			squared_residuals(std::vector<std::vector<double>>(points_.rows, 
				std::vector<double>(model_instances_->size(), -1)))
		{
		}
	};

	template<class _ModelEstimator>
	inline double spatialCoherenceEnergyFunctor(
		int point_idx_1_, // The index of the edge's first point
		int point_idx_2_, // The index of the edge's second point
		int label_1_, // The label of the edge's first point
		int label_2_, // The label of the edge's second point
		void *information_object_) // The object containing all the necessary information
	{
		// Casting the information object to its original type
		EnergyDataStructure<_ModelEstimator> *information_object =
			reinterpret_cast<EnergyDataStructure<_ModelEstimator> *>(information_object_);

		// The weight of the spatial coherence term
		const double &spatial_coherence_weight = 
			information_object->spatial_coherence_weight;

		// If the labels are different, the implied energy is the weight.
		// Otherwise, there is no energy implied. 
		return label_1_ != label_2_ ? 
			spatial_coherence_weight : 
			0;
	}
	 
	template<class _ModelEstimator> // The type of the model estimator used for estimating the model from a set of data points
	inline double dataEnergyFunctor(
		int point_idx_, // The index of the current point
		int label_, // The label of the current point
		void *information_object_) // The object containing all the necessary information
	{
		// Casting the information object to its original type
		EnergyDataStructure<_ModelEstimator> *information_object =
			reinterpret_cast<EnergyDataStructure<_ModelEstimator> *>(information_object_);

		// Getting the truncated inlier-outlier threshold
		const double &truncated_squared_threshold = 
			information_object->truncated_squared_threshold;

		// Getting the <1 - weight of spatial coherence term> value. 
		const double &one_minus_spatial_coherence_weight = 
			information_object->one_minus_spatial_coherence_weight;

		// If the label is <model instance number> + 1, it assigns the point to the outlier instance
		if (label_ == information_object->model_instances->size())
			return one_minus_spatial_coherence_weight; 
		
		// Getting the assigned model instance
		const progx::Model<_ModelEstimator> &model_instance = 
			information_object->model_instances->at(label_);

		// Getting the point coordinates
		const cv::Mat &point = information_object->points.row(point_idx_);

		// Get the stored squared residual value
		double &stored_squared_residual =
			information_object->squared_residuals[point_idx_][label_];

		double squared_residual;
		// If the current residual has not been yet calculated, do it.
		if (stored_squared_residual == -1) 
			stored_squared_residual = model_instance.estimator->squaredResidual(point, model_instance);
		squared_residual = stored_squared_residual;
					
		// If the residual if higher than the threshold and the current model instance is
		// not the outlier instance (checked earlier), return an energy higher than 
		// that of assigning the outlier instance. 
		if (squared_residual > truncated_squared_threshold)
			return 2.0 * one_minus_spatial_coherence_weight;

		return one_minus_spatial_coherence_weight * // (1 - spatial_coherence_weight), i.e. the other part of the linear interpolation
			squared_residual / truncated_squared_threshold; // A value in-between [0,1].
	}

	template<class _NeighborhoodGraph, // The type of the neighborhood graph which is used in the spatial coherence term
		class _ModelEstimator> // The type of the model estimator used for estimating the model from a set of data points
	class PEARL
	{
	public:
		PEARL(double inlier_outlier_threshold_,
			double spatial_coherence_weight_,
			size_t minimum_inlier_number_,
			std::vector<double> point_weights_,
			const size_t maximum_iteration_number_ = 100,
			const bool do_logging_ = false) :
			maximum_iteration_number(maximum_iteration_number_),
			inlier_outlier_threshold(inlier_outlier_threshold_),
			spatial_coherence_weight(spatial_coherence_weight_),
			model_complexity_weight(minimum_inlier_number_),
			epsilon(1e-5),
			minimum_inlier_number(minimum_inlier_number_),
			alpha_expansion_engine(nullptr),
			do_logging(do_logging_),
			point_weights(point_weights_)
		{
		}

		~PEARL()
		{
			if (alpha_expansion_engine != nullptr)
				delete alpha_expansion_engine;
		}

		bool run(const cv::Mat &data_,
			const _NeighborhoodGraph *neighborhood_graph_,
			_ModelEstimator *model_estimator_,
			std::vector<progx::Model<_ModelEstimator>> *models_);
		
		const std::pair<const std::vector<size_t> *, size_t> getLabeling();
		void getLabeling(std::vector<size_t> &labeling_, 
			size_t &instance_number_);

		size_t getOutlierNumber() { return outliers.size(); }
		
	protected:
		// The alpha-expansion engine used to obtain a labeling
		GCoptimizationGeneralGraph *alpha_expansion_engine;
		// The weights used for the non-minimal fitting
		std::vector<double> point_weights;

		// The weight of the spatial coherence term
		double spatial_coherence_weight,
			// The model complexity weigth to reject insignificant model instances
			model_complexity_weight,
			// The inlier-outlier threshold
			inlier_outlier_threshold,
			// The stopping criterion threshold for the iteration
			epsilon;

		// Flag determining if logging is required
		bool do_logging;

		// The number of points
		size_t point_number,
			// The minimum inlier number to accept a model
			minimum_inlier_number,
			// The maximum number of iterations
			maximum_iteration_number;

		std::vector<size_t> point_to_instance_labeling;
		std::vector<std::vector<size_t>> points_per_instance;
		std::vector<size_t> outliers;

		bool labeling(const cv::Mat &data_, // All data points
			const std::vector<progx::Model<_ModelEstimator>> *models_, // The models instances stored
			const _ModelEstimator *model_estimator_, // The model estimator used for estimating the model from a set of data points
			const _NeighborhoodGraph *neighborhood_graph_, // The neighborhood graph which is used in the spatial coherence term
			const bool initialize_with_previous_labeling_, // A flag indicating if the labeling should be initialized using the previous one. 
			double &energy_); // The energy of the labeling

		void parameterEstimation(const cv::Mat &data_,
			std::vector<progx::Model<_ModelEstimator>> *models_,
			const _ModelEstimator *model_estimator_,
			bool &changed_);

		void rejectInstances(const cv::Mat &data_,
			std::vector<progx::Model<_ModelEstimator>> * models_,
			bool &changed_);
	};

	template<class _NeighborhoodGraph, // The type of the neighborhood graph which is used in the spatial coherence term
		class _ModelEstimator> // The type of the model estimator used for estimating the model from a set of data points
	void PEARL<_NeighborhoodGraph, _ModelEstimator>::getLabeling(std::vector<size_t> &labeling_, // The labeling
		size_t &instance_number_) // The number of instances used on the labeling
	{
		// Resize the containers
		point_to_instance_labeling.resize(point_number);
		labeling_.resize(point_number);
		
		// If the alpha-expansion engine has not been initialized yet, return the labeling as
		// if all points are outliers.
		if (alpha_expansion_engine == nullptr)
		{
			std::fill(point_to_instance_labeling.begin(), point_to_instance_labeling.end(), 0);
			std::fill(labeling_.begin(), labeling_.end(), 0);
			instance_number_ = 0; 
			return;
		}

		// Initialize the instance number to zero to prevent having
		// some memory garbage in it.
		instance_number_ = 0;

		// Iterate through the points and save the obtained labels.
		for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
		{
			const size_t &label = alpha_expansion_engine->whatLabel(point_idx); // The label of the current point
			point_to_instance_labeling[point_idx] = label; // Set the element in the local container
			labeling_[point_idx] = label; // Set the element in the container
			instance_number_ = MAX(instance_number_, label); // Update the biggest label.
		}
	}

	template<class _NeighborhoodGraph, // The type of neighborhood graph which is used in the spatial coherence term
		class _ModelEstimator> // The type of the model estimator used for estimating the model from a set of data points
	const std::pair<const std::vector<size_t> *, size_t> PEARL<_NeighborhoodGraph, _ModelEstimator>::getLabeling()	 
	{
		// Resizing the container of the labeling
		point_to_instance_labeling.resize(point_number, 0);
		// If the alpha-expansion engine has not been initialized yet, return the labeling as
		// if all points are outliers.
		if (alpha_expansion_engine == nullptr)
			return std::make_pair(&point_to_instance_labeling, 1);
				
		size_t max_label = 0; // Store the label number, i.e., the number of model instances used in the labeling
		// Iterate through the points and save the obtained labels.
		for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
		{			
			const size_t &label = alpha_expansion_engine->whatLabel(point_idx); // The label of the current point
			point_to_instance_labeling[point_idx] = label; // Set the element in the container
			max_label = MAX(max_label, label); // Update the biggest label.
		}

		// Return a constant reference to the label container together with the distinct label number in it.
		return std::make_pair(&point_to_instance_labeling, max_label);
	}

	template<class _NeighborhoodGraph, // The type of neighborhood graph which is used in the spatial coherence term
		class _ModelEstimator> // The type of the model estimator used for estimating the model from a set of data points
	void PEARL<_NeighborhoodGraph, _ModelEstimator>::rejectInstances(
		const cv::Mat &data_,
		std::vector<progx::Model<_ModelEstimator>> * models_,
		bool &changed_)
	{
		// Remove instances which do not have enough inliers.
		// Starting from the last one to avoid updating the index after case of removal. 
		for (int instance_idx = models_->size() - 1; instance_idx >= 0; --instance_idx)
		{
			// The inliers of the current model instance
			const std::vector<size_t> &current_inliers =
				points_per_instance[instance_idx];
			// The number of inliers of the current model instance
			const size_t inlier_number = current_inliers.size();
			
			// If the inlier number of lower than the minimum, 
			// reject the instance.
			if (inlier_number < minimum_inlier_number)
			{
				// Add the instance's inliers to the outliers
				outliers.insert(outliers.end(), current_inliers.begin(), current_inliers.end());

				// Remove the container consisting of the instance's inliers
				// from the main one.
				points_per_instance.erase(points_per_instance.begin() + instance_idx);
				
				// Remove the model parameters from the main container.
				models_->erase(models_->begin() + instance_idx);

				// Change the flag to show that something has changed and, therefore,
				// the labeling should be applied again.
				changed_ = true;

				if (do_logging)
					std::cout << "[Optimization] Instance " << instance_idx << 
						" is rejected due to having too few inliers (" << inlier_number  << ").\n";
			}
		}


	}

	template<class _NeighborhoodGraph, // The type of neighborhood graph which is used in the spatial coherence term
		class _ModelEstimator> // The type of the model estimator used for estimating the model from a set of data points
	void PEARL<_NeighborhoodGraph, _ModelEstimator>::parameterEstimation(
		const cv::Mat &data_,
		std::vector<progx::Model<_ModelEstimator>> *models_,
		const _ModelEstimator *model_estimator_,
		bool &changed_)
	{
		// If the alpha-expansion engine has not been initialized, return.
		if (alpha_expansion_engine == nullptr)
		{
			if (do_logging)
				LOG(WARNING) << "The alpha-expansion engine has not been initialized.\n";
			return;
		}

		// The number of currently stored instances
		const size_t instance_number = models_->size();

		// Clear the vectors containing the inliers of each model instance and the outlier instance
		points_per_instance.clear();
		points_per_instance.resize(instance_number);
		outliers.clear();

		// Get the assigned points for each model instance
		for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
		{
			// The label of the current point
			const size_t &label = alpha_expansion_engine->whatLabel(point_idx);

			// If the labeling is lower than the instance number, it points to an existing model instance.
			if (label < instance_number)
				points_per_instance[label].emplace_back(point_idx);
			else // Otherwise, it indicates that the point is assigned to the outlier instance.
				outliers.emplace_back(point_idx);
		}

		// Estimate the model parameters using all the assigned points
		for (size_t instance_idx = 0; instance_idx < instance_number; ++instance_idx)
		{
			// The current model parameters
			const std::vector<size_t> &current_inliers = 
				points_per_instance[instance_idx];
			// The number of inliers assigned to the current model
			const size_t inlier_number = current_inliers.size();

			// If there are not enough points to re-estimate the model parameters,
			// continue to the next instance.
			if (inlier_number < _ModelEstimator::nonMinimalSampleSize())
				continue;

			// Calculate the sum of residuals before the model fitting to all inliers
			double sum_of_residuals_before = 0.0;
			for (const size_t &inlier_idx : current_inliers)
				sum_of_residuals_before += model_estimator_->residual(data_.row(inlier_idx), models_->at(instance_idx));

			// Fit a model to all of the inliers
			std::vector<gcransac::Model> current_models;
			model_estimator_->estimateModelNonminimal(data_, // All data points
				&current_inliers[0], // The indices of the inliers
				inlier_number, // The number of inliers
				&current_models, // The estimated model parameters
				point_weights.size() == 0 ? 
					nullptr : &point_weights[0]); // The weights used for the non-minimal fitting

			// If there are fewer or more models estimated than a single one,
			// continue to the next instance.
			if (current_models.size() != 1)
				continue;

			// Calculate the sum of residuals of the fit model on all inliers
			double sum_of_residuals_after = 0.0;
			for (const size_t &inlier_idx : current_inliers)
				sum_of_residuals_after += model_estimator_->residual(data_.row(inlier_idx), current_models.back());

			// If the sum of residuals is higher after the refitting, do not use the model.
			if (sum_of_residuals_after < sum_of_residuals_before)
			{
				// Setting the descriptor of the new model.
				models_->at(instance_idx).setDescriptor(current_models.back().descriptor); 
				// Something has been changed, thus, the labeling should run again.
				changed_ = true; 
			}
		}
	}

	template<class _NeighborhoodGraph, // The type of neighborhood graph which is used in the spatial coherence term
		class _ModelEstimator> // The type of the model estimator used for estimating the model from a set of data points
	bool PEARL<_NeighborhoodGraph, _ModelEstimator>::run(
		const cv::Mat &data_,
		const _NeighborhoodGraph *neighborhood_graph_,
		_ModelEstimator *model_estimator_,
		std::vector<progx::Model<_ModelEstimator>> *models_)
	{
		point_number = data_.rows; // The number of points
		size_t iteration_number = 0, // The number of current iterations
			iteration_number_without_change; // The number of consecutive iterations when nothing has changed.
		double energy = std::numeric_limits<double>::max(),  // The energy of the alpha-expansion
			previous_energy = -1.0; // The energy of the previous alpha-expansion
		bool model_parameters_changed = false, // A flag to see of the model parameters changed after re-fitting
			model_rejected = false, // A flag to see if any model has been rejected 
			convergenve = false; // A flag to see if the algorithm has converges

		// The main PEARL iteration doing the labeling and model refitting iteratively.
		while (!convergenve && // Break when the results converged.
			iteration_number++ < maximum_iteration_number) // Break when the maximum iteration limit has been exceeded.
		{
			if (do_logging)
				std::cout << "[Optimization] Iteration " << iteration_number << ".\n";

			// A flag to decide if the previous labeling should be used as an initial one.
			// It is true if it is not the first iteration and the number of models has not been changed. 
			bool initialize_with_previous_labeling =
				iteration_number > 1 && 
				!model_rejected;

			// Apply alpha-expansion to get the labeling which assigns each point to a model instance
			labeling(data_, // All data points
				models_,
				model_estimator_, // The model estimator
				neighborhood_graph_, // The neighborhood graph
				initialize_with_previous_labeling, // A flag to decide if the previous labeling should be used as an initial one.
				energy); // The energy of the alpha-expansion

			if (do_logging)
				std::cout << "[Optimization] The energy of the labeling is " << energy << ".\n";

			// A flag to see if the model parameters changed.
			// Initialize as if nothing has changed.
			model_parameters_changed = false;

			// A flag to see if a model has been rejected.
			// Initialize as if nothing has changed.
			model_rejected = false;
			
			// Re-estimate the model parameters based on the determined labeling
			parameterEstimation(data_, // All data points
				models_, // The currently stored models, i.e., the compound model
				model_estimator_, // The model estimator used for estimating the model parameters from a set of data points
				model_parameters_changed); // A flag to see if anything has changed

			rejectInstances(data_, // All data points
				models_, // The currently stored models, i.e., the compound model
				model_rejected); // A flag to see if anything has changed

			// If nothing has changed, terminate
			if (!model_rejected &&
				!model_parameters_changed &&
				abs(energy - previous_energy) < epsilon &&
				iteration_number > 1)
				convergenve = true;

			previous_energy = energy;
		}
		return true;
	}

	template<class _NeighborhoodGraph, // The type of neighborhood graph which is used in the spatial coherence term
		class _ModelEstimator> // The type of the model estimator used for estimating the model from a set of data points
		bool PEARL<_NeighborhoodGraph, _ModelEstimator>::labeling(
			const cv::Mat &data_,
			const std::vector<progx::Model<_ModelEstimator>> *models_,
			const _ModelEstimator *model_estimator_,
			const _NeighborhoodGraph *neighborhood_graph_,
			const bool initialize_with_previous_labeling_,
			double &energy_)
	{
		//std::cout << 821 << std::endl;
		// Return if there are no model instances
		if (models_->size() == 0)
			return false;

		// Set the previous labeling if nothing has changed
		std::vector<size_t> previous_labeling;
		if (initialize_with_previous_labeling_ &&
			alpha_expansion_engine != nullptr)
		{
			previous_labeling.resize(point_number);
			for (size_t i = 0; i < point_number; ++i)
				previous_labeling[i] = alpha_expansion_engine->whatLabel(i);
		}

		// Delete the alpha-expansion engine if it has been used.
		// The graph provided by the GCOptimization library is 
		// not reusable.
		if (alpha_expansion_engine != nullptr)
			delete alpha_expansion_engine;

		// Initializing the alpha-expansion engine with the given number of points and
		// with model number plus one labels. The plus labels is for the outlier class.		
		alpha_expansion_engine =
			new GCoptimizationGeneralGraph(static_cast<int>(point_number), static_cast<int>(models_->size()) + 1);

		//std::cout << 822 << std::endl;
		// The object consisting of all information required for the energy calculations
		EnergyDataStructure<_ModelEstimator> information_object(
			data_, // The data points
			models_, // The model instances represented by the labels
			spatial_coherence_weight, // The weight of the spatial coherence term
			inlier_outlier_threshold); // The inlier-outlier threshold used when assigning points to model instances

		// Set the data cost functor to the alpha-expansion engine
		alpha_expansion_engine->setDataCost(&dataEnergyFunctor<_ModelEstimator>, // The data cost functor
			&information_object); // The object consisting of all information required for the energy calculations

		// Set the spatial cost functor to the alpha-expansion engine if needed
		if (spatial_coherence_weight > 0.0)
			alpha_expansion_engine->setSmoothCost(&spatialCoherenceEnergyFunctor<_ModelEstimator>, // The spatial cost functor
				&information_object); // The object consisting of all information required for the energy calculations

		// Set the model complexity weight to the alpha-expansion engine if needed
		if (model_complexity_weight > 0.0)
			alpha_expansion_engine->setLabelCost(model_complexity_weight);

		// Set neighbourhood of each point
		if (spatial_coherence_weight > 0.0)
			for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
				for (const size_t &neighbor_idx : neighborhood_graph_->getNeighbors(point_idx))
					if (point_idx != neighbor_idx)
						alpha_expansion_engine->setNeighbors(point_idx, neighbor_idx);
		
		//std::cout << 823 << " " << previous_labeling.size() << std::endl;
		// If nothing has changed since the previous labeling, use
		// the previous labels as initial values.
		if (initialize_with_previous_labeling_ &&
			previous_labeling.size() > 0)
		{
			for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
				alpha_expansion_engine->setLabel(point_idx, previous_labeling[point_idx]);
			previous_labeling.clear();
		}

		int iteration_number;
		energy_ = alpha_expansion_engine->expansion(iteration_number, 
			1000);
		//std::cout << 824 << std::endl;
		
		return true;
	}
}
