#pragma once

#include "scoring_function.h"
#include "progx_model.h"

namespace progx
{
	template<class _ModelEstimator>
	class MSACScoringFunctionWithCompoundModel : public gcransac::ScoringFunction<_ModelEstimator>
	{
	protected:
		// Squared truncated threshold
		double squared_truncated_threshold;

		// Number of points
		size_t point_number; 

		// The exponent of the score shared with the compound model instance and
		// substracted from the final score.  
		int exponent_of_shared_score;

		// The pointer of the compound model instance
		const std::vector<Model<_ModelEstimator>> *compound_model;

		// The pointer of the preference vector of the compound model instance
		const Eigen::VectorXd *compound_preference_vector;

	public:
		MSACScoringFunctionWithCompoundModel() : exponent_of_shared_score(2.5)
		{

		}

		~MSACScoringFunctionWithCompoundModel()
		{

		}

		void setExponent(const int exponent_of_shared_score_)
		{
			exponent_of_shared_score = exponent_of_shared_score_;
		}

		void setCompoundModel(
			const std::vector<Model<_ModelEstimator>> *compound_model_, // The pointer of the compound model instance
			const Eigen::VectorXd *compound_preference_vector_) // The pointer of the preference vector of the compound model instance
		{
			compound_preference_vector = compound_preference_vector_;
			compound_model = compound_model_;
		}

		void initialize(
			const double squared_truncated_threshold_, // Squared truncated threshold
			const size_t point_number_) // Number of points
		{
			squared_truncated_threshold = squared_truncated_threshold_;
			point_number = point_number_;
		}

		// Return the score of a model w.r.t. the data points and the threshold
		inline gcransac::Score getScore(
			const cv::Mat &points_, // The input data points
			gcransac::Model &model_, // The current model parameters
			const _ModelEstimator &estimator_, // The model estimator
			const double threshold_, // The inlier-outlier threshold
			std::vector<size_t> &inliers_, // The selected inliers
			const gcransac::Score &best_score_ = gcransac::Score(), // The score of the current so-far-the-best model
			const bool store_inliers_ = true, // A flag to decide if the inliers should be stored
			const std::vector<const std::vector<size_t>*> *index_sets = nullptr) const // Index sets to be verified
		{
			gcransac::Score score; // The current score
			if (store_inliers_) // If the inlier should be stored, clear the variables
				inliers_.clear();
			double squared_residual, score_value; // The point-to-model residual
			Eigen::MatrixXd preference_vector = Eigen::MatrixXd::Zero(point_number, 1); // Initializing the preference vector

			// Iterate through all points, calculate the squared_residuals and store the points as inliers if needed.
			for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
			{
				// Calculate the point-to-model residual
				squared_residual = estimator_.squaredResidual(points_.row(point_idx), model_.descriptor);

				// If the residual is smaller than the threshold, store it as an inlier and
				// increase the score.
				if (squared_residual < squared_truncated_threshold)
				{
					if (store_inliers_) // Store the point as an inlier if needed.
						inliers_.emplace_back(point_idx);

					// Increase the inlier number
					++(score.inlier_number);

					// Calculate the score (coming from the truncated quadratic loss) implied by the current point
					score_value = MAX(0.0, 1.0 - squared_residual / squared_truncated_threshold);

					// Increase the score
					score.value += score_value; 

					// The preference value. It is proportional to the likelihood. 
					preference_vector(point_idx) = score_value;
						
				}

				// Interrupt if there is no chance of being better than the best model
				if (point_number - point_idx + score.inlier_number < best_score_.inlier_number)
					return gcransac::Score();
			}

			// Calculating the support shared with the compound model
			if (compound_model->size() > 0)
			{
				double shared_support = 0; // The shared support

				// Iterate through all points and calculate the shared support
				for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
					shared_support += 
						MIN((*compound_preference_vector)(point_idx), preference_vector(point_idx));

				// Substract the shared support from the score of the putative model
				score.value -= std::pow(shared_support, exponent_of_shared_score);
			}

			// Return the final score
			return score;
		}
	};
}