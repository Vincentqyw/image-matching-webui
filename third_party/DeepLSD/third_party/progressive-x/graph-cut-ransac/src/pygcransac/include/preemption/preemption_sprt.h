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

#include "model.h"
#include <opencv2/core.hpp>
#include <Eigen/Eigen>
#include <random>
#include <algorithm>
#include <iomanip>

#define LOG_ETA_0 log(0.05)

namespace gcransac
{
	namespace preemption
	{
		template <typename _ModelEstimator>
		class SPRTPreemptiveVerfication
		{
		protected:
			class SPRTHistory {
			public:
				double epsilon, delta, A;
				// k is number of samples processed by test
				size_t k;
			};

			/*
			* The probability of a data point being consistent
			* with a ‘bad’ model is modeled as a probability of
			* a random event with Bernoulli distribution with parameter
			* δ : p(1|Hb) = δ.
			*/

			/*
			 * The probability p(1|Hg) = ε
			 * that any randomly chosen data point is consistent with a ‘good’ model
			 * is approximated by the fraction of inliers ε among the data
			 * points
			 */

			 /*
			  * The decision threshold A is the only parameter of the Adapted SPRT
			  */
			  // i
			size_t current_sprt_idx;
			int last_sprt_update;

			double t_M, m_S, threshold, confidence;
			size_t points_size, sample_size, max_iterations, random_pool_idx;
			std::vector<SPRTHistory> sprt_histories;
			
			int number_rejected_models;
			int sum_fraction_data_points;
			size_t * points_random_pool;

			int max_hypothesis_test_before_sprt;

		public:
			double additional_model_probability;

			static constexpr bool providesScore() { return true; }
			static constexpr const char *getName() { return "SPRT"; }

			~SPRTPreemptiveVerfication() {
				delete[] points_random_pool;
			}

			// This function is called only once to calculate the exact model estimation time of 
			// the current model on the current machine. It is required for SPRT to really
			// speed up the verification procedure. 
			void initialize(const cv::Mat &points_,
				const _ModelEstimator &estimator_)
			{
				size_t sample[_ModelEstimator::sampleSize()];
				std::iota(sample, sample + _ModelEstimator::sampleSize(), 0);
				for (size_t sampleIdx = 0; sampleIdx < _ModelEstimator::sampleSize(); ++sampleIdx)
					sample[sampleIdx] = sampleIdx;

				std::vector<Model> models;

				std::chrono::time_point<std::chrono::system_clock> end,
					start = std::chrono::system_clock::now();
				estimator_.estimateModel(
					points_,
					sample,
					&models);
				end = std::chrono::system_clock::now();

				std::chrono::duration<double> elapsedSeconds = end - start;
				t_M = elapsedSeconds.count() * 1000.0; // The time of estimating a model

				if (models.size() > 0)
				{
					start = std::chrono::system_clock::now();
					for (size_t sampleIdx = 0; sampleIdx < _ModelEstimator::sampleSize(); ++sampleIdx)
						estimator_.residual(points_.row(sampleIdx), models[0]);
					end = std::chrono::system_clock::now();
					t_M = (elapsedSeconds.count() * 1000.0 / _ModelEstimator::sampleSize()) / t_M; // The time of estimating a model
				}

				m_S = _ModelEstimator::maximumMinimalSolutions(); // The maximum number of solutions

				/*printf("Setting up SPRT test.\n");
				printf("\tThe estimation of one models takes %f ms.\n", t_M);
				printf("\tAt most %.0f models are returned.\n", m_S);*/
			}

			SPRTPreemptiveVerfication(const cv::Mat &points_,
				const _ModelEstimator &estimator_,
				const double &minimum_inlier_ratio_ = 0.1)
			{
				if (points_.rows < _ModelEstimator::sampleSize())
				{
					fprintf(stderr, "There are not enough points to initialize the SPRT test (%d < %d).\n",
						points_.rows, _ModelEstimator::sampleSize());
					return;
				}

				initialize(points_, estimator_);

				additional_model_probability = 1.0;
				const size_t point_number = points_.rows;

				// Generate array of points
				points_random_pool = new size_t[point_number];
				for (size_t i = 0; i < point_number; i++) {
					points_random_pool[i] = i;
				}
				unsigned int temp;
				int max = point_number;
				// Shuffle random pool of points.
				for (unsigned int i = 0; i < point_number; i++) {
					random_pool_idx = rand() % max;
					temp = points_random_pool[random_pool_idx];
					max--;
					points_random_pool[random_pool_idx] = points_random_pool[max];
					points_random_pool[max] = temp;
				}
				random_pool_idx = 0;

				sprt_histories = std::vector<SPRTHistory>();
				sprt_histories.emplace_back(SPRTHistory());

				sprt_histories.back().delta = 0.01;
				sprt_histories.back().epsilon = minimum_inlier_ratio_;

				current_sprt_idx = 0;
				last_sprt_update = 0;

				sprt_histories.back().A = estimateThresholdA(sprt_histories.back().epsilon, sprt_histories.back().delta);
				sprt_histories.back().k = 0;

				number_rejected_models = 0;
				sum_fraction_data_points = 0;

				max_hypothesis_test_before_sprt = 20;
			}

			/*
			 *                      p(x(r)|Hb)                  p(x(j)|Hb)
			 * lambda(j) = Product (----------) = lambda(j-1) * ----------
			 *                      p(x(r)|Hg)                  p(x(j)|Hg)
			 * Set j = 1
			 * 1.  Check whether j-th data point is consistent with the
			 * model
			 * 2.  Compute the likelihood ratio λj eq. (1)
			 * 3.  If λj >  A, decide the model is ’bad’ (model ”re-jected”),
			 * else increment j or continue testing
			 * 4.  If j = N the number of correspondences decide model ”accepted
			 *
			 * Verifies model and returns model score.
			 */
			bool verifyModel(const gcransac::Model &model_,
				const _ModelEstimator &estimator_, // The model estimator
				const double &threshold_,
				const size_t &iteration_number_,
				const Score &best_score_,
				const cv::Mat &points_,
				const size_t *minimal_sample_,
				const size_t sample_number_,
				std::vector<size_t> &inliers_,
				Score &score_,
				const std::vector<const std::vector<size_t>*> *index_sets_ = nullptr)
			{
				inliers_.clear();
				const size_t &point_number = points_.rows;
				const double squared_threshold = threshold_ * threshold_;
				const double &epsilon = sprt_histories[current_sprt_idx].epsilon;
				const double &delta = sprt_histories[current_sprt_idx].delta;
				const double &A = sprt_histories[current_sprt_idx].A;

				double lambda_new, lambda = 1;
				size_t tested_point = 0, tested_inliers = 0;
				score_ = Score();

				bool valid_model = true;
				if (index_sets_ == nullptr)
				{
					for (tested_point = 0; tested_point < point_number; tested_point++)
					{
						const size_t &point_idx = 
							points_random_pool[random_pool_idx];
						const double squared_residual = 
							estimator_.squaredResidual(points_.row(point_idx), model_);

						// Inliers 
						if (squared_residual < squared_threshold) {
							lambda_new = lambda * (delta / epsilon);

							// Increase the inlier number
							++(score_.inlier_number);
							// Increase the score. The original truncated quadratic loss is as follows: 
							// 1 - residual^2 / threshold^2. For RANSAC, -residual^2 is enough.
							score_.value += 1.0 - squared_residual / squared_threshold; // Truncated quadratic cost

							inliers_.emplace_back(point_idx);
						}
						else {
							lambda_new = lambda * ((1 - delta) / (1 - epsilon));
						}

						// Increase the pool pointer and reset if needed
						if (++random_pool_idx >= point_number)
							random_pool_idx = 0;

						if (lambda_new > A * additional_model_probability) {
							valid_model = false;
							++tested_point;
							break;
						}

						lambda = lambda_new;
					}
				} else
				{
					// Iterating through the index sets
					for (const auto &current_set : *index_sets_)
						// Iterating through the point indices in the current set
						for (const auto point_idx : *current_set)
						{
							const double squared_residual = 
								estimator_.squaredResidual(points_.row(point_idx), model_);

							// Inliers 
							if (squared_residual < squared_threshold) {
								lambda_new = lambda * (delta / epsilon);

								// Increase the inlier number
								++(score_.inlier_number);
								// Increase the score. The original truncated quadratic loss is as follows: 
								// 1 - residual^2 / threshold^2. For RANSAC, -residual^2 is enough.
								score_.value += 1.0 - squared_residual / squared_threshold; // Truncated quadratic cost

								inliers_.emplace_back(point_idx);
							}
							else {
								lambda_new = lambda * ((1 - delta) / (1 - epsilon));
							}

							if (lambda_new > A * additional_model_probability) {
								valid_model = false;
								break;
							}

							lambda = lambda_new;
						}
				}

				if (valid_model)
				{
					/*
						* Model accepted and the largest support so far:
						* design (i+1)-th test (εi + 1= εˆ, δi+1 = δˆ, i = i + 1).
						* Store the current model parameters θ
					*/
					if (score_.value > best_score_.value) {
						SPRTHistory new_sprt_history;

						new_sprt_history.epsilon = (double)score_.inlier_number / point_number;

						new_sprt_history.delta = delta;
						new_sprt_history.A = estimateThresholdA(new_sprt_history.epsilon, delta);
						
						new_sprt_history.k = iteration_number_ - last_sprt_update;
						last_sprt_update = iteration_number_;
						++current_sprt_idx;
						sprt_histories.emplace_back(new_sprt_history);
					}
				}
				else
				{
					/*
						* Since almost all tested models are ‘bad’, the probability
						* δ can be estimated as the average fraction of consistent data points
						* in rejected models.
					*/
					double delta_estimated = (double)score_.inlier_number / point_number;

					if (delta_estimated > 0 && fabs(delta - delta_estimated) / delta > 0.05) 
					{
						SPRTHistory new_sprt_history;

						new_sprt_history.epsilon = epsilon;
						new_sprt_history.delta = delta_estimated;
						new_sprt_history.A = estimateThresholdA(epsilon, delta_estimated);
						new_sprt_history.k = iteration_number_ - last_sprt_update;
						last_sprt_update = iteration_number_;
						current_sprt_idx++;
						sprt_histories.emplace_back(new_sprt_history);
					}
				}

				return valid_model;
			}

			/*
			* A(0) = K1/K2 + 1
			* A(n+1) = K1/K2 + 1 + log (A(n))
			* K1 = t_M / P_g
			* K2 = m_S/(P_g*C)
			* t_M is time needed to instantiate a model hypotheses given a sample
			* P_g = epsilon ^ m, m is the number of data point in the Ransac sample.
			* m_S is the number of models that are verified per sample.
			*                   p (0|Hb)                  p (1|Hb)
			* C = p(0|Hb) log (---------) + p(1|Hb) log (---------)
			*                   p (0|Hg)                  p (1|Hg)
			*/
			double estimateThresholdA(
				const double &epsilon, 
				const double &delta)
			{
				const double C = (1 - delta) * log((1 - delta) / (1 - epsilon)) + delta * (log(delta / epsilon));
				const double K = (t_M * C) / m_S + 1;
				double An_1 = K;
				double An;
				for (unsigned int i = 0; i < 10; ++i) {
					An = K + log(An_1);

					if (fabs(An - An_1) < 1.5e-8) {
						break;
					}
					An_1 = An;
				}

				return An;
			}
		};
	}
}