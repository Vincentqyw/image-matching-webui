// Copyright (c) 2019, Torsten Sattler
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// author: Torsten Sattler, torsten.sattler.de@googlemail.com

#ifndef RANSACLIB_RANSACLIB_HYBRID_RANSAC_H_
#define RANSACLIB_RANSACLIB_HYBRID_RANSAC_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <RansacLib/hybrid_sampling.h>
#include <RansacLib/utils.h>

namespace ransac_lib {

class HybridRansacOptions {
 public:
  HybridRansacOptions()
      : min_num_iterations_(100u),
        max_num_iterations_(10000u),
        max_num_iterations_per_solver_(10000u),
        success_probability_(0.9999),
        random_seed_(0u) {
    squared_inlier_thresholds_.clear();
  }
  uint32_t min_num_iterations_;
  uint32_t max_num_iterations_;
  uint32_t max_num_iterations_per_solver_;
  double success_probability_;
  std::vector<double> squared_inlier_thresholds_;
  // An importance weight for each data type. Higher weight means higher
  // importance.
  std::vector<double> data_type_weights_;
  unsigned int random_seed_;
};

// See Lebeda et al., Fixing the Locally Optimized RANSAC, BMVC, Table 1 for
// details on the variables.
class HybridLORansacOptions : public HybridRansacOptions {
 public:
  HybridLORansacOptions()
      : num_lo_steps_(10),
        threshold_multiplier_(std::sqrt(2.0)),
        num_lsq_iterations_(4),
        min_sample_multiplicator_(7),
        lo_starting_iterations_(50u),
        final_least_squares_(false) {}
  int num_lo_steps_;
  double threshold_multiplier_;
  int num_lsq_iterations_;
  // The maximum number of data points used for least squares refinement is
  // min_sample_multiplicator_ * min_sample_size[solver_idx], where solver_idx
  // is the index of the solver used in the current iteration. Lebeda et al.
  // recommend setting min_sample_multiplicator_ to 7 (empirically determined
  // for epipolar geometry estimation.
  int min_sample_multiplicator_;
  // As suggested in Sec. 4.4 in Lebeda et al., Local Optimization is only
  // performed after the first K_start iterations (set to 50 by Lebeda et al.)
  // to reduce overhead.
  uint32_t lo_starting_iterations_;
  bool final_least_squares_;
};

struct HybridRansacStatistics {
  uint32_t num_iterations_total;
  std::vector<uint32_t> num_iterations_per_solver;
  int best_num_inliers;
  int best_solver_type;  // The type of solver used to create the best model.
  double best_model_score;
  std::vector<double> inlier_ratios;
  std::vector<std::vector<int>> inlier_indices;
  int number_lo_iterations;
};

class HybridRansacBase {
 protected:
  void ResetStatistics(HybridRansacStatistics* statistics) const {
    HybridRansacStatistics& stats = *statistics;
    stats.best_num_inliers = 0;
    stats.best_model_score = std::numeric_limits<double>::max();
    stats.num_iterations_total = 0u;
    stats.num_iterations_per_solver.clear();
    stats.inlier_ratios.clear();
    stats.inlier_indices.clear();
    stats.number_lo_iterations = 0;
    stats.best_solver_type = -1;
  }
};

// Implements Hybrid LO-RANSAC with MSAC (top-hat) scoring, based on the
// descriptions provided in [Camposeco, Cohen, Pollefeys, Sattler, Hybrid Camera
// Pose Estimation, CVPR 2018] and [Lebeda, Matas, Chum, Fixing the Locally
// Optimized RANSAC, BMVC 2012]. Iteratively re-weighted least-squares
// optimization is optional.
template <class Model, class ModelVector, class HybridSolver,
          class Sampler = HybridUniformSampling<HybridSolver> >
class HybridLocallyOptimizedMSAC : public HybridRansacBase {
 public:
  // Estimates a model using a given solver. Notice that the solver contains
  // all data and is responsible to implement a non-minimal solver and
  // least-squares refinement. The latter two are optional, i.e., a dummy
  // implementation returning false is sufficient.
  // Returns the number of inliers.
  int EstimateModel(const HybridLORansacOptions& options,
                    const HybridSolver& solver, Model* best_model,
                    HybridRansacStatistics* statistics) const {
    // Initializes all relevant variables.
    ResetStatistics(statistics);
    HybridRansacStatistics& stats = *statistics;

    const int kNumSolvers = solver.num_minimal_solvers();
    stats.num_iterations_per_solver.resize(kNumSolvers, 0);

    const int kNumDataTypes = solver.num_data_types();
    stats.inlier_ratios.resize(kNumDataTypes, 0.0);
    stats.inlier_indices.resize(kNumDataTypes);

    std::vector<double> prior_probabilities;
    solver.solver_probabilities(&prior_probabilities);

    std::vector<std::vector<int>> min_sample_sizes;
    solver.min_sample_sizes(&min_sample_sizes);

    std::vector<int> num_data;
    solver.num_data(&num_data);

    if (!VerifyData(min_sample_sizes, num_data, kNumSolvers, kNumDataTypes,
                    &prior_probabilities)) {
      return 0;
    }

    Sampler sampler(options.random_seed_, solver);

    uint32_t max_num_iterations =
        std::max(options.max_num_iterations_, options.min_num_iterations_);
    stats.num_iterations_per_solver.resize(kNumSolvers, 0u);
    std::vector<uint32_t> max_num_iterations_per_solver(
        kNumSolvers, std::max(options.max_num_iterations_per_solver_,
                              options.min_num_iterations_));

    const std::vector<double>& kSqrInlierThresh =
        options.squared_inlier_thresholds_;

    Model best_minimal_model;
    double best_min_model_score = std::numeric_limits<double>::max();

    std::vector<std::vector<int>> minimal_sample(kNumDataTypes);
    ModelVector estimated_models;

    std::mt19937 rng;
    rng.seed(options.random_seed_);

    // Runs random sampling.
    for (stats.num_iterations_total = 0u;
         stats.num_iterations_total < max_num_iterations;
         ++stats.num_iterations_total) {
      // As proposed by Lebeda et al., Local Optimization is not executed in
      // the first lo_starting_iterations_ iterations. We thus run LO on the
      // best model found so far once we reach this iteration.
      if (stats.num_iterations_total == options.lo_starting_iterations_ &&
          best_min_model_score < std::numeric_limits<double>::max()) {
        ++stats.number_lo_iterations;
        LocalOptimization(options, solver, stats.best_solver_type, &rng,
                          best_model, &(stats.best_model_score),
                          &(stats.best_solver_type));

        UpdateRANSACTerminationCriteria(options, solver, *best_model,
                                        statistics,
                                        &max_num_iterations_per_solver);
      }

      const int kSolverType =
          SelectMinimalSolver(solver, prior_probabilities, stats,
                              options.min_num_iterations_, &rng);

      if (kSolverType < -1) {
        // Since no solver could be selected, we stop Hybrid RANSAC here.
        break;
      }

      stats.num_iterations_per_solver[kSolverType] += 1;

      sampler.Sample(min_sample_sizes[kSolverType], &minimal_sample);

      // MinimalSolver returns the number of estimated models.
      const int kNumEstimatedModels =
          solver.MinimalSolver(minimal_sample, kSolverType, &estimated_models);

      if (kNumEstimatedModels > 0) {
        // Finds the best model among all estimated models.
        double best_local_score = std::numeric_limits<double>::max();
        int best_local_model_id = 0;
        GetBestEstimatedModelId(options, solver, estimated_models,
                                kNumEstimatedModels, kSqrInlierThresh,
                                kNumDataTypes, num_data, &best_local_score,
                                &best_local_model_id);

        // Updates the best model found so far.
        if (best_local_score < best_min_model_score ||
            stats.num_iterations_total == options.lo_starting_iterations_) {
          const bool kBestMinModel = best_local_score < best_min_model_score;

          if (kBestMinModel) {
            // New best model (estimated from inliers found. Stores this model
            // and runs local optimization.
            best_min_model_score = best_local_score;
            best_minimal_model = estimated_models[best_local_model_id];

            // Updates the best model.
            UpdateBestModel(best_min_model_score, best_minimal_model,
                            kSolverType, &(stats.best_model_score), best_model,
                            &(stats.best_solver_type));
          }

          const bool kRunLO =
              (stats.num_iterations_total >= options.lo_starting_iterations_ &&
               best_min_model_score < std::numeric_limits<double>::max());
          if ((!kBestMinModel) && (!kRunLO)) continue;

          // Performs local optimization. By construction, the local
          // optimization method returns the best model between all models found
          // by local optimization and the input model, i.e.,
          // score_refined_model <= best_min_model_score holds.
          if (kRunLO) {
            ++stats.number_lo_iterations;
            double score = best_min_model_score;
            LocalOptimization(options, solver, stats.best_solver_type, &rng,
                              &best_minimal_model, &score,
                              &(stats.best_solver_type));

            // Updates the best model.
            UpdateBestModel(score, best_minimal_model, kSolverType,
                            &(stats.best_model_score), best_model,
                            &(stats.best_solver_type));
          }

          // Updates the number of RANSAC iterations for each solver as well
          // as the number of inliers and inlier ratios for each data type.
          UpdateRANSACTerminationCriteria(options, solver, *best_model,
                                          statistics,
                                          &max_num_iterations_per_solver);
        } else {
        }
      }

      // Terminate if the current solver reaches its maximum number of
      // iterations.
      if (stats.num_iterations_per_solver[kSolverType] >=
          max_num_iterations_per_solver[kSolverType]) {
        break;
      }
    }

    // As proposed by Lebeda et al., Local Optimization is not executed in
    // the first lo_starting_iterations_ iterations. If LO-MSAC needs less than
    // lo_starting_iterations_ iterations, we run LO now.
    if (stats.num_iterations_total <= options.lo_starting_iterations_ &&
        stats.best_model_score < std::numeric_limits<double>::max()) {
      ++stats.number_lo_iterations;
      LocalOptimization(options, solver, stats.best_solver_type, &rng,
                        best_model, &(stats.best_model_score),
                        &(stats.best_solver_type));

      UpdateRANSACTerminationCriteria(options, solver, *best_model, statistics,
                                      &max_num_iterations_per_solver);
    }

    if (options.final_least_squares_) {
      Model refined_model = *best_model;
      solver.LeastSquares(stats.inlier_indices, &refined_model);

      double score = std::numeric_limits<double>::max();
      ScoreModel(options, solver, refined_model, kSqrInlierThresh,
                 kNumDataTypes, num_data, &score);
      if (score < stats.best_model_score) {
        stats.best_model_score = score;
        *best_model = refined_model;

        // Updates the inlier ratios and the number of inliers. Updating the
        // number of RANSAC iterations is not necessary, but done here to avoid
        // code duplication.
        UpdateRANSACTerminationCriteria(options, solver, *best_model,
                                        statistics,
                                        &max_num_iterations_per_solver);
      }
    }

    return stats.best_num_inliers;
  }

 protected:
  // Randomly selects a minimal solver. See Eq. 1 in Camposeco et al.
  int SelectMinimalSolver(const HybridSolver& solver,
                          const std::vector<double> prior_probabilities,
                          const HybridRansacStatistics& stats,
                          const uint32_t min_num_iterations,
                          std::mt19937* rng) const {
    double sum_probabilities = 0.0;
    const int kNumSolvers = static_cast<int>(prior_probabilities.size());
    std::vector<std::vector<int>> min_sample_sizes;
    solver.min_sample_sizes(&min_sample_sizes);
    std::vector<double> probabilities(kNumSolvers, 0);

    const int kNumDataTypes = solver.num_data_types();

    // There is a special case where all inlier ratios are 0. In this case, the
    // solvers should be sampled based on the priors.
    const double kSumInlierRatios = std::accumulate(
        stats.inlier_ratios.begin(), stats.inlier_ratios.end(), 0.0);

    if (kSumInlierRatios == 0.0) {
      for (int i = 0; i < kNumSolvers; ++i) {
        probabilities[i] = prior_probabilities[i];
        sum_probabilities += probabilities[i];
      }
    } else {
      for (int i = 0; i < kNumSolvers; ++i) {
        double num_iters =
            static_cast<double>(stats.num_iterations_per_solver[i]);
        if (num_iters > 0.0) {
          num_iters -= 1.0;
        }

        double all_inlier_prob = 1.0;
        for (int j = 0; j < kNumDataTypes; ++j) {
          all_inlier_prob *=
              std::pow(stats.inlier_ratios[j],
                       static_cast<double>(min_sample_sizes[i][j]));
        }

        if (num_iters < static_cast<double>(min_num_iterations)) {
          probabilities[i] = all_inlier_prob * prior_probabilities[i];
        } else {
          probabilities[i] = all_inlier_prob *
                             std::pow(1.0 - all_inlier_prob, num_iters) *
                             prior_probabilities[i];
        }
        sum_probabilities += probabilities[i];
      }
    }

    std::uniform_real_distribution<double> dist(0.0, sum_probabilities);

    const double kProb = dist(*rng);
    double current_prob = 0.0;
    for (int i = 0; i < kNumSolvers; ++i) {
      if (prior_probabilities[i] == 0.0) continue;

      current_prob += probabilities[i];
      if (kProb <= current_prob) return i;
    }
    return -1;
  }

  void GetBestEstimatedModelId(
      const HybridLORansacOptions& options, const HybridSolver& solver,
      const ModelVector& models, const int num_models,
      const std::vector<double>& squared_inlier_thresholds,
      const int num_data_types, const std::vector<int> num_data,
      double* best_score, int* best_model_id) const {
    *best_score = std::numeric_limits<double>::max();
    *best_model_id = 0;

    for (int m = 0; m < num_models; ++m) {
      double score = std::numeric_limits<double>::max();
      ScoreModel(options, solver, models[m], squared_inlier_thresholds,
                 num_data_types, num_data, &score);

      if (score < *best_score) {
        *best_score = score;
        *best_model_id = m;
      }
    }
  }

  void ScoreModel(const HybridLORansacOptions& options,
                  const HybridSolver& solver, const Model& model,
                  const std::vector<double>& squared_inlier_thresholds,
                  const int num_data_types, const std::vector<int> num_data,
                  double* score) const {
    *score = 0.0;

    for (int t = 0; t < num_data_types; ++t) {
      for (int i = 0; i < num_data[t]; ++i) {
        double squared_error = solver.EvaluateModelOnPoint(model, t, i);
        *score += ComputeScore(squared_error, squared_inlier_thresholds[t]) *
                  options.data_type_weights_[t];
      }
    }
  }

  // MSAC (top-hat) scoring function.
  inline double ComputeScore(const double squared_error,
                             const double squared_error_threshold) const {
    return std::min(squared_error, squared_error_threshold);
  }

  int GetInliers(const HybridSolver& solver, const Model& model,
                 const std::vector<double>& squared_inlier_thresholds,
                 std::vector<std::vector<int>>* inliers) const {
    const int kNumDataTypes = solver.num_data_types();
    std::vector<int> num_data;
    solver.num_data(&num_data);
    if (inliers == nullptr) {
      int num_inliers = 0;
      for (int t = 0; t < kNumDataTypes; ++t) {
        for (int i = 0; i < num_data[i]; ++i) {
          double squared_error = solver.EvaluateModelOnPoint(model, t, i);
          if (squared_error < squared_inlier_thresholds[t]) {
            ++num_inliers;
          }
        }
      }
      return num_inliers;
    } else {
      inliers->clear();
      inliers->resize(kNumDataTypes);
      int num_inliers = 0;
      for (int t = 0; t < kNumDataTypes; ++t) {
        (*inliers)[t].clear();
        for (int i = 0; i < num_data[t]; ++i) {
          double squared_error = solver.EvaluateModelOnPoint(model, t, i);
          if (squared_error < squared_inlier_thresholds[t]) {
            ++num_inliers;
            (*inliers)[t].push_back(i);
          }
        }
      }
      return num_inliers;
    }
  }

  void UpdateRANSACTerminationCriteria(
      const HybridLORansacOptions& options, const HybridSolver& solver,
      const Model& model, HybridRansacStatistics* statistics,
      std::vector<uint32_t>* max_iterations) const {
    statistics->best_num_inliers =
        GetInliers(solver, model, options.squared_inlier_thresholds_,
                   &(statistics->inlier_indices));

    const int kNumDataTypes = solver.num_data_types();
    std::vector<int> num_data;
    solver.num_data(&num_data);

    for (int d = 0; d < kNumDataTypes; ++d) {
      if (num_data[d] > 0) {
        statistics->inlier_ratios[d] =
            static_cast<double>(statistics->inlier_indices[d].size()) /
            static_cast<double>(num_data[d]);
      } else {
        statistics->inlier_ratios[d] = 0.0;
      }
    }

    std::vector<std::vector<int>> min_sample_sizes;
    solver.min_sample_sizes(&min_sample_sizes);
    const int kNumSolvers = solver.num_minimal_solvers();
    for (int s = 0; s < kNumSolvers; ++s) {
      (*max_iterations)[s] = utils::NumRequiredIterations(
          statistics->inlier_ratios, 1.0 - options.success_probability_,
          min_sample_sizes[s], options.min_num_iterations_,
          options.max_num_iterations_per_solver_);
    }
  }

  // See algorithms 2 and 3 in Lebeda et al.
  // The input model is overwritten with the refined model if the latter is
  // better, i.e., has a lower score.
  void LocalOptimization(const HybridLORansacOptions& options,
                         const HybridSolver& solver, const int solver_type,
                         std::mt19937* rng, Model* best_minimal_model,
                         double* score_best_minimal_model,
                         int* best_solver_type) const {
    std::vector<int> num_data;
    solver.num_data(&num_data);
    const int kNumDataTypes = static_cast<int>(num_data.size());

    std::vector<std::vector<int>> min_sample_sizes;
    solver.min_sample_sizes(&min_sample_sizes);

    const double kThreshMult = options.threshold_multiplier_;
    std::vector<double> squared_inlier_thresholds =
        options.squared_inlier_thresholds_;
    std::vector<double> thresh_mult_updates(squared_inlier_thresholds.size());
    for (int i = 0; i < kNumDataTypes; ++i) {
      thresh_mult_updates[i] =
          (kThreshMult - 1.0) * squared_inlier_thresholds[i] /
          static_cast<int>(options.num_lsq_iterations_ - 1);
      squared_inlier_thresholds[i] *= kThreshMult;
    }

    // Performs an initial least squares fit of the best model found by the
    // minimal solver so far and then determines the inliers to that model
    // under a (slightly) relaxed inlier threshold.
    Model m_init = *best_minimal_model;
    LeastSquaresFit(options, squared_inlier_thresholds, solver_type, solver,
                    rng, &m_init);

    double score = std::numeric_limits<double>::max();
    ScoreModel(options, solver, m_init, options.squared_inlier_thresholds_,
               kNumDataTypes, num_data, &score);
    UpdateBestModel(score, m_init, solver_type, score_best_minimal_model,
                    best_minimal_model, best_solver_type);

    std::vector<std::vector<int>> inliers_base;
    GetInliers(solver, m_init, options.squared_inlier_thresholds_,
               &inliers_base);

    // Performs the actual local optimization (LO).
    // Note that compared to the original definition of the LO step, no non-
    // minimal sample is drawn as a non-minimal sample over multiple data types
    // is not well-defined.
    std::vector<int> sample;
    for (int r = 0; r < options.num_lo_steps_; ++r) {
      Model m_non_min = m_init;

      // Iterative least squares refinement. Note that a random subset of all
      // inliers is used.
      LeastSquaresFit(options, options.squared_inlier_thresholds_, solver_type,
                      solver, rng, &m_non_min);

      // The current threshold multiplier and its update.
      std::vector<double> cur_squared_inlier_thresholds = squared_inlier_thresholds;

      for (int i = 0; i < options.num_lsq_iterations_; ++i) {
        LeastSquaresFit(options, cur_squared_inlier_thresholds, solver_type, solver,
                        rng, &m_non_min);

        ScoreModel(options, solver, m_non_min,
                   options.squared_inlier_thresholds_, kNumDataTypes, num_data,
                   &score);
        UpdateBestModel(score, m_non_min, solver_type, score_best_minimal_model,
                        best_minimal_model, best_solver_type);
        for (int j = 0; j < kNumDataTypes; ++j) {
          cur_squared_inlier_thresholds[j] -= thresh_mult_updates[j];
        }
      }
    }
  }

  void LeastSquaresFit(const HybridLORansacOptions& options,
                       const std::vector<double>& thresholds,
                       const int solver_type, const HybridSolver& solver,
                       std::mt19937* rng, Model* model) const {
    std::vector<std::vector<int>> sample_sizes;
    solver.min_sample_sizes(&sample_sizes);

    std::vector<std::vector<int>> inliers;
    int num_inliers = GetInliers(solver, *model, thresholds, &inliers);

    const int kNumDataTypes = solver.num_data_types();
    for (int i = 0; i < kNumDataTypes; ++i) {
      if (static_cast<int>(inliers[i].size()) < sample_sizes[solver_type][i]) {
        // The estimated pose has fewer inliers than required by the minimal
        // sample size. In this case, the least squares solution will likely be
        // very inaccurate and we thus skip the least squares fitting step.
        return;
      }

      sample_sizes[solver_type][i] *= options.min_sample_multiplicator_;
      sample_sizes[solver_type][i] = std::min(
          sample_sizes[solver_type][i], static_cast<int>(inliers[i].size()));
    }

    utils::RandomShuffleAndResize(sample_sizes[solver_type], rng, &inliers);
    solver.LeastSquares(inliers, model);
  }

  inline void UpdateBestModel(const double score_curr, const Model& m_curr,
                              const int solver_type, double* score_best,
                              Model* m_best, int* best_solver_type) const {
    if (score_curr < *score_best) {
      *score_best = score_curr;
      *m_best = m_curr;
      *best_solver_type = solver_type;
    }
  }

  // Determines whether enough data is available to run any of the minimal
  // solvers. Returns false otherwise. For those solvers that are not feasible
  // because not all data is available, the prior probability is set to 0.
  bool VerifyData(const std::vector<std::vector<int>>& min_sample_sizes,
                  const std::vector<int>& num_data, const int num_solvers,
                  const int num_data_types,
                  std::vector<double>* prior_probabilities) const {
    for (int i = 0; i < num_solvers; ++i) {
      for (int j = 0; j < num_data_types; ++j) {
        if (min_sample_sizes[i][j] > num_data[j] ||
            min_sample_sizes[i][j] < 0) {
          (*prior_probabilities)[i] = 0.0;
          break;
        }
      }
    }

    // No need to run HybridRANSAC if none of the solvers can be used.
    bool any_valid_solver = false;
    for (int i = 0; i < num_solvers; ++i) {
      if ((*prior_probabilities)[i] > 0.0) {
        any_valid_solver = true;
        break;
      }
    }

    if (!any_valid_solver) {
      return false;
    }

    return true;
  }
};

}  // namespace ransac_lib

#endif  // RANSACLIB_RANSACLIB_HYBRID_RANSAC_H_
