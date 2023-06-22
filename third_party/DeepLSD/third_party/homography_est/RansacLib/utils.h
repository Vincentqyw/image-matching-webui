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

#ifndef RANSACLIB_RANSACLIB_UTILS_H_
#define RANSACLIB_RANSACLIB_UTILS_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace ransac_lib {
namespace utils {

// This function implements Fisher-Yates shuffling, implemented "manually"
// here following: https://lemire.me/blog/2016/10/10/a-case-study-in-the-
// performance-cost-of-abstraction-cs-stdshuffle/
inline void RandomShuffle(std::mt19937* rng, std::vector<int>* random_sample) {
  std::vector<int>& sample = *random_sample;
  const int kNumElements = static_cast<int>(sample.size());
  for (int i = 0; i < (kNumElements - 1); ++i) {
    std::uniform_int_distribution<int> dist(i, kNumElements - 1);
    int idx = dist(*rng);
    std::swap(sample[i], sample[idx]);
  }
}

template <class T>
inline void RandomShuffle(std::mt19937* rng, std::vector<T>* random_sample) {
  std::vector<T>& sample = *random_sample;
  const int kNumElements = static_cast<int>(sample.size());
  for (int i = 0; i < (kNumElements - 1); ++i) {
    std::uniform_int_distribution<int> dist(i, kNumElements - 1);
    int idx = dist(*rng);
    std::swap(sample[i], sample[idx]);
  }
}

inline void RandomShuffleAndResize(const int target_size, std::mt19937* rng,
                                   std::vector<int>* random_sample) {
  RandomShuffle(rng, random_sample);
  random_sample->resize(target_size);
}

// RandomShuffleAndResize variants for Hybrid RANSAC.
inline void RandomShuffleAndResize(
    const int target_size, std::mt19937* rng,
    std::vector<std::vector<int>>* random_sample) {
  const int kNumDataTypes = static_cast<int>(random_sample->size());
  std::vector<std::pair<int, int>> data;
  for (int i = 0; i < kNumDataTypes; ++i) {
    const int kNumData = static_cast<int>((*random_sample)[i].size());
    for (int j = 0; j < kNumData; ++j) {
      data.push_back(std::make_pair(i, (*random_sample)[i][j]));
    }
    (*random_sample)[i].clear();
  }
  RandomShuffle<std::pair<int, int>>(rng, &data);
  data.resize(target_size);

  for (int i = 0; i < target_size; ++i) {
    (*random_sample)[data[i].first].push_back(data[i].second);
  }
}

inline void RandomShuffleAndResize(
    const std::vector<int> sample_sizes, std::mt19937* rng,
    std::vector<std::vector<int>>* random_sample) {
  const int kNumDataTypes = static_cast<int>(random_sample->size());
  for (int i = 0; i < kNumDataTypes; ++i) {
    const int kNumData = static_cast<int>((*random_sample)[i].size());
    const int kSampleSize = std::min(kNumData, sample_sizes[i]);
    RandomShuffleAndResize(kSampleSize, rng, &((*random_sample)[i]));
  }
}

// Computes the number of RANSAC iterations required for a given inlier
// ratio, the probability of missing the best model, and sample size.
// Assumes that min_iterations <= max_iterations.
inline uint32_t NumRequiredIterations(const double inlier_ratio,
                                      const double prob_missing_best_model,
                                      const int sample_size,
                                      const uint32_t min_iterations,
                                      const uint32_t max_iterations) {
  if (inlier_ratio <= 0.0) {
    return max_iterations;
  }
  if (inlier_ratio >= 1.0) {
    return min_iterations;
  }

  const double kProbNonInlierSample =
      1.0 - std::pow(inlier_ratio, static_cast<double>(sample_size));
  // If the probability of sampling a non-all-inlier sample is at least
  // 0.99999999999999, RANSAC will take at least 1e+13 iterations for 
  // realistic values for prob_missing_best_model (0.5 or smaller).
  // In practice, max_iterations will be smaller.
  if (kProbNonInlierSample >= 0.99999999999999) {
    return max_iterations;
  }
  
  const double kLogNumerator = std::log(prob_missing_best_model);
  const double kLogDenominator = std::log(kProbNonInlierSample);

  double num_iters = std::ceil(kLogNumerator / kLogDenominator + 0.5);
  uint32_t num_req_iterations =
      std::min(static_cast<uint32_t>(num_iters), max_iterations);
  num_req_iterations = std::max(min_iterations, num_req_iterations);
  return num_req_iterations;
}

inline uint32_t NumRequiredIterations(const std::vector<double> inlier_ratios,
                                      const double prob_missing_best_model,
                                      const std::vector<int> sample_sizes,
                                      const uint32_t min_iterations,
                                      const uint32_t max_iterations) {
  const int kNumDataTypes = static_cast<int>(sample_sizes.size());

  double prob_all_inlier_sample = 1.0;
  for (int i = 0; i < kNumDataTypes; ++i) {
    prob_all_inlier_sample *=
        std::pow(inlier_ratios[i], static_cast<double>(sample_sizes[i]));
  }
  if (prob_all_inlier_sample <= 0.0) {
    return max_iterations;
  }
  if (prob_all_inlier_sample >= 1.0) {
    return min_iterations;
  }

  const double kProbNonInlierSample = 1.0 - prob_all_inlier_sample;
  // If the probability of sampling a non-all-inlier sample is at least
  // 0.99999999999999, RANSAC will take at least 1e+13 iterations for 
  // realistic values for prob_missing_best_model (0.5 or smaller).
  // In practice, max_iterations will be smaller.
  if (kProbNonInlierSample >= 0.99999999999999) {
    return max_iterations;
  }
  
  const double kLogNumerator = std::log(prob_missing_best_model);
  const double kLogDenominator = std::log(kProbNonInlierSample);

  double num_iters = std::ceil(kLogNumerator / kLogDenominator + 0.5);
  uint32_t num_req_iterations =
      std::min(static_cast<uint32_t>(num_iters), max_iterations);
  num_req_iterations = std::max(min_iterations, num_req_iterations);
  return num_req_iterations;
}

}  // namespace utils
}  // namespace ransac_lib

#endif  // RANSACLIB_RANSACLIB_UTILS_H_
