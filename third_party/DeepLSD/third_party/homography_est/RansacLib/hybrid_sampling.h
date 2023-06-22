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

#ifndef RANSACLIB_RANSACLIB_HYBRID_SAMPLING_H_
#define RANSACLIB_RANSACLIB_HYBRID_SAMPLING_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace ransac_lib {

// Implements uniform sampling for HybridRANSAC.
template <class Solver>
class HybridUniformSampling {
 public:
  HybridUniformSampling(const unsigned int random_seed,
                        const Solver& solver) {
    solver.num_data(&num_data_);
    rng_.seed(random_seed);
    num_data_types_ = static_cast<int>(num_data_.size());

    uniform_dstr_.resize(num_data_types_);
    for (int i = 0; i < num_data_types_; ++i) {
      if (num_data_[i] < 2) continue;
      uniform_dstr_[i].param(
          std::uniform_int_distribution<int>::param_type(0, num_data_[i] - 1));
    }
  }

  // Draws minimal sample.
  void Sample(const std::vector<int>& num_samples_per_data_type,
              std::vector<std::vector<int>>* random_sample) {
    std::vector<std::vector<int>>& sample = *random_sample;
    sample.resize(num_data_types_);
    for (int i = 0; i < num_data_types_; ++i) {
      sample[i].clear();
      if (num_samples_per_data_type[i] <= 0) continue;
      const double kCoeff =
          static_cast<double>(num_data_[i]) /
          static_cast<double>(num_data_[i] - num_samples_per_data_type[i]);
      if (kCoeff < M_E) {
        DrawSample(num_samples_per_data_type, i, random_sample);
      } else {
        ShuffleSample(num_samples_per_data_type, i, random_sample);
      }
    }
  }

 protected:
  // Draws a minimal sample of size sample_size.
  void DrawSample(const std::vector<int>& num_samples_per_data_type,
                  const int data_type,
                  std::vector<std::vector<int>>* random_sample) {
    std::vector<std::vector<int>>& sample = *random_sample;
    sample[data_type].resize(num_samples_per_data_type[data_type]);
    for (int i = 0; i < num_samples_per_data_type[data_type]; ++i) {
      bool found = true;
      while (found) {
        found = false;
        sample[data_type][i] = uniform_dstr_[data_type](rng_);
        for (int j = 0; j < i; ++j) {
          if (sample[data_type][j] == sample[data_type][i]) {
            found = true;
            break;
          }
        }
      }
    }
  }

  // DrawSample is efficient is sample_size[i] is small enough compared to the
  // number of elements. Otherwise, it is faster to randomly shuffle the
  // elements and pick the first sample_size ones.
  void ShuffleSample(const std::vector<int>& num_samples_per_data_type,
                     const int data_type,
                     std::vector<std::vector<int>>* random_sample) {
    (*random_sample)[data_type].resize(num_data_[data_type]);
    std::iota((*random_sample)[data_type].begin(),
              (*random_sample)[data_type].end(), 0);
    if (num_samples_per_data_type[data_type] == num_data_[data_type]) return;

    // Fisher-Yates shuffling.
    RandomShuffle(&((*random_sample)[data_type]));
    (*random_sample)[data_type].resize(num_samples_per_data_type[data_type]);
  }

  // This function implements Fisher-Yates shuffling, implemented "manually"
  // here following: https://lemire.me/blog/2016/10/10/a-case-study-in-the-
  // performance-cost-of-abstraction-cs-stdshuffle/
  void RandomShuffle(std::vector<int>* random_sample) {
    std::vector<int>& sample = *random_sample;
    const int kNumElements = static_cast<int>(sample.size());
    for (int i = 0; i < (kNumElements - 1); ++i) {
      std::uniform_int_distribution<int> dist(i, kNumElements - 1);
      int idx = dist(rng_);
      std::swap(sample[i], sample[idx]);
    }
  }

  // The random number generator used by RANSAC.
  std::mt19937 rng_;
  std::vector<std::uniform_int_distribution<int>> uniform_dstr_;
  // The number of data types.
  int num_data_types_;
  // The number of data points for each data type.
  std::vector<int> num_data_;
};

// Implements a biased sampling for HybridRANSAC, where each data point has
// an associated weight and points with a higher weight are more likely to be
// sampled. Points with weight 0 are ignored during sampling.
template <class Solver>
class HybridBiasedSampling {
 public:
  HybridBiasedSampling(const unsigned int random_seed,
                       const Solver& solver) {
    std::vector<std::vector<double>> weights;
    solver.get_weights(weights);
    rng_.seed(random_seed);
    num_data_types_ = static_cast<int>(weights.size());

    data_ids_.resize(weights.size());
    distributions_.resize(num_data_types_);

    for (int i = 0; i < num_data_types_; ++i) {
      data_ids_[i].reserve(weights[i].size());
      std::vector<double> selected_weights;
      selected_weights.reserve(weights[i].size());
      const int kNumElements = static_cast<int>(weights[i].size());
      for (int j = 0; j < kNumElements; ++j) {
        if (weights[i][j] > 0.0) {
          selected_weights.push_back(weights[i][j]);
          data_ids_[i].push_back(j);
        }
      }

      std::discrete_distribution<int> dstr(selected_weights.begin(),
                                           selected_weights.end());
      distributions_[i].param(dstr.param());
    }
  }

  // Draws minimal sample.
  void Sample(const std::vector<int>& num_samples_per_data_type,
              std::vector<std::vector<int>>* random_sample) {
    std::vector<std::vector<int>>& sample = *random_sample;
    sample.resize(num_data_types_);
    for (int i = 0; i < num_data_types_; ++i) {
      sample[i].clear();
      if (num_samples_per_data_type[i] <= 0) continue;
      if (num_samples_per_data_type[i] ==
          static_cast<int>(data_ids_[i].size())) {
        (*random_sample)[i] = data_ids_[i];
      } else {
        DrawSample(num_samples_per_data_type, i, random_sample);
      }
    }
  }

 protected:
  // Draws a minimal sample of size sample_size.
  void DrawSample(const std::vector<int>& num_samples_per_data_type,
                  const int data_type,
                  std::vector<std::vector<int>>* random_sample) {
    std::vector<std::vector<int>>& sample = *random_sample;
    sample[data_type].resize(num_samples_per_data_type[data_type]);
    for (int i = 0; i < num_samples_per_data_type[data_type]; ++i) {
      bool found = true;
      while (found) {
        found = false;
        sample[data_type][i] =
            data_ids_[data_type][distributions_[data_type](rng_)];
        for (int j = 0; j < i; ++j) {
          if (sample[data_type][j] == sample[data_type][i]) {
            found = true;
            break;
          }
        }
      }
    }
  }

  // The random number generator used by RANSAC.
  std::mt19937 rng_;
  std::vector<std::discrete_distribution<int>> distributions_;
  // The number of data types.
  int num_data_types_;
  // The data ids for each data type.
  std::vector<std::vector<int>> data_ids_;
};

}  // namespace ransac_lib

#endif  // RANSACLIB_RANSACLIB_HYBRID_SAMPLING_H_
