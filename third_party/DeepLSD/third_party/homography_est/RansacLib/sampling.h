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

#ifndef RANSACLIB_RANSACLIB_SAMPLING_H_
#define RANSACLIB_RANSACLIB_SAMPLING_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace ransac_lib {

// Implements uniform sampling for RANSAC.
template <class Solver>
class UniformSampling {
 public:
  UniformSampling(const unsigned int random_seed, const Solver& solver)
      : num_data_(solver.num_data()), sample_size_(solver.min_sample_size()) {
    rng_.seed(random_seed);
    draw_sample_ = DrawBetterThanShuffle(sample_size_, num_data_);
    uniform_dstr_.param(
        std::uniform_int_distribution<int>::param_type(0, num_data_ - 1));
  }

  // Draws minimal sample.
  void Sample(std::vector<int>* random_sample) {
    if (draw_sample_) {
      DrawSample(random_sample);
    } else {
      ShuffleSample(random_sample);
    }
  }

 protected:
  // Function to decide whether random sampling or shuffling is more
  // efficient. Returns true if sampling is more efficient.
  inline bool DrawBetterThanShuffle(const int sample_size,
                                    const int num_elements) const {
    const double kCoeff = static_cast<double>(num_elements) /
                          static_cast<double>(num_elements - sample_size);
    // Exercise for the interested reader: Determine where this equation comes
    // from. Hint: Use Harmonic numbers and approximate them as
    // H_n \approx ln(n) (actually: ln(n) + 1/n <= H_n <= ln(n) + 1,
    // but we use equality for simplicity here.
    return (kCoeff < M_E);
  }

  // Draws a minimal sample of size sample_size.
  void DrawSample(std::vector<int>* random_sample) {
    std::vector<int>& sample = *random_sample;
    sample.resize(sample_size_);
    for (int i = 0; i < sample_size_; ++i) {
      bool found = true;
      while (found) {
        found = false;
        sample[i] = uniform_dstr_(rng_);
        for (int j = 0; j < i; ++j) {
          if (sample[j] == sample[i]) {
            found = true;
            break;
          }
        }
      }
    }
  }

  // DrawSample is efficient is sample_size is small enough compared to the
  // number of elements. Otherwise, it is faster to randomly shuffle the
  // elements and pick the first sample_size ones.
  void ShuffleSample(std::vector<int>* random_sample) {
    random_sample->resize(num_data_);
    std::iota(random_sample->begin(), random_sample->end(), 0);
    if (sample_size_ == num_data_) return;

    // Fisher-Yates shuffling.
    RandomShuffle(random_sample);
    random_sample->resize(sample_size_);
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
  std::uniform_int_distribution<int> uniform_dstr_;
  // The number of data points.
  int num_data_;
  // The size of a sample.
  int sample_size_;
  // Whether it is cheaper (in terms of expected costs) to draw a sample of to
  // randomly shuffle the sample.
  bool draw_sample_;
};

}  // namespace ransac_lib

#endif  // RANSACLIB_RANSACLIB_SAMPLING_H_
