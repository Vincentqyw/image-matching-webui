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

#include <vector>

namespace gcransac
{
	namespace sampler
	{
		// Purely virtual class used for the sampling consensus methods (e.g. Ransac,
		// Prosac, MLESac, etc.)
		template <class _DataContainer, class _IndexType>
		class Sampler
		{
		protected:
			// The pointer of the container consisting of the data points from which
			// the neighborhood graph is constructed.
			const _DataContainer * const container;

			// A variable showing if the initialization was succesfull
			bool initialized;

		public:
			explicit Sampler(const _DataContainer * const container_) :
				container(container_),
				initialized(false)
			{}

			virtual ~Sampler() {}

			virtual const std::string getName() const = 0;

			virtual void reset() = 0;

			// Initializes any non-trivial variables and sets up sampler if
			// necessary. Must be called before sample is called.
			virtual bool initialize(const _DataContainer * const container_) = 0;

			// Samples the input variable data and fills the std::vector subset with the
			// samples.
			OLGA_INLINE virtual bool sample(const std::vector<_IndexType> &pool_,
				_IndexType * const subset_,
				size_t sample_size_) = 0;

			bool isInitialized() const
			{
				return initialized;
			}
		};
	}
}