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

#include "solver_engine.h"
#include "fundamental_estimator.h"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class EssentialMatrixFivePointSteweniusSolver : public SolverEngine
			{
			public:
				EssentialMatrixFivePointSteweniusSolver()
				{
				}

				~EssentialMatrixFivePointSteweniusSolver()
				{
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 5;
				}

				// Determines if there is a chance of returning multiple models
				// the function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return maximumSolutions() > 1;
				}

				// The maximum number of solutions returned by the estimator
				static constexpr size_t maximumSolutions()
				{
					return 10;
				}

				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				OLGA_INLINE bool estimateModel(
					const cv::Mat& data_, // The set of data points
					const size_t *sample_, // The sample used for the estimation
					size_t sample_number_, // The size of the sample
					std::vector<Model> &models_, // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point

			protected:
				inline Eigen::Matrix<double, 1, 10> multiplyDegOnePoly(
					const Eigen::RowVector4d& a,
					const Eigen::RowVector4d& b) const;

				inline Eigen::Matrix<double, 1, 20> multiplyDegTwoDegOnePoly(
					const Eigen::Matrix<double, 1, 10>& a,
					const Eigen::RowVector4d& b) const;

				inline Eigen::Matrix<double, 10, 20> buildConstraintMatrix(
					const Eigen::Matrix<double, 1, 4> nullSpace[3][3]) const;

				inline Eigen::Matrix<double, 9, 20> getTraceConstraint(
					const Eigen::Matrix<double, 1, 4> nullSpace[3][3]) const;

				inline Eigen::Matrix<double, 1, 10>
					computeEETranspose(const Eigen::Matrix<double, 1, 4> nullSpace[3][3], int i, int j) const;

				inline Eigen::Matrix<double, 1, 20> getDeterminantConstraint(
					const Eigen::Matrix<double, 1, 4> nullSpace[3][3]) const;
			};

			OLGA_INLINE bool EssentialMatrixFivePointSteweniusSolver::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				if (sample_ == nullptr)
					sample_number_ = data_.rows;

				Eigen::MatrixXd coefficients(sample_number_, 9);
				const double *data_ptr = reinterpret_cast<double *>(data_.data);
				const int cols = data_.cols;

				// Step 1. Create the nx9 matrix containing epipolar constraints.
				//   Essential matrix is a linear combination of the 4 vectors spanning the null space of this
				//   matrix.
				int offset;
				double x0, y0, x1, y1, weight = 1.0;
				for (size_t i = 0; i < sample_number_; i++)
				{
					if (sample_ == nullptr)
					{
						offset = cols * i;
						if (weights_ != nullptr)
							weight = weights_[i];
					}
					else
					{
						offset = cols * sample_[i];
						if (weights_ != nullptr)
							weight = weights_[sample_[i]];
					}

					x0 = data_ptr[offset];
					y0 = data_ptr[offset + 1];
					x1 = data_ptr[offset + 2];
					y1 = data_ptr[offset + 3];
					
					// Precalculate these values to avoid calculating them multiple times
					const double
						weight_times_x0 = weight * x0,
						weight_times_x1 = weight * x1,
						weight_times_y0 = weight * y0,
						weight_times_y1 = weight * y1;

					coefficients.row(i) <<
						weight_times_x0 * x1,
						weight_times_x0 * y1,
						weight_times_x0,
						weight_times_y0 * x1,
						weight_times_y0 * y1,
						weight_times_y0,
						weight_times_x1,
						weight_times_y1,
						weight;
				}

				// Extract the null space from a minimal sampling (using LU) or non-minimal sampling (using SVD).
				Eigen::Matrix<double, 9, 4> nullSpace;

				if (sample_number_ == 5) {
					const Eigen::FullPivLU<Eigen::MatrixXd> lu(coefficients);
					if (lu.dimensionOfKernel() != 4) {
						return false;
					}
					nullSpace = lu.kernel();
				}
				else {
					const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(
						coefficients.transpose() * coefficients);
					const Eigen::MatrixXd &Q = qr.matrixQ();
					nullSpace = Q.rightCols<4>();
				}

				const Eigen::Matrix<double, 1, 4> nullSpaceMatrix[3][3] = {
					{nullSpace.row(0), nullSpace.row(3), nullSpace.row(6)},
					{nullSpace.row(1), nullSpace.row(4), nullSpace.row(7)},
					{nullSpace.row(2), nullSpace.row(5), nullSpace.row(8)} };

				// Step 2. Expansion of the epipolar constraints on the determinant and trace.
				const Eigen::Matrix<double, 10, 20> constraintMatrix = buildConstraintMatrix(nullSpaceMatrix);

				// Step 3. Eliminate part of the matrix to isolate polynomials in z.
				Eigen::FullPivLU<Eigen::Matrix<double, 10, 10>> c_lu(constraintMatrix.block<10, 10>(0, 0));
				const Eigen::Matrix<double, 10, 10> eliminatedMatrix = c_lu.solve(constraintMatrix.block<10, 10>(0, 10));

				Eigen::Matrix<double, 10, 10> actionMatrix = Eigen::Matrix<double, 10, 10>::Zero();
				actionMatrix.block<3, 10>(0, 0) = eliminatedMatrix.block<3, 10>(0, 0);
				actionMatrix.row(3) = eliminatedMatrix.row(4);
				actionMatrix.row(4) = eliminatedMatrix.row(5);
				actionMatrix.row(5) = eliminatedMatrix.row(7);
				actionMatrix(6, 0) = -1.0;
				actionMatrix(7, 1) = -1.0;
				actionMatrix(8, 3) = -1.0;
				actionMatrix(9, 6) = -1.0;

				Eigen::EigenSolver<Eigen::Matrix<double, 10, 10>> eigensolver(actionMatrix);
				const Eigen::VectorXcd& eigenvalues = eigensolver.eigenvalues();

				// Now that we have x, y, and z we need to substitute them back into the null space to get a valid
				// essential matrix solution.
				for (size_t i = 0; i < 10; i++) {
					// Only consider real solutions.
					if (eigenvalues(i).imag() != 0) {
						continue;
					}
					Eigen::Matrix3d E_dst_src;
					Eigen::Map<Eigen::Matrix<double, 9, 1>>(E_dst_src.data()) =
						nullSpace * eigensolver.eigenvectors().col(i).tail<4>().real();

					EssentialMatrix model;
					model.descriptor = E_dst_src;
					models_.push_back(model);
				}

				return models_.size() > 0;
			}

			// Multiply two degree one polynomials of variables x, y, z.
			// E.g. p1 = a[0]x + a[1]y + a[2]z + a[3]
			// Output order: x^2 xy y^2 xz yz z^2 x y z 1 (GrevLex)
			inline Eigen::Matrix<double, 1, 10> EssentialMatrixFivePointSteweniusSolver::multiplyDegOnePoly(
				const Eigen::RowVector4d& a,
				const Eigen::RowVector4d& b) const {
				Eigen::Matrix<double, 1, 10> output;
				// x^2
				output(0) = a(0) * b(0);
				// xy
				output(1) = a(0) * b(1) + a(1) * b(0);
				// y^2
				output(2) = a(1) * b(1);
				// xz
				output(3) = a(0) * b(2) + a(2) * b(0);
				// yz
				output(4) = a(1) * b(2) + a(2) * b(1);
				// z^2
				output(5) = a(2) * b(2);
				// x
				output(6) = a(0) * b(3) + a(3) * b(0);
				// y
				output(7) = a(1) * b(3) + a(3) * b(1);
				// z
				output(8) = a(2) * b(3) + a(3) * b(2);
				// 1
				output(9) = a(3) * b(3);
				return output;
			}

			// Multiply a 2 deg poly (in x, y, z) and a one deg poly in GrevLex order.
			// x^3 x^2y xy^2 y^3 x^2z xyz y^2z xz^2 yz^2 z^3 x^2 xy y^2 xz yz z^2 x y z 1
			inline Eigen::Matrix<double, 1, 20> EssentialMatrixFivePointSteweniusSolver::multiplyDegTwoDegOnePoly(
				const Eigen::Matrix<double, 1, 10>& a,
				const Eigen::RowVector4d& b) const {
				Eigen::Matrix<double, 1, 20> output;
				// x^3
				output(0) = a(0) * b(0);
				// x^2y
				output(1) = a(0) * b(1) + a(1) * b(0);
				// xy^2
				output(2) = a(1) * b(1) + a(2) * b(0);
				// y^3
				output(3) = a(2) * b(1);
				// x^2z
				output(4) = a(0) * b(2) + a(3) * b(0);
				// xyz
				output(5) = a(1) * b(2) + a(3) * b(1) + a(4) * b(0);
				// y^2z
				output(6) = a(2) * b(2) + a(4) * b(1);
				// xz^2
				output(7) = a(3) * b(2) + a(5) * b(0);
				// yz^2
				output(8) = a(4) * b(2) + a(5) * b(1);
				// z^3
				output(9) = a(5) * b(2);
				// x^2
				output(10) = a(0) * b(3) + a(6) * b(0);
				// xy
				output(11) = a(1) * b(3) + a(6) * b(1) + a(7) * b(0);
				// y^2
				output(12) = a(2) * b(3) + a(7) * b(1);
				// xz
				output(13) = a(3) * b(3) + a(6) * b(2) + a(8) * b(0);
				// yz
				output(14) = a(4) * b(3) + a(7) * b(2) + a(8) * b(1);
				// z^2
				output(15) = a(5) * b(3) + a(8) * b(2);
				// x
				output(16) = a(6) * b(3) + a(9) * b(0);
				// y
				output(17) = a(7) * b(3) + a(9) * b(1);
				// z
				output(18) = a(8) * b(3) + a(9) * b(2);
				// 1
				output(19) = a(9) * b(3);
				return output;
			}

			inline Eigen::Matrix<double, 1, 20> EssentialMatrixFivePointSteweniusSolver::getDeterminantConstraint(
				const Eigen::Matrix<double, 1, 4> nullSpace[3][3]) const {
				// Singularity constraint.
				return multiplyDegTwoDegOnePoly(
					multiplyDegOnePoly(nullSpace[0][1], nullSpace[1][2]) -
					multiplyDegOnePoly(nullSpace[0][2], nullSpace[1][1]),
					nullSpace[2][0]) +
					multiplyDegTwoDegOnePoly(
						multiplyDegOnePoly(nullSpace[0][2], nullSpace[1][0]) -
						multiplyDegOnePoly(nullSpace[0][0], nullSpace[1][2]),
						nullSpace[2][1]) +
					multiplyDegTwoDegOnePoly(
						multiplyDegOnePoly(nullSpace[0][0], nullSpace[1][1]) -
						multiplyDegOnePoly(nullSpace[0][1], nullSpace[1][0]),
						nullSpace[2][2]);
			}

			// Shorthand for multiplying the Essential matrix with its transpose.
			inline Eigen::Matrix<double, 1, 10> EssentialMatrixFivePointSteweniusSolver::computeEETranspose(
				const Eigen::Matrix<double, 1, 4> nullSpace[3][3],
				int i,
				int j) const {
				return multiplyDegOnePoly(nullSpace[i][0], nullSpace[j][0]) +
					multiplyDegOnePoly(nullSpace[i][1], nullSpace[j][1]) +
					multiplyDegOnePoly(nullSpace[i][2], nullSpace[j][2]);
			}

			// Builds the trace constraint: EEtE - 1/2 trace(EEt)E = 0
			inline Eigen::Matrix<double, 9, 20> EssentialMatrixFivePointSteweniusSolver::getTraceConstraint(
				const Eigen::Matrix<double, 1, 4> nullSpace[3][3]) const {
				Eigen::Matrix<double, 9, 20> traceConstraint;

				// Compute EEt.
				Eigen::Matrix<double, 1, 10> eet[3][3];
				for (int i = 0; i < 3; i++) {
					for (int j = 0; j < 3; j++) {
						eet[i][j] = 2 * computeEETranspose(nullSpace, i, j);
					}
				}

				// Compute the trace.
				const Eigen::Matrix<double, 1, 10> trace = eet[0][0] + eet[1][1] + eet[2][2];

				// Multiply EEt with E.
				for (auto i = 0; i < 3; i++) {
					for (auto j = 0; j < 3; j++) {
						traceConstraint.row(3 * i + j) = multiplyDegTwoDegOnePoly(eet[i][0], nullSpace[0][j]) +
							multiplyDegTwoDegOnePoly(eet[i][1], nullSpace[1][j]) +
							multiplyDegTwoDegOnePoly(eet[i][2], nullSpace[2][j]) -
							0.5 * multiplyDegTwoDegOnePoly(trace, nullSpace[i][j]);
					}
				}

				return traceConstraint;
			}

			inline Eigen::Matrix<double, 10, 20> EssentialMatrixFivePointSteweniusSolver::buildConstraintMatrix(
				const Eigen::Matrix<double, 1, 4> nullSpace[3][3]) const {
				Eigen::Matrix<double, 10, 20> constraintMatrix;
				constraintMatrix.block<9, 20>(0, 0) = getTraceConstraint(nullSpace);
				constraintMatrix.row(9) = getDeterminantConstraint(nullSpace);
				return constraintMatrix;
			}
		}
	}
}