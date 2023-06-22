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

#include <Eigen/Eigen>
#include "solver_engine.h"
#include "fundamental_estimator.h"
#include "unsupported/Eigen/Polynomials"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class RadialHomography5PC : public SolverEngine
			{
			public:
				RadialHomography5PC() : K1(Eigen::Matrix3d::Identity()),
										K2(Eigen::Matrix3d::Identity())
				{
				}

				// Copy constructor 
				RadialHomography5PC(const RadialHomography5PC &solver_)
				{
					K1 = solver_.K1; 
					K2 = solver_.K2;
				}

				~RadialHomography5PC()
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

				static constexpr char * getName()
				{
					return "H5l1l2";
				}

				// The maximum number of returned solutions
				static constexpr size_t maximumSolutions()
				{
					return 3;
				}

				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				OLGA_INLINE bool estimateModel(
					const cv::Mat &data_,					 // The set of data points
					const size_t *sample_,					 // The sample used for the estimation
					size_t sample_number_,					 // The size of the sample
					std::vector<Model> &models_,			 // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point

				void setIntrinsics(
					const Eigen::Matrix3d& K1_,
					const Eigen::Matrix3d& K2_)
				{
					K1 = K1_;
					K2 = K2_;
				}

			protected:
				Eigen::Matrix3d K1, K2;

				OLGA_INLINE int h5gb(Eigen::Matrix<double, 31, 1> &params, Eigen::Matrix<double, 3, 5> &sols, double pivtol = 1e-16) const;

				template <typename Derived>
				OLGA_INLINE void colEchelonForm(Eigen::MatrixBase<Derived> &M, double pivtol = 1e-12) const;
			};

			template <typename Derived>
			OLGA_INLINE void RadialHomography5PC::colEchelonForm(Eigen::MatrixBase<Derived> &M, double pivtol) const
			{
				typedef typename Derived::Scalar Scalar;

				int n = M.rows();
				int m = M.cols();
				int i = 0, j = 0, k = 0;
				int col = 0;
				Scalar p, tp;

				while ((i < m) && (j < n))
				{
					p = std::numeric_limits<Scalar>::min();
					col = i;

					for (k = i; k < m; k++)
					{
						tp = std::abs(M(j, k));
						if (tp > p)
						{
							p = tp;
							col = k;
						}
					}

					if (p < Scalar(pivtol))
					{
						M.block(j, i, 1, m - i).setZero();
						j++;
					}
					else
					{
						if (col != i)
							M.block(j, i, n - j, 1).swap(M.block(j, col, n - j, 1));

						M.block(j + 1, i, n - j - 1, 1) /= M(j, i);
						M(j, i) = 1.0;

						for (k = 0; k < m; k++)
						{
							if (k == i)
								continue;

							M.block(j, k, n - j, 1) -= M(j, k) * M.block(j, i, n - j, 1);
						}

						i++;
						j++;
					}
				}
			}

			OLGA_INLINE int RadialHomography5PC::h5gb(Eigen::Matrix<double, 31, 1> &params, Eigen::Matrix<double, 3, 5> &sols, double pivtol) const
			{
				Eigen::Matrix<double, 24, 1> c;

				c(0) = params(0);
				c(1) = params(1);
				c(2) = params(2);
				c(3) = params(30)*params(1);
				c(4) = -params(6);
				c(5) = params(4) - params(7);
				c(6) = -params(8);
				c(7) = -params(30)*params(7) + params(5);
				c(8) = -params(10);
				c(9) = -params(11);
				c(10) = params(25);
				c(11) = params(30)*params(25);
				c(12) = -params(12) + params(28);
				c(13) = -params(13);
				c(14) = params(29) - params(14);
				c(15) = -params(30)*params(13);
				c(16) = -params(16);
				c(17) = -params(17);
				c(18) = params(25);
				c(19) = params(30)*params(25);
				c(20) = -params(19) + params(28);
				c(21) = -params(21) + params(29);
				c(22) = -params(22);
				c(23) = -params(23);

				Eigen::Matrix<double, 21, 16> M;
				M.setZero();

				M(211) = c(0);  M(87) = c(0);  M(68) = c(0);
				M(8) = c(0);  M(212) = c(1);  M(88) = c(1);
				M(69) = c(1);  M(9) = c(1);  M(213) = c(2);
				M(91) = c(2);  M(71) = c(2);  M(12) = c(2);
				M(214) = c(3);  M(72) = c(3);  M(13) = c(3);
				M(215) = c(4);  M(92) = c(4);  M(73) = c(4);
				M(14) = c(4);  M(216) = c(5);  M(93) = c(5);
				M(74) = c(5);  M(15) = c(5);  M(218) = c(6);
				M(96) = c(6);  M(77) = c(6);  M(17) = c(6);
				M(219) = c(7);  M(97) = c(7);  M(78) = c(7);
				M(18) = c(7);  M(221) = c(8);  M(99) = c(8);
				M(79) = c(8);  M(19) = c(8);  M(225) = c(9);
				M(102) = c(9);  M(82) = c(9);  M(20) = c(9);
				M(252) = c(10);  M(232) = c(10);  M(129) = c(10);
				M(110) = c(10);  M(29) = c(10);  M(234) = c(11);
				M(133) = c(11);  M(113) = c(11);  M(33) = c(11);
				M(255) = c(12);  M(236) = c(12);  M(134) = c(12);
				M(115) = c(12);  M(35) = c(12);  M(256) = c(13);
				M(237) = c(13);  M(135) = c(13);  M(116) = c(13);
				M(36) = c(13);  M(259) = c(14);  M(239) = c(14);
				M(138) = c(14);  M(119) = c(14);  M(38) = c(14);
				M(240) = c(15);  M(139) = c(15);  M(120) = c(15);
				M(39) = c(15);  M(261) = c(16);  M(242) = c(16);
				M(141) = c(16);  M(121) = c(16);  M(40) = c(16);
				M(265) = c(17);  M(246) = c(17);  M(144) = c(17);
				M(124) = c(17);  M(41) = c(17);  M(315) = c(18);
				M(295) = c(18);  M(275) = c(18);  M(192) = c(18);
				M(172) = c(18);  M(153) = c(18);  M(51) = c(18);
				M(297) = c(19);  M(277) = c(19);  M(196) = c(19);
				M(156) = c(19);  M(55) = c(19);  M(318) = c(20);
				M(299) = c(20);  M(279) = c(20);  M(197) = c(20);
				M(177) = c(20);  M(158) = c(20);  M(57) = c(20);
				M(322) = c(21);  M(302) = c(21);  M(282) = c(21);
				M(201) = c(21);  M(181) = c(21);  M(162) = c(21);
				M(60) = c(21);  M(323) = c(22);  M(304) = c(22);
				M(284) = c(22);  M(203) = c(22);  M(183) = c(22);
				M(163) = c(22);  M(61) = c(22);  M(327) = c(23);
				M(308) = c(23);  M(288) = c(23);  M(206) = c(23);
				M(186) = c(23);  M(166) = c(23);  M(62) = c(23);

				colEchelonForm(M, pivtol);

				Eigen::Matrix<double, 5, 5> A;
				A.setZero();

				A(0, 2) = 1.0000000;  A(1, 0) = -M(20, 15);  A(1, 1) = -M(19, 15);
				A(1, 2) = -M(18, 15);  A(1, 3) = -M(17, 15);  A(1, 4) = -M(16, 15);
				A(2, 0) = -M(20, 13);  A(2, 1) = -M(19, 13);  A(2, 2) = -M(18, 13);
				A(2, 3) = -M(17, 13);  A(2, 4) = -M(16, 13);  A(3, 0) = -M(20, 12);
				A(3, 1) = -M(19, 12);  A(3, 2) = -M(18, 12);  A(3, 3) = -M(17, 12);
				A(3, 4) = -M(16, 12);  A(4, 0) = -M(20, 11);  A(4, 1) = -M(19, 11);
				A(4, 2) = -M(18, 11);  A(4, 3) = -M(17, 11);  A(4, 4) = -M(16, 11);

				Eigen::EigenSolver<Eigen::Matrix<double, 5, 5> > eig(A);
				Eigen::Matrix<std::complex<double>, 5, 3> esols;
				esols.col(0).array() = eig.eigenvectors().row(3).array() / eig.eigenvectors().row(0).array();
				esols.col(1).array() = eig.eigenvectors().row(2).array() / eig.eigenvectors().row(0).array();
				esols.col(2).array() = eig.eigenvectors().row(1).array() / eig.eigenvectors().row(0).array();

				int nsols = 0;
				for (int i = 0; i < 5; i++)
				{
					if (esols.row(i).imag().isZero(100.0 * std::numeric_limits<double>::epsilon()))
						sols.col(nsols++) = esols.row(i).real();
				}

				return nsols;
			}

			OLGA_INLINE bool RadialHomography5PC::estimateModel(
				const cv::Mat &data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				using namespace Eigen;

				constexpr double pivtol = 1e-16;

				// Building the coefficient matrices
				const double *data_ptr = reinterpret_cast<double *>(data_.data);
				const size_t cols = data_.cols;
				const size_t pointNumber = sampleSize(); // 3

				// this solver is stable
				Eigen::Matrix<double, 5, 2> X, U;
				for (size_t sampleIdx = 0; sampleIdx < pointNumber; ++sampleIdx)
				{
					const auto &pointIdx = sample_[sampleIdx];
					X(sampleIdx, 0) = data_.at<double>(pointIdx, 0);
					X(sampleIdx, 1) = data_.at<double>(pointIdx, 1);
					U(sampleIdx, 0) = data_.at<double>(pointIdx, 2);
					U(sampleIdx, 1) = data_.at<double>(pointIdx, 3);
				}

				Eigen::Matrix<double, 5, 8> M;
				const Eigen::Array<double, 5, 1> &x0 = X.col(0).array();
				const Eigen::Array<double, 5, 1> &x1 = X.col(1).array();
				const Eigen::Array<double, 5, 1> &u0 = U.col(0).array();
				const Eigen::Array<double, 5, 1> &u1 = U.col(1).array();

				Eigen::Array<double, 5, 1> x2 = x0.square() + x1.square();
				Eigen::Array<double, 5, 1> u2 = u0.square() + u1.square();
				Eigen::Array<double, 5, 1> ones, zeros;
				ones.setOnes();
				zeros.setZero();

				M.col(0) = -x1 * u0;
				M.col(1) = -x1 * u1;
				M.col(2) = -x1;
				M.col(3) = x0 * u0;
				M.col(4) = x0 * u1;
				M.col(5) = x0;
				M.col(6) = -x1 * u2;
				M.col(7) = x0 * u2;
				
				Eigen::JacobiSVD<Eigen::Matrix<double, 5, 8>, Eigen::FullPivHouseholderQRPreconditioner> Svd(M, Eigen::ComputeFullV);
				const Eigen::Matrix<double, 8, 8> &N = Svd.matrixV();

				Eigen::Array<double, 5, 1> tmp0 = N(0, 5) * u0 + N(1, 5) * u1 + N(2, 5) * ones + N(6, 5) * u2;
				Eigen::Array<double, 5, 1> tmp1 = N(0, 6) * u0 + N(1, 6) * u1 + N(2, 6) * ones + N(6, 6) * u2;
				Eigen::Array<double, 5, 1> tmp2 = N(0, 7) * u0 + N(1, 7) * u1 + N(2, 7) * ones + N(6, 7) * u2;

				Eigen::Matrix<double, 7, 13> M2;
				Eigen::Matrix<double, 13, 7> M3;
				M2.template block<5, 1>(0, 0) = -M.col(3);
				M2.template block<5, 1>(0, 1) = -M.col(4);
				M2.template block<5, 1>(0, 2) = -x0;
				M2.template block<5, 1>(0, 3) = -M.col(7);
				M2.template block<5, 1>(0, 4) = x2 * tmp0;
				M2.template block<5, 1>(0, 5) = zeros;
				M2.template block<5, 1>(0, 6) = tmp0;
				M2.template block<5, 1>(0, 7) = x2 * tmp1;
				M2.template block<5, 1>(0, 8) = zeros;
				M2.template block<5, 1>(0, 9) = x2 * tmp2;
				M2.template block<5, 1>(0, 10) = zeros;
				M2.template block<5, 1>(0, 11) = tmp1;
				M2.template block<5, 1>(0, 12) = tmp2;

				M2.row(5) << 0, 0, 0, 0, 0, -N(2, 5), N(6, 5), 0, -N(2, 6), 0, -N(2, 7), N(6, 6), N(6, 7);
				M2.row(6) << 0, 0, 0, 0, 0, -N(5, 5), N(7, 5), 0, -N(5, 6), 0, -N(5, 7), N(7, 6), N(7, 7);

				M3 = M2.transpose();
				colEchelonForm(M3, pivtol);

				if (abs(M3(8, 2)) < std::numeric_limits<double>::epsilon())
					return 0;

				Eigen::Matrix<double, 31, 1> gbparams;
				Eigen::Matrix<double, 3, 5> gbsols;
				gbparams(30) = M3(10, 2) / M3(8, 2);
				gbparams.template segment<6>(0) = M3.template block<6, 1>(7, 2);
				gbparams.template segment<6>(6) = M3.template block<6, 1>(7, 3);
				gbparams.template segment<6>(12) = M3.template block<6, 1>(7, 4);
				gbparams.template segment<6>(18) = M3.template block<6, 1>(7, 5);
				gbparams.template segment<6>(24) = M3.template block<6, 1>(7, 6);

				int nsols = h5gb(gbparams, gbsols, pivtol);
				
				Eigen::Matrix<double, 3, 1> v;
				Eigen::Matrix<double, 6, 1> m;
				models_.reserve(nsols);
				for (int i = 0; i < nsols; i++)
				{
					RadialHomography model;
					model.descriptor(0, 3) = gbsols(0, i);
					model.descriptor(1, 3) = gbsols(1, i);

					if (model.descriptor(0, 3) < -10 || model.descriptor(0, 3) > 2)
						continue;
					if (model.descriptor(1, 3) < -10 || model.descriptor(1, 3) > 2)
						continue;

					double &b = gbsols(2, i);
					m << gbsols(0, i) * gbsols(2, i), gbsols(1, i) * gbsols(2, i), gbsols(0, i), gbsols(1, i), gbsols(2, i), 1;
					v << m.dot(-M3. template block<6, 1>(7, 6)), gbsols(2, i), 1;
					
					model.descriptor(0, 0) = v.dot(N.block<1, 3>(0, 5));
					model.descriptor(1, 0) = v.dot(N.block<1, 3>(1, 5));
					model.descriptor(2, 0) = v.dot(N.block<1, 3>(2, 5));
					model.descriptor(0, 1) = v.dot(N.block<1, 3>(3, 5));
					model.descriptor(1, 1) = v.dot(N.block<1, 3>(4, 5));
					model.descriptor(2, 1) = v.dot(N.block<1, 3>(5, 5));
					model.descriptor(0, 2) = m.dot(-M3.block<6, 1>(7, 0));
					model.descriptor(1, 2) = m.dot(-M3.block<6, 1>(7, 1));
					model.descriptor(2, 2) = m.dot(-M3.block<6, 1>(7, 2));

					model.descriptor.block<3, 3>(0, 0).transposeInPlace();
					model.descriptor.block<3, 3>(0, 4) = model.descriptor.block<3, 3>(0, 0);
					model.descriptor.block<3, 3>(0, 0) = model.descriptor.block<3, 3>(0, 0).inverse().eval();

					models_.emplace_back(model);
				}

				return models_.size() > 0;
			}

		} // namespace solver
	}	  // namespace estimator
} // namespace gcransac
