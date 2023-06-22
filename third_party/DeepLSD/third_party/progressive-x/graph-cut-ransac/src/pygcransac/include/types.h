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
#include <numeric>

#include "estimators/fundamental_estimator.h"
#include "estimators/homography_estimator.h"
#include "estimators/radial_homography_estimator.h"
#include "estimators/essential_estimator.h"
#include "estimators/perspective_n_point_estimator.h"
#include "estimators/rigid_transformation_estimator.h"
#include "estimators/linear_model_estimator.h"

#include "estimators/solver_fundamental_matrix_seven_point.h"
#include "estimators/solver_fundamental_matrix_eight_point.h"
#include "estimators/solver_fundamental_matrix_bundle_adjustment.h"
#include "estimators/solver_p3p.h"
#include "estimators/solver_epnp_lm.h"
#include "estimators/solver_dls_pnp.h"
#include "estimators/solver_essential_matrix_five_point_stewenius.h"
#include "estimators/solver_essential_matrix_bundle_adjustment.h"
#include "estimators/solver_rigid_transformation_svd.h"
#include "estimators/solver_linear_model.h"
#include "estimators/solver_homography_four_point.h"
#include "estimators/solver_radial_homography_5pc.h"
#include "estimators/solver_radial_homography_6pc.h"

namespace gcransac
{
	namespace utils
	{
		// The default estimator for fundamental matrix fitting
		typedef estimator::FundamentalMatrixEstimator<estimator::solver::FundamentalMatrixSevenPointSolver, // The solver used for fitting a model to a minimal sample
			estimator::solver::FundamentalMatrixBundleAdjustmentSolver> // The solver used for fitting a model to a non-minimal sample
			DefaultFundamentalMatrixEstimator;

		// The default estimator for homography fitting
		typedef estimator::RobustHomographyEstimator<estimator::solver::HomographyFourPointSolver, // The solver used for fitting a model to a minimal sample
			estimator::solver::HomographyFourPointSolver> // The solver used for fitting a model to a non-minimal sample
			DefaultHomographyEstimator;

		// The default estimator for essential matrix fitting
		typedef estimator::EssentialMatrixEstimator<estimator::solver::EssentialMatrixFivePointSteweniusSolver, // The solver used for fitting a model to a minimal sample
			estimator::solver::EssentialMatrixBundleAdjustmentSolver> // The solver used for fitting a model to a non-minimal sample
			DefaultEssentialMatrixEstimator;

		// The default estimator for PnP fitting
		typedef estimator::PerspectiveNPointEstimator<estimator::solver::P3PSolver, // The solver used for fitting a model to a minimal sample
			estimator::solver::EPnPLM> // The solver used for fitting a model to a non-minimal sample
			DefaultPnPEstimator;

		// The default estimator for PnP fitting
		typedef estimator::RigidTransformationEstimator<estimator::solver::RigidTransformationSVDBasedSolver, // The solver used for fitting a model to a minimal sample
			estimator::solver::RigidTransformationSVDBasedSolver> // The solver used for fitting a model to a non-minimal sample
			DefaultRigidTransformationEstimator;

		// The default estimator for 2D line fitting
		typedef estimator::LinearModelEstimator<estimator::solver::LinearModelSolver<2>, // The solver used for fitting a model to a minimal sample
			estimator::solver::LinearModelSolver<2>,  // The solver used for fitting a model to a non-minimal sample
			2> // The dimensionality of the problem
			Default2DLineEstimator;

		// The default estimator for 3D plane fitting
		typedef estimator::LinearModelEstimator<estimator::solver::LinearModelSolver<3>, // The solver used for fitting a model to a minimal sample
			estimator::solver::LinearModelSolver<3>,  // The solver used for fitting a model to a non-minimal sample
			3> // The dimensionality of the problem
			Default3DPlaneEstimator;

		// The default estimator for radial distortion homography fitting
		typedef estimator::RadialHomographyEstimator<estimator::solver::RadialHomography5PC, // The solver used for fitting a model to a minimal sample
			estimator::solver::RadialHomography6PC>  // The solver used for fitting a model to a non-minimal sample
			DefaultRadialHomographyEstimator;
	}
}