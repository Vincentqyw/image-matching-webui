// Copyright (c) 2020, Viktor Larsson
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

#pragma once

#include <Eigen/Dense>
#include <vector>
#include "bundle.h"

namespace pose_lib {

    // Computes the essential matrix from the camera motion
    void essential_from_motion(const CameraPose& pose, Eigen::Matrix3d* E);

    // Checks the cheirality of the point correspondences, i.e. that
    //    lambda_2 * x2 = R * ( lambda_1 * x1 ) + t
    // with lambda_1 and lambda_2 positive
    bool check_cheirality(const CameraPose& pose, const Eigen::Vector3d& x1, const Eigen::Vector3d& x2, double min_depth = 0.0);


    /**
    * @brief Given an essential matrix computes the 2 rotations and the 2 translations. The method also takes one point correspondence that is used to filter for cheirality.
    * that can generate four possible motions.
    * @param E Essential matrix
    * @param[out] relative_poses The 4 possible relative poses
    * @ref Multiple View Geometry - Richard Hartley, Andrew Zisserman - second edition
    * @see HZ 9.7 page 259 (Result 9.19)
       
    */
    void motion_from_essential_svd(const Eigen::Matrix3d& E, const Eigen::Vector3d& x1, const Eigen::Vector3d& x2, pose_lib::CameraPoseVector* relative_poses);

    /*
    Computes the factorization using the closed-form SVD suggested in 
       Nister, An Efficient Solution to the Five-Point Relative Pose Problem, PAMI 2004
    The method also takes one point correspondence that is used to filter for cheirality.
    */
    void motion_from_essential(const Eigen::Matrix3d& E, const Eigen::Vector3d &x1, const Eigen::Vector3d& x2, pose_lib::CameraPoseVector* relative_poses);

    /* 
    Factorizes the essential matrix into the relative poses. Assumes that the essential matrix corresponds to 
    planar motion, i.e. that we have      
          E = [0   e01  0;
               e10  0  e12;
               0   e21  0]

    Only returns the solution where the rotation is on the form
         R = [a 0 -b; 
             0  1  0;
             b  0  a];
    Note that there is another solution where the rotation is on the form
         R = [a 0   b;
             0  -1  0;
             b  0  -a];
    which is not returned!

    The method also takes one point correspondence that is used to filter for cheirality.
    */
    void motion_from_essential_planar(double e01, double e21, double e10, double e12, const Eigen::Vector3d& x1, const Eigen::Vector3d& x2, pose_lib::CameraPoseVector* relative_poses);

}