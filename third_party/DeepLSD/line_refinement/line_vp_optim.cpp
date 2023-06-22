/**
 *  Regress line endpoints from a line distance field with optimization.
 *  Can process each line independently or all together.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <array>
#include <algorithm>

#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "cost_functions.h"
#include "vp_det.h"

using namespace std;
using namespace ceres;
namespace py = pybind11;


static const int pad = 3;
static const double pad_val[pad] = {10, 5, 3};


/** Compute the cosine and sine of an angle array.
 */
void get_trigo(double* angles, size_t len, double* cos_angle, double* sin_angle)
{
    for(int i=0; i < len; i++)
    {
        cos_angle[i] = std::cos(angles[i]);
        sin_angle[i] = std::sin(angles[i]);
    }
}


/*------------------------- Line optimization ----------------------------*/

/** Pad and preprocess the line distance field.
 */
void preprocess_df(vector<double> df, double* df_pad, int rows, int cols)
{
    // Put high DF values as very far away
    for(int pos=0; pos < rows * cols; pos++)
    {
        if(df[pos] > 1.1)  // 1.5
            df_pad[((pos / cols) + pad) * (cols + 2 * pad)
                    + (pos % cols) + pad] = 2.;
        else
            df_pad[((pos / cols) + pad) * (cols + 2 * pad)
                    + (pos % cols) + pad] = df[pos];
    }

    // Pad df with high values on the boundary
    for(int x=0; x < cols + 2 * pad; x++)
    {
        for(int y=0; y < pad; y++)
        {
            df_pad[y * (cols + 2 * pad) + x] = pad_val[y];
            df_pad[(rows + 2 * pad - 1 - y) * (cols + 2 * pad)
                    + x] = pad_val[y];
        }
    }
    for(int y=0; y < rows + 2 * pad; y++)
    {
        for(int x=0; x < pad; x++)
        {
            df_pad[y * (cols + 2 * pad) + x] = pad_val[x];
            df_pad[y * (cols + 2 * pad)
                    + cols + 2 * pad - 1 - x] = pad_val[x];
        }
    }
}


/** Optimize a set of line endpoints based on a line distance function.
 */
tuple<vector<array<double, 4>>, vector<int>, vector<array<double, 3>>> optimize_lines(
    vector<array<double, 5>> lines, int rows, int cols,
    double* df, double* line_level, bool use_vps, bool optimize_vps, double lambda_df,
    double lambda_grad, double lambda_vp, const double &threshold, const size_t &max_iters,
	const size_t &minimum_point_number, const int &maximum_model_number,
	const double &scoring_exponent, bool verbose)
{
    // Initialize the bicubic interpolator on the DF
    Grid2D<double, 1> df_grid(df, -pad, rows + pad, -pad, cols + pad);
    BiCubicInterpolator<Grid2D<double, 1>> df_interpolator(df_grid);

    // Initialize the cosine and sine arrays
    size_t size = (size_t) (rows * cols);
    double* cos_angles = (double *) calloc(size, sizeof(double));
    double* sin_angles = (double *) calloc(size, sizeof(double));
    get_trigo(line_level, size, cos_angles, sin_angles);

    // Initialize the line segments
    const int num_lines = lines.size();
    vector<array<double, 4>> x(num_lines);
    for(int i = 0; i < num_lines; i++)
        x[i] = {lines[i][0], lines[i][1], lines[i][2], lines[i][3]};
    double c_x, c_y, ori, len, perp_dist;

    // Extract VPs
    vector<int> vp_labels;
    vector<array<double, 3> > vps;
    std::tie(vps, vp_labels) = compute_vps(
        x, cols, rows, threshold, max_iters, minimum_point_number,
        maximum_model_number, scoring_exponent);

    // Alternate between fitting VPs and optimizing the lines
    int num_iter = 5;
    // Set the solver properties up
    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = verbose;
    options.num_threads = 4;
    options.max_num_iterations = 20;
    if(!optimize_vps)
    {
        num_iter = 1;
        options.max_num_iterations = 100;
    }
    Solver::Summary summary;
    for(int it=0; it < num_iter; it++)
    {
        // Optimize the lines independently using the current VPs
        for(int i=0; i < num_lines; i++)
        {   
            /* Optimization with fixed line length and no tangential translation (2 DOFs) */
            c_x = (x[i][0] + x[i][2]) / 2.;
            c_y = (x[i][1] + x[i][3]) / 2.;
            perp_dist = 0.;
            len = std::sqrt(pow(x[i][2] - x[i][0], 2) + pow(x[i][3] - x[i][1], 2));
            ori = lines[i][4];

            // Define the problem
            Problem problem;

            // DF constraint
            LossFunction* df_loss = new ScaledLoss(
                NULL, lambda_df / (n_samples + 1),
                ceres::TAKE_OWNERSHIP);
            CostFunction* df_cost_function = DfCostFunctor::Create(
                df_interpolator, len, c_x, c_y);
            problem.AddResidualBlock(df_cost_function, df_loss, &perp_dist, &ori);

            // Add the angle constraint
            if(lambda_grad > 0)
            {
                LossFunction* grad_loss = new ScaledLoss(
                    NULL, lambda_grad / (n_samples - 1),
                    ceres::TAKE_OWNERSHIP);
                CostFunction* grad_cost_function = GradCostFunctor::Create(
                    cos_angles, sin_angles, rows, cols, len, c_x, c_y);
                problem.AddResidualBlock(grad_cost_function, grad_loss,
                                         &perp_dist, &ori);
            }

            // Optionally add the VP constraint
            if(use_vps && vp_labels[i] >= 0 && lambda_vp > 0)
            {
                LossFunction* vp_loss = new ScaledLoss(
                    NULL, lambda_vp, ceres::TAKE_OWNERSHIP);
                CostFunction* vp_cost_function = LineVpCostFunctor::Create(
                    len, c_x, c_y, &(vps[vp_labels[i]])[0]);
                problem.AddResidualBlock(vp_cost_function, vp_loss, &perp_dist, &ori);
            }

            // Solve the optimization problem
            Solve(options, &problem, &summary);
            x[i][0] = c_x + perp_dist * std::cos(ori + M_PI_2) + len / 2 * std::cos(ori);
            x[i][1] = c_y + perp_dist * std::sin(ori + M_PI_2) + len / 2 * std::sin(ori);
            x[i][2] = c_x + perp_dist * std::cos(ori + M_PI_2) - len / 2 * std::cos(ori);
            x[i][3] = c_y + perp_dist * std::sin(ori + M_PI_2) - len / 2 * std::sin(ori);
        }

        if(optimize_vps)
        {
            // Reassign VPs to the refined lines
            assign_vps(x, vps, vp_labels, threshold);

            // Optimize VPs based on the lines
            for(int i=0; i < vps.size(); i++)
            {
                // Define the problem
                Problem problem;

                // Gather all lines voting for the current VP
                for(int j=0; j < num_lines; j++)
                {
                    if(vp_labels[j] == i)
                    {
                        // Get the middle point
                        c_x = (x[j][0] + x[j][2]) / 2;
                        c_y = (x[j][1] + x[j][3]) / 2;
                        len = std::sqrt(pow(x[i][2] - x[i][0], 2) + pow(x[i][3] - x[i][1], 2));
                        LossFunction* vp_loss = new ScaledLoss(
                            new CauchyLoss(0.5), len, ceres::TAKE_OWNERSHIP);
                        CostFunction* vp_cost_function = VpCostFunctor::Create(
                            x[j][0], x[j][1], c_x, c_y);
                        problem.AddResidualBlock(vp_cost_function, vp_loss, &(vps[i])[0]);
                    }
                }
                if (problem.HasParameterBlock(&(vps[i])[0])) {
#ifdef CERES_PARAMETERIZATION_ENABLED
                    ceres::LocalParameterization* homo3d_parameterization = new ceres::HomogeneousVectorParameterization(3);
                    problem.SetParameterization(&(vps[i])[0], homo3d_parameterization);
#else
                    ceres::Manifold* homo3d_manifold = new ceres::SphereManifold<3>;
                    problem.SetManifold(&(vps[i])[0], homo3d_manifold);
#endif
                }
                // Solve the optimization problem
                Solve(options, &problem, &summary);
            }

            // Reassign the refined VPs to the lines
            assign_vps(x, vps, vp_labels, threshold);
        }
    }

    // Free memory
    free((void*) cos_angles);
    free((void*) sin_angles);

    return make_tuple(x, vp_labels, vps);
}


/** Convert a distance fields to a list of line segments.
 *  Args:
 *      lines: list of line segments defined by a pair of endpoints
               and an orientation.
 *      df: a flattened line distance field.
 *      line_level: line level angle orientation.
 *      rows, cols: original image dimensions.
 *      use_vps: true to use VPs in the optimization.
 *      optimize_vps: true to optimize the VPs as well.
 *      verbose: true to print optimization details.
 * Returns:
 *      A list of lines defined as [x1, y1, x2, y2].
 */
tuple<vector<array<double, 4>>, vector<int>, vector<array<double, 3>>> line_optim(
    vector<array<double, 5>> lines, vector<double> df, vector<double> line_level,
    int rows, int cols, bool use_vps, bool optimize_vps, double lambda_df,
    double lambda_grad, double lambda_vp, const double &threshold,
    const size_t &max_iters, const size_t &minimum_point_number,
    const int &maximum_model_number, const double &scoring_exponent, bool verbose)
{
    // Pad and preprocess the distance field
    double *df_pad = (double *) calloc((size_t) (
        (rows + 2 * pad) * (cols + 2 * pad)), sizeof(double));
    preprocess_df(df, df_pad, rows, cols);

    // Extract lines from the distance field
    tuple<vector<array<double, 4>>, vector<int>, vector<array<double, 3>>> out = optimize_lines(
        lines, rows, cols, df_pad, &line_level[0], use_vps, optimize_vps,
        lambda_df, lambda_grad, lambda_vp, threshold, max_iters,
	    minimum_point_number, maximum_model_number, scoring_exponent, verbose);

    // Free memory
    free((void*) df_pad);

    return out;
}


// Python bindings
PYBIND11_MODULE(line_refinement, m) {
    m.doc() = "LineRefinement";
    m.def("line_optim", &line_optim,
          "Optimize the endpoints of line segments given a distance and angle fields.",
          py::arg("lines"),
          py::arg("df"),
          py::arg("line_level"),
          py::arg("rows"),
          py::arg("cols"),
          py::arg("use_vps")=false,
          py::arg("optimize_vps")=false,
          py::arg("lambda_df")=1.,
          py::arg("lambda_grad")=1.,
          py::arg("lambda_vp")=0.5,
          py::arg("threshold")=1.,
          py::arg("max_iters")=100000,
          py::arg("minimum_point_number")=2,
          py::arg("maximum_model_number")=-1,
          py::arg("scoring_exponent")=1.,
          py::arg("verbose")=false,
          py::call_guard<py::scoped_ostream_redirect,
                         py::scoped_estream_redirect>());
    m.def("compute_vps", &compute_vps,
          "Compute vanishing points from line segments.",
          py::arg("lines"),
          py::arg("w"),
          py::arg("h"),
          py::arg("threshold")=1.5,
          py::arg("max_iters")=100000,
          py::arg("minimum_point_number")=2,
          py::arg("maximum_model_number")=-1,
          py::arg("scoring_exponent")=1.,
          py::arg("conf")=0.99,
          py::arg("spatial_coherence_weight")=0.,
          py::arg("neighborhood_ball_radius")=1.,
          py::arg("maximum_tanimoto_similarity")=1.,
          py::arg("sampler_id")=0,
          py::arg("do_logging")=false,
          py::arg("min_num_supports")=10,
          py::call_guard<py::scoped_ostream_redirect,
                         py::scoped_estream_redirect>());
}
