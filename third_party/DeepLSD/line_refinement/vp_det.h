// #ifndef VP_DET_H
// #define VP_DET_H

#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <array>
#include <algorithm>

#include "progressivex_python.h"
#include "progressivex_python.cpp"


/*---------------------- Vanishing point estimation ----------------------*/

/** Compute VPs from a set of line segments, with Progressive-X.
 */
pair<vector<array<double, 3> >, vector<int> > compute_vps(
    const vector<array<double, 4> >& lines, const size_t &w, const size_t &h,
    const double &threshold = 1.5,
	const size_t &max_iters = 100000,
	const size_t &minimum_point_number = 2,
	const int &maximum_model_number = -1,
	const double &scoring_exponent = 1.,
    const double &conf = 0.99,
    const double &spatial_coherence_weight = 0.,
    const double &neighborhood_ball_radius = 1.,
    const double &maximum_tanimoto_similarity = 1.,
	const size_t &sampler_id = 0,
	const bool do_logging = false,
    const int &min_num_supports = 10)
{
    int num_lines = lines.size();
    vector<double> lines_flat(num_lines * 4);
	for(int i=0; i < num_lines; i++)
    {
        for(int j=0; j < 4; j++)
            lines_flat[i * 4 + j] = lines[i][j];
    }
    vector<double> vanishingPoints;	
	vector<size_t> labeling(num_lines);

    // Compute weights proportional to the line length
    vector<double> weights(num_lines);
    double max_w = 0.;
    for(int i=0; i < num_lines; i++)
    {
        weights[i] = sqrt(pow(lines[i][0] - lines[i][2], 2.)
                          + pow(lines[i][1] - lines[i][3], 2.));
        max_w = max(max_w, weights[i]);
    }
    for(int i=0; i < num_lines; i++)
        weights[i] /= max_w;

    int num_models = findVanishingPoints_(
		lines_flat, weights, labeling, vanishingPoints, w, h,
        spatial_coherence_weight, threshold, conf, neighborhood_ball_radius,
        maximum_tanimoto_similarity, max_iters, minimum_point_number,
		maximum_model_number, sampler_id, scoring_exponent, do_logging);

    // Count the number of inliers for each VP
    // and keep only VPs with enough support
    vector<int> counts(num_models, 0);
    for(int i=0; i < num_lines; i++)
        if((int) labeling[i] < num_models)
            counts[(int) labeling[i]]++;
    vector<int> reassign(num_models + 1, -1);
    int counter = 0;
    for(int i=0; i < num_models; i++)
    {
        if(counts[i] >= min_num_supports)
        {
            reassign[i] = counter;
            counter++;
        }
    }

    // Convert to the output format
    vector<int> labels(num_lines);
    for(int i = 0; i < num_lines; i++)
        labels[i] = reassign[(int) labeling[i]];
    vector<array<double, 3> > vps(counter);
    for(int i = 0; i < num_models; i++)
    {
        if(reassign[i] >= 0)
            vps[reassign[i]] = {vanishingPoints[3 * i],
                                vanishingPoints[3 * i + 1],
                                vanishingPoints[3 * i + 2]};
    }
    return std::make_pair(vps, labels);
}


/** Compute the distance of a line to a VP.
 */
double dist_line_vp(const array<double, 4> &line, const array<double, 3> &vp)
{
    // Middle point
    double cx = (line[0] + line[2]) / 2;
    double cy = (line[1] + line[3]) / 2;

    // Line from the mid point to the VP
    double l1 = cy * vp[2] - vp[1];
    double l2 = vp[0] - cx * vp[2];
    double l3 = cx * vp[1] - cy * vp[0];

    // Dist = max orthogonal distance of the two endpoints to l
    double dist = (std::abs(line[0] * l1 + line[1] * l2 + l3)
                   / std::sqrt(l1 * l1 + l2 * l2));
    return dist;
}


/** Re-assign VPs to a set of lines.
 */
void assign_vps(const vector<array<double, 4> > &lines, const vector<array<double, 3> > &vps,
                vector<int> &vp_labels, double tol=1.5)
{
    int num_lines = lines.size();
    double min_dist, dist;
    int min_idx;
    for(int i=0; i < num_lines; i++)
    {
        min_idx = -1;
        min_dist = tol;
        for(int j=0; j < vps.size(); j++)
        {
            dist = dist_line_vp(lines[i], vps[j]);
            if(dist < min_dist)
            {
                min_dist = dist;
                min_idx = j;
            }
        }
        vp_labels[i] = min_idx;
    }
}

// #endif
