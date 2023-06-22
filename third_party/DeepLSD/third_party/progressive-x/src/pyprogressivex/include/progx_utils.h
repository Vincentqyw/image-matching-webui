#pragma once

#include <algorithm>
#include <fstream>
#include <functional>
#include <sstream>
#include <vector>
#include <iterator>
#include <set>

#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Eigen>

#include "progx_model.h"

/****************************************
************** Declaration **************
****************************************/
bool loadPointsWithLabels(
	cv::Mat &points_,
	std::vector<size_t> &labels_,
	size_t &reference_model_number_,
	const char* file_);

/*****************************************
************* Implementation *************
*****************************************/
bool loadPointsWithLabels(
	cv::Mat &points_,
	std::vector<size_t> &labels_,
	size_t &reference_model_number_,
	const char* file_)
{
	std::ifstream infile(file_);

	if (!infile.is_open())
	{
		fprintf(stderr, "A problem occured when loading the points from file '%s'. The file does not exist.", file_);
		return false;
	}

	std::string line;
	std::vector<std::vector<double>> loaded_coordinates;

	while (getline(infile, line))
	{
		std::vector<double> split_elements;
		std::istringstream split(line);
		double element;
		size_t column_idx = 0;

		while (split >> element)
		{
			++column_idx;
			// Skip the homogeneous coordinates (i.e., ones)
			if (column_idx == 3 || column_idx == 6)
				continue;
			split_elements.emplace_back(element);
		}

		if (loaded_coordinates.size() > 0 &&
			loaded_coordinates.back().size() != split_elements.size())
		{
			fprintf(stderr, "A problem occured when loading the points from file '%s'. The number of coordinates varies in the rows.", file_);
			return false;
		}

		loaded_coordinates.emplace_back(split_elements);
	}
	infile.close();

	const size_t point_number = loaded_coordinates.size(); // The number of points
	const size_t coordinate_number = loaded_coordinates[0].size() - 1; // The number of columns. The last one is consists of the labels

	points_.create(point_number, // The number of points
		coordinate_number, // The number of columns. The last one is consists of the labels
		CV_64F); // The type of the data
	double *points_ptr = reinterpret_cast<double *>(points_.data);

	labels_.resize(point_number);
	reference_model_number_ = 0;

	for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
	{
		for (size_t coordinate_idx = 0; coordinate_idx < coordinate_number; ++coordinate_idx)
			*(points_ptr++) = loaded_coordinates[point_idx][coordinate_idx];
		labels_[point_idx] = loaded_coordinates[point_idx].back();
		reference_model_number_ = MAX(reference_model_number_, labels_[point_idx]);
	}

	return true;
}

template <typename _ModelEstimator>
inline double getMisclassificationError(
	std::vector< progx::Model<_ModelEstimator> > const &models,
	std::vector<size_t> const &annotation, 
	int K, 
	int K_annot)
{
	if (K > 9)
	{
		fprintf(stderr, "The misclassification error calculation would take too long time since there are more than 9 model instances (%d)", K);
		return -1;
	}

	const int max_allowed_K = 9;
	const int max_allowed_perm = 1000000;

	const int min_K = MIN(K, K_annot);
	const int max_K = MAX(K, K_annot);
	const double probability_of_choosing = 0.5 * static_cast<double>(max_allowed_K * max_allowed_K) / (max_K * max_K);
	int *indices = new int[max_K];
	for (int i = 0; i < max_K; ++i)
		indices[i] = i;

	std::vector<std::vector<short>> permutations;
	do {
		if (max_K > max_allowed_K && static_cast<double>(rand()) / RAND_MAX > 1 - probability_of_choosing)
			continue;

		permutations.push_back(std::vector<short>(max_K));
		for (int i = 0; i < max_K; ++i)
			permutations.back()[max_K - i - 1] = max_K - indices[i] - 1;
	} while (std::next_permutation(indices, indices + max_K) && permutations.size() < max_allowed_perm);
	std::vector<std::vector<int>> preferences(K, std::vector<int>(annotation.size(), 0));
	for (int i = 0; i < models.size(); ++i)
	{
		for (int j = 0; j < annotation.size(); ++j)
			if (models[i].preference_vector(j) > FLT_EPSILON)
				preferences[i][j] = 1;
	}

	int *linking = new int[min_K];
	int best_error = INT_MAX;
	for (int i = 0; i < permutations.size(); ++i)
	{
		//vector<vector<int>> temp_preferences = preferences;

		//if (K <= K_annot)
		{
			std::vector<std::vector<int>> weak_labeling(annotation.size());
			std::vector<double> prefs(annotation.size(), 0);

			for (int m = 0; m < models.size(); ++m)
			{
				int m_idx = permutations[i][m] + 1;
				for (int j = 0; j < annotation.size(); ++j)
				{					
					if (preferences[m][j] > prefs[j])
					{
						prefs[j] = preferences[m][j];
						if (weak_labeling[j].size() == 0)
							weak_labeling[j].push_back(m_idx);
						else
							weak_labeling[j][0] = m_idx;
					}
				}
			}

			int error = 0;
			for (int j = 0; j < annotation.size(); ++j)
			{
				if (weak_labeling[j].size() == 0 && annotation[j] == 0)
					continue;

				bool found = false;
				for (int w = 0; w < weak_labeling[j].size(); ++w)
				{
					if (weak_labeling[j][w] == annotation[j])
					{
						found = true;
						break;
					}
				}

				if (!found)
					++error;
			}

			best_error = MIN(best_error, error);
		}

	}
	delete linking;

	return 100 * static_cast<double>(best_error) / annotation.size();
}

inline double getMisclassificationError(
	std::vector<int> const &labeling, 
	std::vector<int> const &annotation,
	int K, 
	int K_annot)
{
	std::vector<std::set<int>> obtainedClusters(K + 1);
	std::vector<std::set<int>> gtClusters(K_annot + 1);

	std::vector<int> gtPair(K + 1, 0);
	const int N = labeling.size();

	std::vector<int> labelsAll1(N, -1);
	std::vector<int> labelsAll2(N, -1);

	for (int i = 0; i < labeling.size(); ++i)
	{
		if (labeling[i] + 1 > K)
			continue;

		labelsAll1[i] = labeling[i] + 1;
		obtainedClusters[labeling[i] + 1].insert(i);
	}

	for (int i = 0; i < N; ++i)
	{
		labelsAll2[i] = annotation[i];
		gtClusters[annotation[i]].insert(i);
	}

	// Find pairs
	std::vector<bool> usedMask(obtainedClusters.size(), false);
	for (int i = 1; i < gtClusters.size(); ++i)
	{
		int bestCluster = -1;
		int bestSize = -1;

		for (int j = 1; j < obtainedClusters.size(); ++j)
		{
			if (usedMask[j])
				continue;

			std::set<int> intersect;
			std::set_intersection(obtainedClusters[j].begin(), obtainedClusters[j].end(), gtClusters[i].begin(), gtClusters[i].end(),
				std::inserter(intersect, intersect.begin()));

			if (bestCluster == -1 || bestSize < intersect.size())
			{
				bestSize = intersect.size();
				bestCluster = j;
			}
		}

		if (bestCluster == -1)
			continue;

		usedMask[bestCluster] = true;
		gtPair[i] = bestCluster;
	}

	for (int j = 0; j < labeling.size(); ++j)
	{
		if (labeling[j] != -1 && labelsAll1[j] == gtPair[labelsAll2[j]])
		{
			labelsAll1[j] = labelsAll2[j];
		}
	}

	int error = 0;
	for (int i = 0; i < labelsAll1.size(); ++i)
	{
		//cout << labelsAll1[i] << " " << labelsAll2[i] << endl;
		if (labelsAll1[i] != labelsAll2[i])
		{
			++error;
		}
	}

	return 100.0 * (error / (double)labelsAll1.size());

}

