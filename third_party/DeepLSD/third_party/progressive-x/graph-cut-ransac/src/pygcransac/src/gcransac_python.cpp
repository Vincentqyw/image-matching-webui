#include "gcransac_python.h"
#include <vector>
#include <thread>
#include "utils.h"
#include <opencv2/core/core.hpp>
#include <Eigen/Eigen>

#include "GCRANSAC.h"
#include "neighborhood/flann_neighborhood_graph.h"
#include "neighborhood/grid_neighborhood_graph.h"

#include "samplers/uniform_sampler.h"
#include "samplers/prosac_sampler.h"
#include "samplers/progressive_napsac_sampler.h"
#include "samplers/importance_sampler.h"
#include "samplers/adaptive_reordering_sampler.h"

#include "estimators/fundamental_estimator.h"
#include "estimators/homography_estimator.h"
#include "estimators/essential_estimator.h"

#include "preemption/preemption_sprt.h"

#include "inlier_selectors/empty_inlier_selector.h"
#include "inlier_selectors/space_partitioning_ransac.h"

#include "estimators/solver_fundamental_matrix_seven_point.h"
#include "estimators/solver_fundamental_matrix_eight_point.h"
#include "estimators/solver_homography_four_point.h"
#include "estimators/solver_essential_matrix_five_point_stewenius.h"

#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>

using namespace gcransac;

 int findLine2D_(std::vector<double>& input_points,
    			std::vector<double> &point_probabilities,
				std::vector<bool>& inliers,
				std::vector<double>& estimated_line,
				int w, int h,
				double threshold,
				double conf,
				int max_iters,
				double spatial_coherence_weight,
				bool use_sprt,
				double min_inlier_ratio_for_sprt,
				int sampler_id,
				int neighborhood_id,
				double neighborhood_size)
{
	// The number of points provided
	const size_t &num_points = input_points.size() / 2;

	// The matrix containing the points that will be passed to GC-RANSAC
	cv::Mat points(num_points, 2, CV_64F, &input_points[0]);

	// Initializing the neighborhood structure based on the provided paramereters
	typedef neighborhood::NeighborhoodGraph<cv::Mat> AbstractNeighborhood;
	std::unique_ptr<AbstractNeighborhood> neighborhood_graph;

	// The cell size or radius-search radius of the neighborhood graph
	const size_t cell_number_in_neighborhood_graph_ = 
		static_cast<size_t>(neighborhood_size);

	// If the spatial weight is 0.0, the neighborhood graph should not be created 
	if (spatial_coherence_weight <= std::numeric_limits<double>::epsilon())
	{
		cv::Mat emptyPoints(0, 2, CV_64F);

		neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
			new neighborhood::GridNeighborhoodGraph<4>(&emptyPoints, // The input points
			{ 0, // The cell size along axis X
				0 }, // The cell size along axis Y
			1)); // The cell number along every axis
	} else // Initializing a grid-based neighborhood graph
	{
		if (neighborhood_id == 0)
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::GridNeighborhoodGraph<4>(&points, // The input points
				{ w / static_cast<double>(cell_number_in_neighborhood_graph_), // The cell size along axis X
					h / static_cast<double>(cell_number_in_neighborhood_graph_) }, // The cell size along axis Y
				cell_number_in_neighborhood_graph_)); // The cell number along every axis
		else if (neighborhood_id == 1) // Initializing the neighbhood graph by FLANN
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::FlannNeighborhoodGraph(&points, neighborhood_size));
		else
		{
			fprintf(stderr, "Unknown neighborhood-graph identifier: %d. The accepted values are 0 (Grid-based), 1 (FLANN-based neighborhood)\n",
				neighborhood_id);
			return 0;
		}
	}

	// Checking if the neighborhood graph is initialized successfully.
	if (!neighborhood_graph->isInitialized())
	{
		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		fprintf(stderr, "The neighborhood graph is not initialized successfully.\n");
		return 0;
	}

	// Initializing the line estimator
	utils::Default2DLineEstimator estimator;

	// Initializing the model object
	Line2D model;

	// Initialize the samplers
	// The main sampler is used for sampling in the main RANSAC loop
	typedef sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Initializing a RANSAC-like uniformly random sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::UniformSampler(&points));
	else if (sampler_id == 1)  // Initializing a PROSAC sampler. This requires the points to be ordered according to the quality.
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::ProsacSampler(&points, estimator.sampleSize()));
	else if (sampler_id == 2) // Initializing a Progressive NAPSAC sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::ProgressiveNapsacSampler<4>(&points,
			{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
								// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
			estimator.sampleSize(), // The size of a minimal sample
			{ static_cast<double>(w), // The width of the image
				static_cast<double>(h) },  // The height of the image
			0.5)); // The length (i.e., 0.5 * <point number> iterations) of fully blending to global sampling 
	else
	{
		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 2 (P-NAPSAC sampling)\n",
			sampler_id);

		// It is ugly: the unique_ptr does not check for virtual descructors in the base class.
		// Therefore, the derived class's objects are not deleted automatically. 
		// This causes a memory leaking. I hate C++.
		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		return 0;
	}

 	// The local optimization sampler is used inside the local optimization
	sampler::UniformSampler local_optimization_sampler(&points);

	// Checking if the samplers are initialized successfully.
	if (!main_sampler->isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		AbstractSampler *sampler_ptr = main_sampler.release();
		delete sampler_ptr;

		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return 0;
	}

	utils::RANSACStatistics statistics;
	
	// Initializing the fast inlier selector object
	inlier_selector::EmptyInlierSelector<utils::Default2DLineEstimator, 
		AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

	if (use_sprt)
	{
		// Initializing SPRT test
		preemption::SPRTPreemptiveVerfication<utils::Default2DLineEstimator> preemptive_verification(
			points,
			estimator);

		GCRANSAC<utils::Default2DLineEstimator,
			AbstractNeighborhood,
			MSACScoringFunction<utils::Default2DLineEstimator>,
			preemption::SPRTPreemptiveVerfication<utils::Default2DLineEstimator>> gcransac;
		gcransac.settings.threshold = threshold; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = conf; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball

		// Start GC-RANSAC
		gcransac.run(points,
			estimator,
			main_sampler.get(),
			&local_optimization_sampler,
			neighborhood_graph.get(),
			model,
			preemptive_verification,
			inlier_selector);

		statistics = gcransac.getRansacStatistics();
	}
	else
	{
		GCRANSAC<utils::Default2DLineEstimator, AbstractNeighborhood> gcransac;
		gcransac.settings.threshold = threshold; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = conf; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball

		// Start GC-RANSAC
		gcransac.run(points,
			estimator,
			main_sampler.get(),
			&local_optimization_sampler,
			neighborhood_graph.get(),
			model);

		statistics = gcransac.getRansacStatistics();
	}

	estimated_line.resize(3);

	for (int i = 0; i < 3; i++) {
		estimated_line[i] = model.descriptor(i);
	}

	inliers.resize(num_points);

	const int num_inliers = statistics.inliers.size();
	for (auto pt_idx = 0; pt_idx < num_points; ++pt_idx) {
		inliers[pt_idx] = 0;

	}
	for (auto pt_idx = 0; pt_idx < num_inliers; ++pt_idx) {
		inliers[statistics.inliers[pt_idx]] = 1;
	}

	// It is ugly: the unique_ptr does not check for virtual descructors in the base class.
	// Therefore, the derived class's objects are not deleted automatically. 
	// This causes a memory leaking. I hate C++.
	AbstractSampler *sampler_ptr = main_sampler.release();
	delete sampler_ptr;

	AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
	delete neighborhood_graph_ptr;

	// The number of inliers found
	return num_inliers;
}

int find6DPose_(std::vector<double>& imagePoints,
	std::vector<double>& worldPoints,
	std::vector<double> &point_probabilities,
	std::vector<bool>& inliers,
	std::vector<double> &pose,
	double spatial_coherence_weight,
	double threshold,
	double conf,
	int max_iters,
	bool use_sprt,
	double min_inlier_ratio_for_sprt,
	int sampler_id,
	int neighborhood_id,
	double neighborhood_size)
{
	size_t num_tents = imagePoints.size() / 2;
	cv::Mat points(num_tents, 5, CV_64F);
	size_t iterations = 0;
	for (size_t i = 0; i < num_tents; ++i) {
		points.at<double>(i, 0) = imagePoints[2 * i];
		points.at<double>(i, 1) = imagePoints[2 * i + 1];
		points.at<double>(i, 2) = worldPoints[3 * i];
		points.at<double>(i, 3) = worldPoints[3 * i + 1];
		points.at<double>(i, 4) = worldPoints[3 * i + 2];
	}

	typedef neighborhood::NeighborhoodGraph<cv::Mat> AbstractNeighborhood;
	std::unique_ptr<AbstractNeighborhood> neighborhood_graph;

	const size_t cell_number_in_neighborhood_graph_ = 
		static_cast<size_t>(neighborhood_size);

	// Initializing a grid-based neighborhood graph
	if (neighborhood_id == 0) // Initializing the neighbhood graph by FLANN
		neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
			new neighborhood::FlannNeighborhoodGraph(&points, neighborhood_size));
	else
	{
		fprintf(stderr, "Unknown neighborhood-graph identifier: %d. The accepted values are 0 (FLANN-based neighborhood)\n",
			neighborhood_id);
		return 0;
	}

	// Apply Graph-cut RANSAC
	utils::DefaultPnPEstimator estimator;
	Pose6D model;

	// Initialize the samplers	
	// The main sampler is used for sampling in the main RANSAC loop
	typedef sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Initializing a RANSAC-like uniform sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::UniformSampler(&points));
	else if (sampler_id == 1) // Initializing a PROSAC sampler. This requires the points to be ordered according to the quality.
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::ProsacSampler(&points, estimator.sampleSize()));
	else
	{
		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling)\n",
			sampler_id);
		return 0;
	}

	sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!main_sampler->isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");

		// It is ugly: the unique_ptr does not check for virtual descructors in the base class.
		// Therefore, the derived class's objects are not deleted automatically. 
		// This causes a memory leaking. I hate C++.
		AbstractSampler *sampler_ptr = main_sampler.release();
		delete sampler_ptr;

		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		return 0;
	}

	utils::RANSACStatistics statistics;
	
	// Initializing the fast inlier selector object
	inlier_selector::EmptyInlierSelector<utils::DefaultPnPEstimator, 
		AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

	if (use_sprt)
	{
		// Initializing SPRT test
		preemption::SPRTPreemptiveVerfication<utils::DefaultPnPEstimator> preemptive_verification(
			points,
			estimator,
			min_inlier_ratio_for_sprt);

		GCRANSAC<utils::DefaultPnPEstimator,
			AbstractNeighborhood,
			MSACScoringFunction<utils::DefaultPnPEstimator>,
			preemption::SPRTPreemptiveVerfication<utils::DefaultPnPEstimator>> gcransac;
		gcransac.settings.threshold = threshold; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = conf; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = 8; // The radius of the neighborhood ball

		// Start GC-RANSAC
		gcransac.run(points,
			estimator,
			main_sampler.get(),
			&local_optimization_sampler,
			neighborhood_graph.get(),
			model,
			preemptive_verification,
			inlier_selector);

		statistics = gcransac.getRansacStatistics();
	}
	else
	{
		GCRANSAC<utils::DefaultPnPEstimator,
			AbstractNeighborhood> gcransac;
		gcransac.settings.threshold = threshold; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = conf; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = 8; // The radius of the neighborhood ball

		// Start GC-RANSAC
		gcransac.run(points,
			estimator,
			main_sampler.get(),
			&local_optimization_sampler,
			neighborhood_graph.get(),
			model);

		statistics = gcransac.getRansacStatistics();
	}

	inliers.resize(num_tents);

	const int num_inliers = statistics.inliers.size();
	for (auto pt_idx = 0; pt_idx < num_tents; ++pt_idx) {
		inliers[pt_idx] = 0;
	}

	for (auto pt_idx = 0; pt_idx < num_inliers; ++pt_idx) {
		inliers[statistics.inliers[pt_idx]] = 1;
	}

	pose.resize(12);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 4; j++) {
			pose[i * 4 + j] = (double)model.descriptor(i, j);
		}
	}

	// It is ugly: the unique_ptr does not check for virtual descructors in the base class.
	// Therefore, the derived class's objects are not deleted automatically. 
	// This causes a memory leaking. I hate C++.
	AbstractSampler *sampler_ptr = main_sampler.release();
	delete sampler_ptr;

	AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
	delete neighborhood_graph_ptr;

	// The number of inliers found
	return num_inliers;
}

int findRigidTransform_(std::vector<double>& points1,
	std::vector<double>& points2,
	std::vector<double> &point_probabilities,
	std::vector<bool>& inliers,
	std::vector<double> &pose,
	double spatial_coherence_weight,
	double threshold,
	double conf,
	int max_iters,
	bool use_sprt,
	double min_inlier_ratio_for_sprt,
	int sampler_id,
	int neighborhood_id,
	double neighborhood_size)
{
	size_t num_tents = points1.size() / 3;
	cv::Mat points(num_tents, 6, CV_64F);
	size_t iterations = 0;
	for (size_t i = 0; i < num_tents; ++i) {
		points.at<double>(i, 0) = points1[3 * i];
		points.at<double>(i, 1) = points1[3 * i + 1];
		points.at<double>(i, 2) = points1[3 * i + 2];
		points.at<double>(i, 3) = points2[3 * i];
		points.at<double>(i, 4) = points2[3 * i + 1];
		points.at<double>(i, 5) = points2[3 * i + 2];
	}

	typedef neighborhood::NeighborhoodGraph<cv::Mat> AbstractNeighborhood;
	std::unique_ptr<AbstractNeighborhood> neighborhood_graph;

	const size_t cell_number_in_neighborhood_graph_ = 
		static_cast<size_t>(neighborhood_size);

	// Initializing the neighbhood graph by FLANN
	if (neighborhood_id == 0)
		neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
			new neighborhood::FlannNeighborhoodGraph(&points, neighborhood_size));
	else
	{
		fprintf(stderr, "Unknown neighborhood-graph identifier: %d. The accepted values are 0 (FLANN-based neighborhood)\n",
			neighborhood_id);
		return 0;
	}

	// Apply Graph-cut RANSAC
	utils::DefaultRigidTransformationEstimator estimator;
	RigidTransformation model;

	// Initialize the samplers	
	// The main sampler is used for sampling in the main RANSAC loop
	typedef sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Initializing a RANSAC-like uniformly random sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::UniformSampler(&points));
	else if (sampler_id == 1)  // Initializing a PROSAC sampler. This requires the points to be ordered according to the quality.
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::ProsacSampler(&points, estimator.sampleSize()));
	else
	{
		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling)\n",
			sampler_id);
		return 0;
	}

	sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!main_sampler->isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");

		// It is ugly: the unique_ptr does not check for virtual descructors in the base class.
		// Therefore, the derived class's objects are not deleted automatically. 
		// This causes a memory leaking. I hate C++.
		AbstractSampler *sampler_ptr = main_sampler.release();
		delete sampler_ptr;

		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		return 0;
	}

	utils::RANSACStatistics statistics;
	
	// Initializing the fast inlier selector object
	inlier_selector::EmptyInlierSelector<utils::DefaultRigidTransformationEstimator, 
		AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

	if (use_sprt)
	{
		// Initializing SPRT test
		preemption::SPRTPreemptiveVerfication<utils::DefaultRigidTransformationEstimator> preemptive_verification(
			points,
			estimator,
			min_inlier_ratio_for_sprt);

		GCRANSAC<utils::DefaultRigidTransformationEstimator,
			AbstractNeighborhood,
			MSACScoringFunction<utils::DefaultRigidTransformationEstimator>,
			preemption::SPRTPreemptiveVerfication<utils::DefaultRigidTransformationEstimator>> gcransac;
		gcransac.settings.threshold = threshold; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = conf; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = 20; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = 20; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = 8; // The radius of the neighborhood ball

		// Start GC-RANSAC
		gcransac.run(points,
			estimator,
			main_sampler.get(),
			&local_optimization_sampler,
			neighborhood_graph.get(),
			model,
			preemptive_verification,
			inlier_selector);

		statistics = gcransac.getRansacStatistics();
	}
	else
	{
		GCRANSAC<utils::DefaultRigidTransformationEstimator,
			AbstractNeighborhood> gcransac;
		gcransac.setFPS(-1); // Set the desired FPS (-1 means no limit)
		gcransac.settings.threshold = threshold; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = conf; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = 8; // The radius of the neighborhood ball

		// Start GC-RANSAC
		gcransac.run(points,
			estimator,
			main_sampler.get(),
			&local_optimization_sampler,
			neighborhood_graph.get(),
			model);

		statistics = gcransac.getRansacStatistics();
	}

	inliers.resize(num_tents);

	const int num_inliers = statistics.inliers.size();
	for (auto pt_idx = 0; pt_idx < num_tents; ++pt_idx) {
		inliers[pt_idx] = 0;
	}

	for (auto pt_idx = 0; pt_idx < num_inliers; ++pt_idx) {
		inliers[statistics.inliers[pt_idx]] = 1;
	}

	pose.resize(16);

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			pose[i * 4 + j] = (double)model.descriptor(i, j);
		}
	}

	// It is ugly: the unique_ptr does not check for virtual descructors in the base class.
	// Therefore, the derived class's objects are not deleted automatically. 
	// This causes a memory leaking. I hate C++.
	AbstractSampler *sampler_ptr = main_sampler.release();
	delete sampler_ptr;

	AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
	delete neighborhood_graph_ptr;

	// The number of inliers found
	return num_inliers;
}

int findFundamentalMatrix_(
	std::vector<double>& correspondences,
	std::vector<double> &point_probabilities,
	std::vector<bool>& inliers,
	std::vector<double>& F,
	int h1, int w1, int h2, int w2,
	double spatial_coherence_weight,
	double threshold,
	double conf,
	int max_iters,
	bool use_sprt,
	double min_inlier_ratio_for_sprt,
	int sampler_id,
	int neighborhood_id,
	double neighborhood_size)
{
	int num_tents = correspondences.size() / 4;
	cv::Mat points(num_tents, 4, CV_64F, &correspondences[0]);

	typedef neighborhood::NeighborhoodGraph<cv::Mat> AbstractNeighborhood;
	std::unique_ptr<AbstractNeighborhood> neighborhood_graph;

	const size_t cell_number_in_neighborhood_graph_ = 
		static_cast<size_t>(neighborhood_size);

	// If the spatial weight is 0.0, the neighborhood graph should not be created 
	if (spatial_coherence_weight <= std::numeric_limits<double>::epsilon())
	{
		cv::Mat emptyPoints(0, 4, CV_64F);

		neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
			new neighborhood::GridNeighborhoodGraph<4>(&emptyPoints, // The input points
			{ 0, // The cell size along axis X
				0 }, // The cell size along axis Y
			1)); // The cell number along every axis
	} else // Initializing a grid-based neighborhood graph
	{
		// Initializing a grid-based neighborhood graph
		if (neighborhood_id == 0)
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::GridNeighborhoodGraph<4>(&points,
				{ w1 / static_cast<double>(cell_number_in_neighborhood_graph_),
					h1 / static_cast<double>(cell_number_in_neighborhood_graph_),
					w2 / static_cast<double>(cell_number_in_neighborhood_graph_),
					h2 / static_cast<double>(cell_number_in_neighborhood_graph_) },
				cell_number_in_neighborhood_graph_));
		else if (neighborhood_id == 1) // Initializing the neighbhood graph by FLANN
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::FlannNeighborhoodGraph(&points, neighborhood_size));
		else
		{
			fprintf(stderr, "Unknown neighborhood-graph identifier: %d. The accepted values are 0 (Grid-based), 1 (FLANN-based neighborhood)\n",
				neighborhood_id);
			return 0;
		}

		// Checking if the neighborhood graph is initialized successfully.
		if (!neighborhood_graph->isInitialized())
		{
			AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
			delete neighborhood_graph_ptr;

			fprintf(stderr, "The neighborhood graph is not initialized successfully.\n");
			return 0;
		}
	}

	// Calculating the maximum image diagonal to be used for setting the threshold
	// adaptively for each image pair. 
	const double max_image_diagonal =
		sqrt(pow(MAX(w1, w2), 2) + pow(MAX(h1, h2), 2));

	// Apply Graph-cut RANSAC
	utils::DefaultFundamentalMatrixEstimator estimator;
	FundamentalMatrix model;

	// Initialize the samplers
	// The main sampler is used for sampling in the main RANSAC loop
	typedef sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Initializing a RANSAC-like uniformly random sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::UniformSampler(&points));
	else if (sampler_id == 1)  // Initializing a PROSAC sampler. This requires the points to be ordered according to the quality.
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::ProsacSampler(&points, estimator.sampleSize()));
	else if (sampler_id == 2) // Initializing a Progressive NAPSAC sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::ProgressiveNapsacSampler<4>(&points,
			{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
								// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
			estimator.sampleSize(), // The size of a minimal sample
			{ static_cast<double>(w1), // The width of the source image
				static_cast<double>(h1), // The height of the source image
				static_cast<double>(w2), // The width of the destination image
				static_cast<double>(h2) },  // The height of the destination image
			0.5)); // The length (i.e., 0.5 * <point number> iterations) of fully blending to global sampling 
	else if (sampler_id == 3)
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ImportanceSampler(&points, 
            point_probabilities,
            estimator.sampleSize()));
	else if (sampler_id == 4)
    {
		double variance = 0.1;
        double max_prob = 0;
        for (const auto &prob : point_probabilities)
            max_prob = MAX(max_prob, prob);
        for (auto &prob : point_probabilities)
            prob /= max_prob;
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::AdaptiveReorderingSampler(&points, 
            point_probabilities,
            estimator.sampleSize(),
            variance));
	}
	else
	{
		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 2 (P-NAPSAC sampling)\n",
			sampler_id);
		return 0;
	}

	sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!main_sampler->isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");

		// It is ugly: the unique_ptr does not check for virtual descructors in the base class.
		// Therefore, the derived class's objects are not deleted automatically. 
		// This causes a memory leaking.
		AbstractSampler *sampler_ptr = main_sampler.release();
		delete sampler_ptr;

		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		return 0;
	}


	utils::RANSACStatistics statistics;
	
	// Initializing the fast inlier selector object
	inlier_selector::EmptyInlierSelector<utils::DefaultFundamentalMatrixEstimator, 
		AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

	if (use_sprt)
	{
		// Initializing SPRT test
		preemption::SPRTPreemptiveVerfication<utils::DefaultFundamentalMatrixEstimator> preemptive_verification(
			points,
			estimator,
			min_inlier_ratio_for_sprt);

		GCRANSAC<utils::DefaultFundamentalMatrixEstimator,
			AbstractNeighborhood,
			MSACScoringFunction<utils::DefaultFundamentalMatrixEstimator>,
			preemption::SPRTPreemptiveVerfication<utils::DefaultFundamentalMatrixEstimator>> gcransac;
		gcransac.settings.threshold = 0.0005 * threshold * max_image_diagonal; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = conf; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = 8; // The radius of the neighborhood ball

		// Start GC-RANSAC
		gcransac.run(points,
			estimator,
			main_sampler.get(),
			&local_optimization_sampler,
			neighborhood_graph.get(),
			model,
			preemptive_verification,
			inlier_selector);

		statistics = gcransac.getRansacStatistics();
	}
	else
	{
		GCRANSAC<utils::DefaultFundamentalMatrixEstimator, AbstractNeighborhood> gcransac;
		gcransac.settings.threshold = 0.0005 * threshold * max_image_diagonal; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = conf; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = 8; // The radius of the neighborhood ball

		// Start GC-RANSAC
		gcransac.run(points,
			estimator,
			main_sampler.get(),
			&local_optimization_sampler,
			neighborhood_graph.get(),
			model);

		statistics = gcransac.getRansacStatistics();
	}

	inliers.resize(num_tents);

	const int num_inliers = statistics.inliers.size();
	for (auto pt_idx = 0; pt_idx < num_tents; ++pt_idx) {
		inliers[pt_idx] = 0;
	}

	for (auto pt_idx = 0; pt_idx < num_inliers; ++pt_idx) {
		inliers[statistics.inliers[pt_idx]] = 1;
	}

	F.resize(9);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			F[i * 3 + j] = (double)model.descriptor(i, j);
		}
	}

	// It is ugly: the unique_ptr does not check for virtual descructors in the base class.
	// Therefore, the derived class's objects are not deleted automatically. 
	// This causes a memory leaking. I hate C++.
	AbstractSampler *sampler_ptr = main_sampler.release();
	delete sampler_ptr;

	AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
	delete neighborhood_graph_ptr;

	// The number of inliers found
	return num_inliers;
}

int findEssentialMatrix_(
	std::vector<double>& correspondences,
	std::vector<double> &point_probabilities,
	std::vector<bool>& inliers,
	std::vector<double>&E,
	std::vector<double>& src_K,
	std::vector<double>& dst_K,
	int h1, int w1, int h2, int w2,
	double spatial_coherence_weight,
	double threshold,
	double conf,
	int max_iters,
	bool use_sprt,
	double min_inlier_ratio_for_sprt,
	int sampler_id,
	int neighborhood_id,
	double neighborhood_size)
{
	int num_tents = correspondences.size() / 4;
	cv::Mat points(num_tents, 4, CV_64F, &correspondences[0]);

	typedef neighborhood::NeighborhoodGraph<cv::Mat> AbstractNeighborhood;
	std::unique_ptr<AbstractNeighborhood> neighborhood_graph;

	const size_t cell_number_in_neighborhood_graph_ = 
		static_cast<size_t>(neighborhood_size);

	// If the spatial weight is 0.0, the neighborhood graph should not be created 
	if (spatial_coherence_weight <= std::numeric_limits<double>::epsilon())
	{
		cv::Mat emptyPoints(0, 4, CV_64F);

		neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
			new neighborhood::GridNeighborhoodGraph<4>(&emptyPoints, // The input points
			{ 0, // The cell size along axis X
				0 }, // The cell size along axis Y
			1)); // The cell number along every axis
	} else // Initializing a grid-based neighborhood graph
	{
		// Initializing a grid-based neighborhood graph
		if (neighborhood_id == 0)
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::GridNeighborhoodGraph<4>(&points,
				{ w1 / static_cast<double>(cell_number_in_neighborhood_graph_),
					h1 / static_cast<double>(cell_number_in_neighborhood_graph_),
					w2 / static_cast<double>(cell_number_in_neighborhood_graph_),
					h2 / static_cast<double>(cell_number_in_neighborhood_graph_) },
				cell_number_in_neighborhood_graph_));
		else if (neighborhood_id == 1) // Initializing the neighbhood graph by FLANN
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::FlannNeighborhoodGraph(&points, neighborhood_size));
		else
		{
			fprintf(stderr, "Unknown neighborhood-graph identifier: %d. The accepted values are 0 (Grid-based), 1 (FLANN-based neighborhood)\n",
				neighborhood_id);
			return 0;
		}
		
		// Checking if the neighborhood graph is initialized successfully.
		if (!neighborhood_graph->isInitialized())
		{
			AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
			delete neighborhood_graph_ptr;

			fprintf(stderr, "The neighborhood graph is not initialized successfully.\n");
			return 0;
		}
	}

	Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> intrinsics_src(&src_K[0]);
	Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> intrinsics_dst(&dst_K[0]);

	// Calculating the maximum image diagonal to be used for setting the threshold
	// adaptively for each image pair. 
	const double threshold_normalizer =
		0.25 * (intrinsics_src(0,0) + intrinsics_src(1,1) + intrinsics_dst(0,0) + intrinsics_dst(1,1));

	cv::Mat normalized_points(points.size(), CV_64F);
	utils::normalizeCorrespondences(points,
		intrinsics_src,
		intrinsics_dst,
		normalized_points);

	// Apply Graph-cut RANSAC
	utils::DefaultEssentialMatrixEstimator estimator(intrinsics_src,
		intrinsics_dst);
	EssentialMatrix model;

	// Initialize the samplers
	// The main sampler is used for sampling in the main RANSAC loop
	typedef sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Initializing a RANSAC-like uniformly random sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::UniformSampler(&points));
	else if (sampler_id == 1)  // Initializing a PROSAC sampler. This requires the points to be ordered according to the quality.
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::ProsacSampler(&points, estimator.sampleSize()));
	else if (sampler_id == 2) // Initializing a Progressive NAPSAC sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::ProgressiveNapsacSampler<4>(&points,
			{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
								// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
			estimator.sampleSize(), // The size of a minimal sample
			{ static_cast<double>(w1), // The width of the source image
				static_cast<double>(h1), // The height of the source image
				static_cast<double>(w2), // The width of the destination image
				static_cast<double>(h2) },  // The height of the destination image
			0.5)); // The length (i.e., 0.5 * <point number> iterations) of fully blending to global sampling 
	else if (sampler_id == 3)
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ImportanceSampler(&points, 
            point_probabilities,
            estimator.sampleSize()));
	else if (sampler_id == 4)
    {
		double variance = 0.1;
        double max_prob = 0;
        for (const auto &prob : point_probabilities)
            max_prob = MAX(max_prob, prob);
        for (auto &prob : point_probabilities)
            prob /= max_prob;
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::AdaptiveReorderingSampler(&points, 
            point_probabilities,
            estimator.sampleSize(),
            variance));
	}
	else
	{
		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 2 (P-NAPSAC sampling)\n",
			sampler_id);
		return 0;
	}
	
	// The local optimization sampler is used inside the local optimization
	sampler::UniformSampler local_optimization_sampler(&points);

	// Checking if the samplers are initialized successfully.
	if (!main_sampler->isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		// It is ugly: the unique_ptr does not check for virtual descructors in the base class.
		// Therefore, the derived class's objects are not deleted automatically. 
		// This causes a memory leaking. I hate C++.
		AbstractSampler *sampler_ptr = main_sampler.release();
		delete sampler_ptr;

		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return 0;
	}
	
	utils::RANSACStatistics statistics;
	
	// Initializing the fast inlier selector object
	inlier_selector::EmptyInlierSelector<utils::DefaultEssentialMatrixEstimator, 
		AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

	if (use_sprt)
	{
		// Initializing SPRT test
		preemption::SPRTPreemptiveVerfication<utils::DefaultEssentialMatrixEstimator> preemptive_verification(
			points,
			estimator,
			min_inlier_ratio_for_sprt);

		GCRANSAC<utils::DefaultEssentialMatrixEstimator,
			AbstractNeighborhood,
			MSACScoringFunction<utils::DefaultEssentialMatrixEstimator>,
			preemption::SPRTPreemptiveVerfication<utils::DefaultEssentialMatrixEstimator>> gcransac;
		gcransac.settings.threshold = threshold / threshold_normalizer; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = conf; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = 100; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = 1000; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball
		gcransac.settings.do_final_iterated_least_squares = false;
		gcransac.settings.max_graph_cut_number = 100;

		// Start GC-RANSAC
		gcransac.run(normalized_points,
			estimator,
			main_sampler.get(),
			&local_optimization_sampler,
			neighborhood_graph.get(),
			model,
			preemptive_verification,
			inlier_selector);

		statistics = gcransac.getRansacStatistics();
	}
	else
	{
		
		GCRANSAC<utils::DefaultEssentialMatrixEstimator, AbstractNeighborhood> gcransac;
		gcransac.settings.threshold = threshold / threshold_normalizer; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = conf; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = 1000; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball

		// Start GC-RANSAC
		gcransac.run(normalized_points,
			estimator,
			main_sampler.get(),
			&local_optimization_sampler,
			neighborhood_graph.get(),
			model);

		statistics = gcransac.getRansacStatistics();
	}

	inliers.resize(num_tents);

	const int num_inliers = statistics.inliers.size();
	for (auto pt_idx = 0; pt_idx < num_tents; ++pt_idx) {
		inliers[pt_idx] = 0;

	}
	for (auto pt_idx = 0; pt_idx < num_inliers; ++pt_idx) {
		inliers[statistics.inliers[pt_idx]] = 1;
	}

	E.resize(9);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			E[i * 3 + j] = (double)model.descriptor(i, j);
		}
	}

	// It is ugly: the unique_ptr does not check for virtual descructors in the base class.
	// Therefore, the derived class's objects are not deleted automatically. 
	// This causes a memory leaking. I hate C++.
	AbstractSampler *sampler_ptr = main_sampler.release();
	delete sampler_ptr;

	AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
	delete neighborhood_graph_ptr;

	// The number of inliers found
	return num_inliers;
}


int findHomography_(
	std::vector<double>& correspondences,
	std::vector<double> &point_probabilities,
	std::vector<bool>& inliers,
	std::vector<double>& H,
	int h1, int w1, int h2, int w2,
	double spatial_coherence_weight,
	double threshold,
	double conf,
	int max_iters,
	bool use_sprt,
	double min_inlier_ratio_for_sprt,
	int sampler_id,
	int neighborhood_id,
	double neighborhood_size,
	bool use_space_partitioning)
{
	int num_tents = correspondences.size() / 4;
	cv::Mat points(num_tents, 4, CV_64F, &correspondences[0]);
	
	typedef neighborhood::NeighborhoodGraph<cv::Mat> AbstractNeighborhood;
	std::unique_ptr<AbstractNeighborhood> neighborhood_graph;

	const size_t cell_number_in_neighborhood_graph_ = 
		static_cast<size_t>(neighborhood_size);

	if (use_space_partitioning && neighborhood_id != 0)
	{
		fprintf(stderr, "Space Partitioning only works with Grid neighorbood yet. Thus, setting neighborhood_id = 0.\n");
		neighborhood_id = 0;
	}

	// If the spatial weight is 0.0, the neighborhood graph should not be created 
	// Initializing a grid-based neighborhood graph
	if (neighborhood_id == 0)
		neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
			new neighborhood::GridNeighborhoodGraph<4>(&points,
			{ (w1 + std::numeric_limits<double>::epsilon()) / static_cast<double>(cell_number_in_neighborhood_graph_),
				(h1 + std::numeric_limits<double>::epsilon()) / static_cast<double>(cell_number_in_neighborhood_graph_),
				(w2 + std::numeric_limits<double>::epsilon()) / static_cast<double>(cell_number_in_neighborhood_graph_),
				(h2 + std::numeric_limits<double>::epsilon()) / static_cast<double>(cell_number_in_neighborhood_graph_) },
			cell_number_in_neighborhood_graph_));
	else if (neighborhood_id == 1) // Initializing the neighbhood graph by FLANN
		neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
			new neighborhood::FlannNeighborhoodGraph(&points, neighborhood_size));
	else
	{
		fprintf(stderr, "Unknown neighborhood-graph identifier: %d. The accepted values are 0 (Grid-based), 1 (FLANN-based neighborhood)\n",
			neighborhood_id);
		return 0;
	}

	// Checking if the neighborhood graph is initialized successfully.
	if (!neighborhood_graph->isInitialized())
	{
		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		fprintf(stderr, "The neighborhood graph is not initialized successfully.\n");
		return 0;
	}

	// Calculating the maximum image diagonal to be used for setting the threshold
	// adaptively for each image pair. 
	const double max_image_diagonal =
		sqrt(pow(MAX(w1, w2), 2) + pow(MAX(h1, h2), 2));

	utils::DefaultHomographyEstimator estimator;
	Homography model;

	// Initialize the samplers
	// The main sampler is used for sampling in the main RANSAC loop
	typedef sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Initializing a RANSAC-like uniformly random sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::UniformSampler(&points));
	else if (sampler_id == 1)  // Initializing a PROSAC sampler. This requires the points to be ordered according to the quality.
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::ProsacSampler(&points, estimator.sampleSize()));
	else if (sampler_id == 2) // Initializing a Progressive NAPSAC sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::ProgressiveNapsacSampler<4>(&points,
			{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
								// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
			estimator.sampleSize(), // The size of a minimal sample
			{ static_cast<double>(w1), // The width of the source image
				static_cast<double>(h1), // The height of the source image
				static_cast<double>(w2), // The width of the destination image
				static_cast<double>(h2) },  // The height of the destination image
			0.5)); // The length (i.e., 0.5 * <point number> iterations) of fully blending to global sampling 
	else if (sampler_id == 3)
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ImportanceSampler(&points, 
            point_probabilities,
            estimator.sampleSize()));
	else if (sampler_id == 4)
    {
		double variance = 0.1;
        double max_prob = 0;
        for (const auto &prob : point_probabilities)
            max_prob = MAX(max_prob, prob);
        for (auto &prob : point_probabilities)
            prob /= max_prob;
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::AdaptiveReorderingSampler(&points, 
            point_probabilities,
            estimator.sampleSize(),
            variance));
	}
	else
	{
		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 2 (P-NAPSAC sampling)\n",
			sampler_id);
		return 0;
	}

	sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!main_sampler->isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		// It is ugly: the unique_ptr does not check for virtual descructors in the base class.
		// Therefore, the derived class's objects are not deleted automatically. 
		// This causes a memory leaking. I hate C++.
		AbstractSampler *sampler_ptr = main_sampler.release();
		delete sampler_ptr;

		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return 0;
	}

	utils::RANSACStatistics statistics;

	if (use_sprt)
	{
		// Initializing SPRT test
		preemption::SPRTPreemptiveVerfication<utils::DefaultHomographyEstimator> preemptive_verification(
			points,
			estimator);

		if (use_space_partitioning)
		{			
			inlier_selector::SpacePartitioningRANSAC<utils::DefaultHomographyEstimator, AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

			GCRANSAC<utils::DefaultHomographyEstimator,
				AbstractNeighborhood,
				MSACScoringFunction<utils::DefaultHomographyEstimator>,
				preemption::SPRTPreemptiveVerfication<utils::DefaultHomographyEstimator>,
				inlier_selector::SpacePartitioningRANSAC<utils::DefaultHomographyEstimator, AbstractNeighborhood>> gcransac;
			gcransac.settings.threshold = threshold; // The inlier-outlier threshold
			gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
			gcransac.settings.confidence = conf; // The required confidence in the results
			gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
			gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
			gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
			gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball

			// Start GC-RANSAC
			gcransac.run(points,
				estimator,
				main_sampler.get(),
				&local_optimization_sampler,
				neighborhood_graph.get(),
				model,
				preemptive_verification,
				inlier_selector);

			statistics = gcransac.getRansacStatistics();
		} else
		{
			inlier_selector::EmptyInlierSelector<utils::DefaultHomographyEstimator, AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

			GCRANSAC<utils::DefaultHomographyEstimator,
				AbstractNeighborhood,
				MSACScoringFunction<utils::DefaultHomographyEstimator>,
				preemption::SPRTPreemptiveVerfication<utils::DefaultHomographyEstimator>,
				inlier_selector::EmptyInlierSelector<utils::DefaultHomographyEstimator, AbstractNeighborhood>> gcransac;
			gcransac.settings.threshold = threshold; // The inlier-outlier threshold
			gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
			gcransac.settings.confidence = conf; // The required confidence in the results
			gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
			gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
			gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
			gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball

			// Start GC-RANSAC
			gcransac.run(points,
				estimator,
				main_sampler.get(),
				&local_optimization_sampler,
				neighborhood_graph.get(),
				model,
				preemptive_verification,
				inlier_selector);

			statistics = gcransac.getRansacStatistics();
		}
	}
	else
	{
		// Initializing an empty preemption
		preemption::EmptyPreemptiveVerfication<utils::DefaultHomographyEstimator> preemptive_verification;

		if (use_space_partitioning)
		{			
			inlier_selector::SpacePartitioningRANSAC<utils::DefaultHomographyEstimator, AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

			GCRANSAC<utils::DefaultHomographyEstimator,
				AbstractNeighborhood,
				MSACScoringFunction<utils::DefaultHomographyEstimator>,
				preemption::EmptyPreemptiveVerfication<utils::DefaultHomographyEstimator>,
				inlier_selector::SpacePartitioningRANSAC<utils::DefaultHomographyEstimator, AbstractNeighborhood>> gcransac;
			gcransac.settings.threshold = threshold; // The inlier-outlier threshold
			gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
			gcransac.settings.confidence = conf; // The required confidence in the results
			gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
			gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
			gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
			gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball

			// Start GC-RANSAC
			gcransac.run(points,
				estimator,
				main_sampler.get(),
				&local_optimization_sampler,
				neighborhood_graph.get(),
				model,
				preemptive_verification,
				inlier_selector);

			statistics = gcransac.getRansacStatistics();
		} else
		{
			inlier_selector::EmptyInlierSelector<utils::DefaultHomographyEstimator, AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

			GCRANSAC<utils::DefaultHomographyEstimator,
				AbstractNeighborhood,
				MSACScoringFunction<utils::DefaultHomographyEstimator>,
				preemption::EmptyPreemptiveVerfication<utils::DefaultHomographyEstimator>,
				inlier_selector::EmptyInlierSelector<utils::DefaultHomographyEstimator, AbstractNeighborhood>> gcransac;
			gcransac.settings.threshold = threshold; // The inlier-outlier threshold
			gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
			gcransac.settings.confidence = conf; // The required confidence in the results
			gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
			gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
			gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
			gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball

			// Start GC-RANSAC
			gcransac.run(points,
				estimator,
				main_sampler.get(),
				&local_optimization_sampler,
				neighborhood_graph.get(),
				model,
				preemptive_verification,
				inlier_selector);

			statistics = gcransac.getRansacStatistics();
		}
	}

	H.resize(9);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			H[i * 3 + j] = model.descriptor(i, j);
		}
	}

	inliers.resize(num_tents);

	const int num_inliers = statistics.inliers.size();
	for (auto pt_idx = 0; pt_idx < num_tents; ++pt_idx) {
		inliers[pt_idx] = 0;

	}
	for (auto pt_idx = 0; pt_idx < num_inliers; ++pt_idx) {
		inliers[statistics.inliers[pt_idx]] = 1;
	}

	// It is ugly: the unique_ptr does not check for virtual descructors in the base class.
	// Therefore, the derived class's objects are not deleted automatically. 
	// This causes a memory leaking. I hate C++.
	AbstractSampler *sampler_ptr = main_sampler.release();
	delete sampler_ptr;

	AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
	delete neighborhood_graph_ptr;

	// Return the number of inliers found
	return num_inliers;
}
