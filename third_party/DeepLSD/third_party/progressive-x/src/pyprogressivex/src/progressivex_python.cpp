#include "progressivex_python.h"
#include <vector>
#include <thread>
#include "utils.h"
#include <opencv2/core/core.hpp>
#include <Eigen/Eigen>

#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>

#include "progx_utils.h"
#include "utils.h"
#include "GCoptimization.h"
#include "neighborhood/grid_neighborhood_graph.h"
#include "neighborhood/flann_neighborhood_graph.h"

#include "samplers/uniform_sampler.h"
#include "samplers/prosac_sampler.h"
#include "samplers/napsac_sampler.h"
#include "samplers/progressive_napsac_sampler.h"

#include "estimators/fundamental_estimator.h"
#include "estimators/homography_estimator.h"
#include "estimators/essential_estimator.h"
#include "vanishing_point_estimator.h"
#include "solver_vanishing_point_two_lines.h"

#include "progressive_x.h"

#include <ctime>
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
	#include <direct.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>

#include <mutex>
#include <glog/logging.h>

int find6DPoses_(
	const std::vector<double>& imagePoints,
	const std::vector<double>& worldPoints,
	const std::vector<double>& intrinsicParams,
	std::vector<size_t>& labeling,
	std::vector<double>& poses,
	const double &spatial_coherence_weight,
	const double &threshold,
	const double &confidence,
	const double &neighborhood_ball_radius,
	const double &maximum_tanimoto_similarity,
	const size_t &max_iters,
	const size_t &minimum_point_number,
	const int &maximum_model_number)
{
	// Initialize Google's logging library.
	static bool isLoggingInitialized = false;
	if (!isLoggingInitialized)
	{
		google::InitGoogleLogging("pyprogessivex");
		isLoggingInitialized = true;
	}
	
	// Calculate the inverse of the intrinsic camera parameters
	Eigen::Matrix3d K;
	K << intrinsicParams[0], intrinsicParams[1], intrinsicParams[2],
		intrinsicParams[3], intrinsicParams[4], intrinsicParams[5],
		intrinsicParams[6], intrinsicParams[7], intrinsicParams[8];
	const Eigen::Matrix3d Kinv =
		K.inverse();
	
	Eigen::Vector3d vec;
	vec(2) = 1;
	size_t num_tents = imagePoints.size() / 2;
	cv::Mat points(num_tents, 5, CV_64F);
	cv::Mat normalized_points(num_tents, 5, CV_64F);
	size_t iterations = 0;
	for (size_t i = 0; i < num_tents; ++i) {
		vec(0) = imagePoints[2 * i];
		vec(1) = imagePoints[2 * i + 1];
		
		points.at<double>(i, 0) = imagePoints[2 * i];
		points.at<double>(i, 1) = imagePoints[2 * i + 1];
		points.at<double>(i, 2) = worldPoints[3 * i];
		points.at<double>(i, 3) = worldPoints[3 * i + 1];
		points.at<double>(i, 4) = worldPoints[3 * i + 2];
		
		normalized_points.at<double>(i, 0) = Kinv.row(0) * vec;
		normalized_points.at<double>(i, 1) = Kinv.row(1) * vec;
		normalized_points.at<double>(i, 2) = worldPoints[3 * i];
		normalized_points.at<double>(i, 3) = worldPoints[3 * i + 1];
		normalized_points.at<double>(i, 4) = worldPoints[3 * i + 2];
	}
	
	// Normalize the threshold
	const double f = 0.5 * (K(0,0) + K(1,1));
	const double normalized_threshold =
		threshold / f;
	
	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&points, // All data points
		neighborhood_ball_radius); // The radius of the neighborhood ball for determining the neighborhoods.
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds

	printf("Neighborhood calculation time = %f secs.\n", elapsed_seconds.count());

	// The main sampler is used inside the local optimization
	gcransac::sampler::UniformSampler main_sampler(&points);

	// The local optimization sampler is used inside the local optimization
	gcransac::sampler::UniformSampler local_optimization_sampler(&points);

	// Applying Progressive-X
	progx::ProgressiveX<gcransac::neighborhood::FlannNeighborhoodGraph, // The type of the used neighborhood-graph
		gcransac::utils::DefaultPnPEstimator, // The type of the used model estimator
		gcransac::sampler::UniformSampler, // The type of the used main sampler in GC-RANSAC
		gcransac::sampler::UniformSampler> // The type of the used sampler in the local optimization of GC-RANSAC
		progressive_x(nullptr);

	// Set the parameters of Progressive-X
	progx::MultiModelSettings &settings = progressive_x.getMutableSettings();
	// The minimum number of inlier required to keep a model instance.
	// This value is used to determine the label cost weight in the alpha-expansion of PEARL.
	settings.minimum_number_of_inliers = minimum_point_number;
	// The inlier-outlier threshold
	settings.inlier_outlier_threshold = normalized_threshold;
	// The required confidence in the results
	settings.setConfidence(confidence);
	// The maximum Tanimoto similarity of the proposal and compound instances
	settings.maximum_tanimoto_similarity = maximum_tanimoto_similarity;
	// The weight of the spatial coherence term
	settings.spatial_coherence_weight = spatial_coherence_weight;
	// Setting the maximum iteration number
	settings.proposal_engine_settings.max_iteration_number = max_iters;
	// Setting the maximum model number if needed
	if (maximum_model_number > 0)
		settings.maximum_model_number = maximum_model_number;

	progressive_x.run(normalized_points, // All data points
		neighborhood, // The neighborhood graph
		main_sampler, // The main sampler used in GC-RANSAC
		local_optimization_sampler); // The sampler used in the local optimization of GC-RANSAC
	
	// The obtained labeling
	labeling = progressive_x.getStatistics().labeling;
	poses.reserve(12 * progressive_x.getModelNumber());
	
	// Saving the homography parameters
	for (size_t model_idx = 0; model_idx < progressive_x.getModelNumber(); ++model_idx)
	{
		const auto &model = progressive_x.getModels()[model_idx];
		poses.emplace_back(model.descriptor(0, 0));
		poses.emplace_back(model.descriptor(0, 1));
		poses.emplace_back(model.descriptor(0, 2));
		poses.emplace_back(model.descriptor(0, 3));
		poses.emplace_back(model.descriptor(1, 0));
		poses.emplace_back(model.descriptor(1, 1));
		poses.emplace_back(model.descriptor(1, 2));
		poses.emplace_back(model.descriptor(1, 3));
		poses.emplace_back(model.descriptor(2, 0));
		poses.emplace_back(model.descriptor(2, 1));
		poses.emplace_back(model.descriptor(2, 2));
		poses.emplace_back(model.descriptor(2, 3));
	}
	
	return progressive_x.getModelNumber();
}

int findHomographies_(
	std::vector<double>& correspondences,
	std::vector<size_t>& labeling,
	std::vector<double>& homographies,
	const size_t &source_image_width,
	const size_t &source_image_height,
	const size_t &destination_image_width,
	const size_t &destination_image_height,
	const double &spatial_coherence_weight,
	const double &threshold,
	const double &confidence,
	const double &neighborhood_ball_radius,
	const double &maximum_tanimoto_similarity,
	const size_t &max_iters,
	const size_t &minimum_point_number,
	const int &maximum_model_number,
	const size_t &sampler_id,
	const double &scoring_exponent,
	const bool do_logging)
{
	// Initialize Google's logging library.
	static bool isLoggingInitialized = false;
	if (!isLoggingInitialized)
	{
		google::InitGoogleLogging("pyprogessivex");
		isLoggingInitialized = true;
	}
	
	const size_t num_tents = correspondences.size() / 4;
		
	cv::Mat points(num_tents, 4, CV_64F, &correspondences[0]);
	
	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&points, // All data points
		neighborhood_ball_radius); // The radius of the neighborhood ball for determining the neighborhoods.

	// Initialize the samplers
	// The main sampler is used for sampling in the main RANSAC loop
	constexpr size_t kSampleSize = gcransac::utils::DefaultHomographyEstimator::sampleSize();
	typedef gcransac::sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Initializing a RANSAC-like uniformly random sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::UniformSampler(&points));
	else if (sampler_id == 1)  // Initializing a PROSAC sampler. This requires the points to be ordered according to the quality.
	{
		if (do_logging)
			printf("Note: PROSAC sampler requires the correspondences to be order by quality, e.g., SNN ratio.\n");
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ProsacSampler(&points, kSampleSize));
	}
	else if (sampler_id == 2) // Initializing a Progressive NAPSAC sampler. This requires the points to be ordered according to the quality.
	{
		if (do_logging)
			printf("Note: Progressive NAPSAC sampler requires the correspondences to be order by quality, e.g., SNN ratio.\n");
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ProgressiveNapsacSampler<4>(&points,
			{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
								// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
			kSampleSize, // The size of a minimal sample
			{ source_image_width, // The width of the source image
				source_image_height, // The height of the source image
				destination_image_width, // The width of the destination image
				destination_image_height }, // The height of the destination image
			0.5)); // The length (i.e., 0.5 * <point number> iterations) of fully blending to global sampling 
	}
	else if (sampler_id == 3) // Initializing a NAPSAC sampler
		main_sampler = std::unique_ptr<AbstractSampler>(
			new gcransac::sampler::NapsacSampler<gcransac::neighborhood::FlannNeighborhoodGraph>(&points, &neighborhood));
	else
	{
		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 2 (P-NAPSAC sampling)\n",
			sampler_id);
		return 0;
	}

	// The local optimization sampler is used inside the local optimization
	gcransac::sampler::UniformSampler local_optimization_sampler(&points);

	// Applying Progressive-X
	progx::ProgressiveX<gcransac::neighborhood::FlannNeighborhoodGraph, // The type of the used neighborhood-graph
		gcransac::utils::DefaultHomographyEstimator, // The type of the used model estimator
		AbstractSampler, // The type of the used main sampler in GC-RANSAC
		gcransac::sampler::UniformSampler> // The type of the used sampler in the local optimization of GC-RANSAC
		progressive_x(nullptr);

	// Set the parameters of Progressive-X
	progx::MultiModelSettings &settings = progressive_x.getMutableSettings();
	// The minimum number of inlier required to keep a model instance.
	// This value is used to determine the label cost weight in the alpha-expansion of PEARL.
	settings.minimum_number_of_inliers = minimum_point_number;
	// The inlier-outlier threshold
	settings.inlier_outlier_threshold = threshold;
	// The required confidence in the results
	settings.setConfidence(confidence);
	// The maximum Tanimoto similarity of the proposal and compound instances
	settings.maximum_tanimoto_similarity = maximum_tanimoto_similarity;
	// The weight of the spatial coherence term
	settings.spatial_coherence_weight = spatial_coherence_weight;
	// Setting the maximum iteration number
	settings.proposal_engine_settings.max_iteration_number = max_iters;
	// Setting the maximum model number if needed
	if (maximum_model_number > 0)
		settings.maximum_model_number = maximum_model_number;
	// Setting the scoring exponent
	progressive_x.setScoringExponent(scoring_exponent);

	progressive_x.run(points, // All data points
		neighborhood, // The neighborhood graph
		*main_sampler.get(), // The main sampler used in GC-RANSAC
		local_optimization_sampler); // The sampler used in the local optimization of GC-RANSAC
	
	// The obtained labeling
	labeling = progressive_x.getStatistics().labeling;

	homographies.reserve(9 * progressive_x.getModelNumber());
	
	// Saving the homography parameters
	for (size_t model_idx = 0; model_idx < progressive_x.getModelNumber(); ++model_idx)
	{
		const auto &model = progressive_x.getModels()[model_idx];
		homographies.emplace_back(model.descriptor(0, 0));
		homographies.emplace_back(model.descriptor(0, 1));
		homographies.emplace_back(model.descriptor(0, 2));
		homographies.emplace_back(model.descriptor(1, 0));
		homographies.emplace_back(model.descriptor(1, 1));
		homographies.emplace_back(model.descriptor(1, 2));
		homographies.emplace_back(model.descriptor(2, 0));
		homographies.emplace_back(model.descriptor(2, 1));
		homographies.emplace_back(model.descriptor(2, 2));
	}
	
	return progressive_x.getModelNumber();
}

int findPlanes_(
	std::vector<double>& input_points,
	std::vector<size_t>& labeling,
	std::vector<double>& planes,
	const double &spatial_coherence_weight,
	const double &threshold,
	const double &confidence,
	const double &neighborhood_ball_radius,
	const double &maximum_tanimoto_similarity,
	const size_t &max_iters,
	const size_t &minimum_point_number,
	const int &maximum_model_number,
	const size_t &sampler_id,
	const double &scoring_exponent,
	const bool do_logging)
{
	// Initialize Google's logging library.
	static bool isLoggingInitialized = false;
	if (!isLoggingInitialized)
	{
		google::InitGoogleLogging("pyprogessivex");
		isLoggingInitialized = true;
	}
	
	const size_t num_tents = input_points.size() / 3;
		
	cv::Mat points(num_tents, 3, CV_64F, &input_points[0]);
	
	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&points, // All data points
		neighborhood_ball_radius); // The radius of the neighborhood ball for determining the neighborhoods.

	// Initialize the samplers
	// The main sampler is used for sampling in the main RANSAC loop
	constexpr size_t kSampleSize = gcransac::utils::DefaultHomographyEstimator::sampleSize();
	typedef gcransac::sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Initializing a RANSAC-like uniformly random sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::UniformSampler(&points));
	else if (sampler_id == 1)  // Initializing a PROSAC sampler. This requires the points to be ordered according to the quality.
	{
		if (do_logging)
			printf("Note: PROSAC sampler requires the points to be order by quality, e.g., SNN ratio.\n");
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ProsacSampler(&points, kSampleSize));
	}
	else if (sampler_id == 2) // Initializing a NAPSAC sampler
		main_sampler = std::unique_ptr<AbstractSampler>(
			new gcransac::sampler::NapsacSampler<gcransac::neighborhood::FlannNeighborhoodGraph>(&points, &neighborhood));
	else
	{
		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 2 (P-NAPSAC sampling)\n",
			sampler_id);
		return 0;
	}

	// The local optimization sampler is used inside the local optimization
	gcransac::sampler::UniformSampler local_optimization_sampler(&points);

	// Applying Progressive-X
	progx::ProgressiveX<gcransac::neighborhood::FlannNeighborhoodGraph, // The type of the used neighborhood-graph
		gcransac::utils::Default3DPlaneEstimator, // The type of the used model estimator
		AbstractSampler, // The type of the used main sampler in GC-RANSAC
		gcransac::sampler::UniformSampler> // The type of the used sampler in the local optimization of GC-RANSAC
		progressive_x(nullptr);

	// Set the parameters of Progressive-X
	progx::MultiModelSettings &settings = progressive_x.getMutableSettings();
	// The minimum number of inlier required to keep a model instance.
	// This value is used to determine the label cost weight in the alpha-expansion of PEARL.
	settings.minimum_number_of_inliers = minimum_point_number;
	// The inlier-outlier threshold
	settings.inlier_outlier_threshold = threshold;
	// The required confidence in the results
	settings.setConfidence(confidence);
	// The maximum Tanimoto similarity of the proposal and compound instances
	settings.maximum_tanimoto_similarity = maximum_tanimoto_similarity;
	// The weight of the spatial coherence term
	settings.spatial_coherence_weight = spatial_coherence_weight;
	// Setting the maximum iteration number
	settings.proposal_engine_settings.max_iteration_number = max_iters;
	// Setting the maximum model number if needed
	if (maximum_model_number > 0)
		settings.maximum_model_number = maximum_model_number;
	// Setting the scoring exponent
	progressive_x.setScoringExponent(scoring_exponent);

	progressive_x.run(points, // All data points
		neighborhood, // The neighborhood graph
		*main_sampler.get(), // The main sampler used in GC-RANSAC
		local_optimization_sampler); // The sampler used in the local optimization of GC-RANSAC
	
	// The obtained labeling
	labeling = progressive_x.getStatistics().labeling;

	planes.reserve(4 * progressive_x.getModelNumber());
	
	// Saving the homography parameters
	for (size_t model_idx = 0; model_idx < progressive_x.getModelNumber(); ++model_idx)
	{
		const auto &model = progressive_x.getModels()[model_idx];
		planes.emplace_back(model.descriptor(0));
		planes.emplace_back(model.descriptor(1));
		planes.emplace_back(model.descriptor(2));
		planes.emplace_back(model.descriptor(3));
	}
	
	return progressive_x.getModelNumber();
}

int findVanishingPoints_(
	std::vector<double>& lines,
	std::vector<double>& weights,
	std::vector<size_t>& labeling,
	std::vector<double>& vanishing_points,
	const size_t &image_width,
	const size_t &image_height,
	const double &spatial_coherence_weight,
	const double &threshold,
	const double &confidence,
	const double &neighborhood_ball_radius,
	const double &maximum_tanimoto_similarity,
	const size_t &max_iters,
	const size_t &minimum_point_number,
	const int &maximum_model_number,
	const size_t &sampler_id,
	const double &scoring_exponent,
	const bool do_logging)
{
	// Initialize Google's logging library.
	static bool isLoggingInitialized = false;
	if (!isLoggingInitialized)
	{
		google::InitGoogleLogging("pyprogessivex");
		isLoggingInitialized = true;
	}
	
	const size_t num_lines = lines.size() / 4;
		
	cv::Mat points(num_lines, 4, CV_64F, &lines[0]);
	
	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&points, // All data points
		neighborhood_ball_radius); // The radius of the neighborhood ball for determining the neighborhoods.

	// The default estimator for homography fitting
	typedef gcransac::estimator::VanishingPointEstimator<
		gcransac::estimator::solver::VanishingPointTwoLineSolver, // The solver used for fitting a model to a minimal sample
		gcransac::estimator::solver::VanishingPointTwoLineSolver> // The solver used for fitting a model to a non-minimal sample
		DefaultVanishingPointEstimator;

	// Initialize the samplers
	// The main sampler is used for sampling in the main RANSAC loop
	constexpr size_t kSampleSize = DefaultVanishingPointEstimator::sampleSize();
	typedef gcransac::sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Initializing a RANSAC-like uniformly random sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::UniformSampler(&points));
	else if (sampler_id == 1)  // Initializing a PROSAC sampler. This requires the points to be ordered according to the quality.
	{
		if (do_logging)
			printf("Note: PROSAC sampler requires the correspondences to be order by quality, e.g., SNN ratio.\n");
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ProsacSampler(&points, kSampleSize));
	}
	else
	{
		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 2 (P-NAPSAC sampling)\n",
			sampler_id);
		return 0;
	}

	// The local optimization sampler is used inside the local optimization
	gcransac::sampler::UniformSampler local_optimization_sampler(&points);

	// Applying Progressive-X
	progx::ProgressiveX<gcransac::neighborhood::FlannNeighborhoodGraph, // The type of the used neighborhood-graph
		DefaultVanishingPointEstimator, // The type of the used model estimator
		AbstractSampler, // The type of the used main sampler in GC-RANSAC
		gcransac::sampler::UniformSampler> // The type of the used sampler in the local optimization of GC-RANSAC
		progressive_x(nullptr);

	// Set the parameters of Progressive-X
	progx::MultiModelSettings &settings = progressive_x.getMutableSettings();
	// Weights of the lines used in LSQ fitting
	settings.point_weights = weights;
	// The minimum number of inlier required to keep a model instance.
	// This value is used to determine the label cost weight in the alpha-expansion of PEARL.
	settings.minimum_number_of_inliers = minimum_point_number;
	// The inlier-outlier threshold
	settings.inlier_outlier_threshold = threshold;
	// The required confidence in the results
	settings.setConfidence(confidence);
	// The maximum Tanimoto similarity of the proposal and compound instances
	settings.maximum_tanimoto_similarity = maximum_tanimoto_similarity;
	// The weight of the spatial coherence term
	settings.spatial_coherence_weight = spatial_coherence_weight;
	// Setting the maximum iteration number
	settings.proposal_engine_settings.max_iteration_number = max_iters;
	// Setting the maximum model number if needed
	if (maximum_model_number > 0)
		settings.maximum_model_number = maximum_model_number;
	// Setting the scoring exponent
	progressive_x.setScoringExponent(scoring_exponent);
	// Set the logging parameter
	progressive_x.log(do_logging);

	progressive_x.run(points, // All data points
		neighborhood, // The neighborhood graph
		*main_sampler.get(), // The main sampler used in GC-RANSAC
		local_optimization_sampler); // The sampler used in the local optimization of GC-RANSAC
	
	// The obtained labeling
	labeling = progressive_x.getStatistics().labeling;

	vanishing_points.reserve(3 * progressive_x.getModelNumber());
	
	// Saving the homography parameters
	for (size_t model_idx = 0; model_idx < progressive_x.getModelNumber(); ++model_idx)
	{
		const auto &model = progressive_x.getModels()[model_idx];
		vanishing_points.emplace_back(model.descriptor(0));
		vanishing_points.emplace_back(model.descriptor(1));
		vanishing_points.emplace_back(model.descriptor(2));
	}
	
	return progressive_x.getModelNumber();
}


int findTwoViewMotions_(
	std::vector<double>& correspondences,
	std::vector<size_t>& labeling,
	std::vector<double>& motions,
	const size_t &source_image_width,
	const size_t &source_image_height,
	const size_t &destination_image_width,
	const size_t &destination_image_height,
	const double &spatial_coherence_weight,
	const double &threshold,
	const double &confidence,
	const double &neighborhood_ball_radius,
	const double &maximum_tanimoto_similarity,
	const size_t &max_iters,
	const size_t &minimum_point_number,
	const int &maximum_model_number,
	const size_t &sampler_id,
	const double &scoring_exponent,
	const bool do_logging)
{
	// Initialize Google's logging library.
	static bool isLoggingInitialized = false;
	if (!isLoggingInitialized)
	{
		google::InitGoogleLogging("pyprogessivex");
		isLoggingInitialized = true;
	}
	
	const size_t num_tents = correspondences.size() / 4;
		
	cv::Mat points(num_tents, 4, CV_64F, &correspondences[0]);
	
	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&points, // All data points
		neighborhood_ball_radius); // The radius of the neighborhood ball for determining the neighborhoods.

	// Initialize the samplers
	// The main sampler is used for sampling in the main RANSAC loop
	constexpr size_t kSampleSize = gcransac::utils::DefaultFundamentalMatrixEstimator::sampleSize();
	typedef gcransac::sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Initializing a RANSAC-like uniformly random sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::UniformSampler(&points));
	else if (sampler_id == 1)  // Initializing a PROSAC sampler. This requires the points to be ordered according to the quality.
	{
		if (do_logging)
			printf("Note: PROSAC sampler requires the correspondences to be order by quality, e.g., SNN ratio.\n");
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ProsacSampler(&points, kSampleSize));
	}
	else if (sampler_id == 2) // Initializing a Progressive NAPSAC sampler. This requires the points to be ordered according to the quality.
	{
		if (do_logging)
			printf("Note: Progressive NAPSAC sampler requires the correspondences to be order by quality, e.g., SNN ratio.\n");
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ProgressiveNapsacSampler<4>(&points,
			{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
								// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
			kSampleSize, // The size of a minimal sample
			{ source_image_width, // The width of the source image
				source_image_height, // The height of the source image
				destination_image_width, // The width of the destination image
				destination_image_height }, // The height of the destination image
			0.5)); // The length (i.e., 0.5 * <point number> iterations) of fully blending to global sampling 
	}
	else if (sampler_id == 3) // Initializing a NAPSAC sampler
		main_sampler = std::unique_ptr<AbstractSampler>(
			new gcransac::sampler::NapsacSampler<gcransac::neighborhood::FlannNeighborhoodGraph>(&points, &neighborhood));
	else
	{
		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 2 (P-NAPSAC sampling)\n",
			sampler_id);
		return 0;
	}

	// The local optimization sampler is used inside the local optimization
	gcransac::sampler::UniformSampler local_optimization_sampler(&points);

	// Applying Progressive-X
	progx::ProgressiveX<gcransac::neighborhood::FlannNeighborhoodGraph, // The type of the used neighborhood-graph
		gcransac::utils::DefaultFundamentalMatrixEstimator, // The type of the used model estimator
		AbstractSampler, // The type of the used main sampler in GC-RANSAC
		gcransac::sampler::UniformSampler> // The type of the used sampler in the local optimization of GC-RANSAC
		progressive_x(nullptr);

	// Set the parameters of Progressive-X
	progx::MultiModelSettings &settings = progressive_x.getMutableSettings();
	// The minimum number of inlier required to keep a model instance.
	// This value is used to determine the label cost weight in the alpha-expansion of PEARL.
	settings.minimum_number_of_inliers = minimum_point_number;
	// The inlier-outlier threshold
	settings.inlier_outlier_threshold = threshold;
	// The required confidence in the results
	settings.setConfidence(confidence);
	// The maximum Tanimoto similarity of the proposal and compound instances
	settings.maximum_tanimoto_similarity = maximum_tanimoto_similarity;
	// The weight of the spatial coherence term
	settings.spatial_coherence_weight = spatial_coherence_weight;
	// Setting the maximum iteration number
	settings.proposal_engine_settings.max_iteration_number = max_iters;
	// Setting the maximum model number if needed
	if (maximum_model_number > 0)
		settings.maximum_model_number = maximum_model_number;

	progressive_x.run(points, // All data points
		neighborhood, // The neighborhood graph
		*main_sampler.get(), // The main sampler used in GC-RANSAC
		local_optimization_sampler); // The sampler used in the local optimization of GC-RANSAC
	
	// The obtained labeling
	labeling = progressive_x.getStatistics().labeling;
	
	motions.reserve(9 * progressive_x.getModelNumber());
	
	// Saving the homography parameters
	for (size_t model_idx = 0; model_idx < progressive_x.getModelNumber(); ++model_idx)
	{
		const auto &model = progressive_x.getModels()[model_idx];
		motions.emplace_back(model.descriptor(0, 0));
		motions.emplace_back(model.descriptor(0, 1));
		motions.emplace_back(model.descriptor(0, 2));
		motions.emplace_back(model.descriptor(1, 0));
		motions.emplace_back(model.descriptor(1, 1));
		motions.emplace_back(model.descriptor(1, 2));
		motions.emplace_back(model.descriptor(2, 0));
		motions.emplace_back(model.descriptor(2, 1));
		motions.emplace_back(model.descriptor(2, 2));
	}
	
	return progressive_x.getModelNumber();
}