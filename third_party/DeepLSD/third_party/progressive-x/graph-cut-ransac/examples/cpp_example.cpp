#include <vector>	
#include <thread>
#include "utils.h"
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Eigen>

#include "GCRANSAC.h"

#include "neighborhood/flann_neighborhood_graph.h"
#include "neighborhood/grid_neighborhood_graph.h"

#include "samplers/uniform_sampler.h"
#include "samplers/prosac_sampler.h"
#include "samplers/napsac_sampler.h"
#include "samplers/progressive_napsac_sampler.h"

#include "estimators/fundamental_estimator.h"
#include "estimators/homography_estimator.h"
#include "estimators/essential_estimator.h"
#include "estimators/rigid_transformation_estimator.h"

#include "preemption/preemption_sprt.h"

#include "inlier_selectors/empty_inlier_selector.h"
#include "inlier_selectors/space_partitioning_ransac.h"

#include "estimators/solver_fundamental_matrix_seven_point.h"
#include "estimators/solver_fundamental_matrix_eight_point.h"
#include "estimators/solver_rigid_transformation_svd.h"
#include "estimators/solver_homography_four_point.h"
#include "estimators/solver_essential_matrix_five_point_stewenius.h"
#include "estimators/solver_p3p.h"
#include "estimators/solver_dls_pnp.h"

#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#endif 

struct stat info;

enum Problem {
	LineFitting,
	PerspectiveNPointFitting,
	FundamentalMatrixFitting,
	EssentialMatrixFitting,
	HomographyFitting,
	RigidTransformationFitting
};

// An example function showing how to fit essential matrix by Graph-Cut RANSAC
void testEssentialMatrixFitting(
	const std::string& source_path_, // The source image's path
	const std::string& destination_path_, // The destination image's path
	const std::string& source_intrinsics_path_, // Path where the intrinsics camera matrix of the source image is
	const std::string& destination_intrinsics_path_, // Path where the intrinsics camera matrix of the destination image is
	const std::string& out_correspondence_path_, // The path where the detected correspondences (before the robust estimation) will be saved (or loaded from if exists)
	const std::string& in_correspondence_path_, // The path where the inliers of the estimated fundamental matrices will be saved
	const std::string& output_match_image_path_, // The path where the matched image pair will be saved
	const double confidence_, // The RANSAC confidence value
	const double inlier_outlier_threshold_, // The used inlier-outlier threshold in GC-RANSAC.
	const double spatial_coherence_weight_, // The weight of the spatial coherence term in the graph-cut energy minimization.
	const size_t cell_number_in_neighborhood_graph_, // The radius of the neighborhood ball for determining the neighborhoods.
	const int fps_, // The required FPS limit. If it is set to -1, the algorithm will not be interrupted before finishing.
	const double minimum_inlier_ratio_for_sprt_ = 0.1); // An assumption about the minimum inlier ratio used for the SPRT test

// An example function showing how to fit fundamental matrix by Graph-Cut RANSAC
void testFundamentalMatrixFitting(
	const std::string& source_path_, // The source image's path
	const std::string& destination_path_, // The destination image's path
	const std::string& out_correspondence_path_, // The path where the detected correspondences (before the robust estimation) will be saved (or loaded from if exists)
	const std::string& in_correspondence_path_, // The path where the inliers of the estimated fundamental matrices will be saved
	const std::string& output_match_image_path_, // The path where the matched image pair will be saved
	const double confidence_, // The RANSAC confidence value
	const double inlier_outlier_threshold_, // The used inlier-outlier threshold in GC-RANSAC.
	const double spatial_coherence_weight_, // The weight of the spatial coherence term in the graph-cut energy minimization.
	const size_t cell_number_in_neighborhood_graph_, // The radius of the neighborhood ball for determining the neighborhoods.
	const int fps_, // The required FPS limit. If it is set to -1, the algorithm will not be interrupted before finishing.
	const double minimum_inlier_ratio_for_sprt_ = 0.1); // An assumption about the minimum inlier ratio used for the SPRT test

// An example function showing how to fit homography by Graph-Cut RANSAC
void testHomographyFitting(
	const std::string& source_path_, // The source image's path
	const std::string& destination_path_, // The destination image's path
	const std::string& out_correspondence_path_, // The path where the detected correspondences (before the robust estimation) will be saved (or loaded from if exists)
	const std::string& in_correspondence_path_, // The path where the inliers of the estimated fundamental matrices will be saved
	const std::string& output_match_image_path_, // The path where the matched image pair will be saved
	const double confidence_, // The RANSAC confidence value
	const double inlier_outlier_threshold_, // The used inlier-outlier threshold in GC-RANSAC.
	const double spatial_coherence_weight_, // The weight of the spatial coherence term in the graph-cut energy minimization.
	const size_t cell_number_in_neighborhood_graph_, // The radius of the neighborhood ball for determining the neighborhoods.
	const int fps_, // The required FPS limit. If it is set to -1, the algorithm will not be interrupted before finishing.
	const double minimum_inlier_ratio_for_sprt_ = 0.1); // An assumption about the minimum inlier ratio used for the SPRT test

// An example function showing how to fit 6D pose to 2D-3D correspondences by Graph-Cut RANSAC
void test6DPoseFitting(
	const std::string& intrinsics_path_, // The path where the intrinsic camera matrix can be found
	const std::string& ground_truth_pose_path_, // The path where the ground truth pose can be found
	const std::string& points_path_, // The path of the points 
	const std::string& inliers_point_path_, // The path where the inlier correspondences are saved
	const double confidence_, // The RANSAC confidence value
	const double inlier_outlier_threshold_, // The used inlier-outlier threshold in GC-RANSAC.
	const double spatial_coherence_weight_, // The weight of the spatial coherence term in the graph-cut energy minimization.
	const double sphere_radius_, // The radius of the sphere used for determining the neighborhood-graph
	const int fps_, // The required FPS limit. If it is set to -1, the algorithm will not be interrupted before finishing.
	const bool numerical_optimization_ = true, // A flag to decide if numerical optimization should be applied as a post-processing step
	const double minimum_inlier_ratio_for_sprt_ = 0.1); // An assumption about the minimum inlier ratio used for the SPRT test

// An example function showing how to fit calculate a rigid transformation from a set of 3D-3D correspondences by Graph-Cut RANSAC
void testRigidTransformFitting(
	const std::string& ground_truth_pose_path_, // The path where the ground truth pose can be found
	const std::string& points_path_, // The path of the points 
	const std::string& inliers_point_path_, // The path where the inlier correspondences are saved
	const double confidence_, // The RANSAC confidence value
	const double inlier_outlier_threshold_, // The used inlier-outlier threshold in GC-RANSAC.
	const double spatial_coherence_weight_, // The weight of the spatial coherence term in the graph-cut energy minimization.
	const double sphere_radius_,  // The radius of the sphere used for determining the neighborhood-graph
	const double minimum_inlier_ratio_for_sprt_ = 0.1); // An assumption about the minimum inlier ratio used for the SPRT test

// An example function showing how to fit calculate a 2D line to a set of 2D points by Graph-Cut RANSAC
void test2DLineFitting(
	const std::string& image_path_, // The path where the ground truth pose can be found
	const double confidence_, // The RANSAC confidence value
	const double inlier_outlier_threshold_, // The used inlier-outlier threshold in GC-RANSAC.
	const double spatial_coherence_weight_, // The weight of the spatial coherence term in the graph-cut energy minimization.
	const double sphere_radius_,  // The radius of the sphere used for determining the neighborhood-graph
	const double minimum_inlier_ratio_for_sprt_ = 0.1); // An assumption about the minimum inlier ratio used for the SPRT test

std::vector<std::string> getAvailableTestScenes(Problem problem_);

// Setting the paths for the data used in homography and fundamental matrix fitting
bool initializeScene(
	const std::string& scene_name_, // The name of the scene
	std::string& src_image_path_, // The path to the source image
	std::string& dst_image_path_, // The path to the destination image
	std::string& input_correspondence_path_, // The path to the correspondences
	std::string& output_correspondence_path_, // The path where the found inliers should be saved
	std::string& output_matched_image_path_, // The path where the matched image should be saved,
	const std::string root_directory_ = ""); // The root directory where the "results" and "data" folder are

// Setting the paths for the data used in essential matrix fitting
bool initializeScene(
	const std::string& scene_name_, // The name of the scene
	std::string& src_image_path_, // The path to the source image
	std::string& dst_image_path_, // The path to the destination image
	std::string& src_intrinsics_path_, // The path where the intrinsic parameters of the source camera can be found
	std::string& dst_intrinsics_path_, // The path where the intrinsic parameters of the destination camera can be found
	std::string& input_correspondence_path_, // The path to the correspondences
	std::string& output_correspondence_path_, // The path where the found inliers should be saved
	std::string& output_matched_image_path_, // The path where the matched image should be saved
	const std::string root_directory_ = ""); // The root directory where the "results" and "data" folder are

// Setting the paths for the data used in 6D pose fitting
bool initializeScenePnP(
	const std::string& scene_name_, // The name of the scene
	std::string& intrinsics_path_, // The path where the intrinsic parameters of the camera can be found
	std::string& ground_truth_pose_path_, // The path of the ground truth pose used for evaluating the results
	std::string& points_path_, // The path where the 2D-3D correspondences can be found
	std::string& inlier_points_path_, // The path where the inlier correspondences should be saved
	const std::string root_directory_ = ""); // The root directory where the "results" and "data" folder are

// Setting the paths for the data used in rigid transformation fitting
bool initializeSceneRigidPose(
	const std::string& scene_name_, // The name of the scene
	std::string& ground_truth_pose_path_, // The path of the ground truth pose used for evaluating the results
	std::string& points_path_, // The path where the 2D-3D correspondences can be found
	std::string& inlier_points_path_, // The path where the inlier correspondences should be saved
	const std::string root_directory_ = ""); // The root directory where the "results" and "data" folder are

using namespace gcransac;

int main(int argc, const char* argv[])
{
	srand(static_cast<int>(time(NULL)));

	const std::string data_directory = "";
	const double confidence = 0.99; // The RANSAC confidence value
	const int fps = -1; // The required FPS limit. If it is set to -1, the algorithm will not be interrupted before finishing.
	const double inlier_outlier_threshold_essential_matrix = 3.00; // The used inlier-outlier threshold in GC-RANSAC for essential matrix estimation.
	const double inlier_outlier_threshold_fundamental_matrix = 0.0003; // The used adaptive (i.e., it is the percentage of the maximum image diagonal) inlier-outlier threshold in GC-RANSAC for fundamental matrix estimation.
	const double inlier_outlier_threshold_rigid_pose = 10.0; // The used adaptive (i.e., it is the percentage of the maximum image diagonal) inlier-outlier threshold in GC-RANSAC for fundamental matrix estimation.
	const double inlier_outlier_threshold_2d_line = 2.0; // The used adaptive (i.e., it is the percentage of the maximum image diagonal) inlier-outlier threshold in GC-RANSAC for fundamental matrix estimation.
	const double inlier_outlier_threshold_homography = 2.00; // The used inlier-outlier threshold in GC-RANSAC for homography estimation.
	const double inlier_outlier_threshold_pnp = 5.50; // The used inlier-outlier threshold in GC-RANSAC for homography estimation.
	const double spatial_coherence_weight = 0.975; // The weigd_t of the spatial coherence term in the graph-cut energy minimization.
	const size_t cell_number_in_neighborhood_graph = 4; // The number of cells along each axis in the neighborhood graph.

	printf("------------------------------------------------------------\n2D line fitting\n------------------------------------------------------------\n");
	for (const std::string& scene : getAvailableTestScenes(Problem::LineFitting))
	{
		printf("Processed scene = '%s'\n", scene.c_str());
		// The path of the image which will be used for the example line fitting
		std::string image_path = data_directory + "data/" + scene + "/" + scene + "1.png";

		// Estimating a 2D line by the Graph-Cut RANSAC algorithm
		test2DLineFitting(
			image_path, // The path where the inlier points should be saved
			confidence, // The RANSAC confidence value
			inlier_outlier_threshold_2d_line, // The used inlier-outlier threshold in GC-RANSAC.
			0.8, // The weight of the spatial coherence term in the graph-cut energy minimization.
			20.0, // The radius of the neighborhood ball for determining the neighborhoods.
			0.001); // Minimum inlier ratio of an unknown model.
	}

	printf("------------------------------------------------------------\nRigid transformation fitting to 3D-3D correspondences\n------------------------------------------------------------\n");
	for (const std::string& scene : getAvailableTestScenes(Problem::RigidTransformationFitting))
	{
		printf("Processed scene = '%s'\n", scene.c_str());
		std::string points_path, // Path of the image and world points
			ground_truth_pose_path, // Path where the ground truth pose is found
			inlier_points_path; // Path where the inlier points are saved

		// Initializing the paths
		initializeSceneRigidPose(scene,
			ground_truth_pose_path,
			points_path,
			inlier_points_path,
			data_directory);

		// Estimating a rigid transformation by the Graph-Cut RANSAC algorithm
		testRigidTransformFitting(
			ground_truth_pose_path, // Path where the ground truth pose is found
			points_path, // The path where the image and world points can be found
			inlier_points_path, // The path where the inlier points should be saved
			confidence, // The RANSAC confidence value
			inlier_outlier_threshold_rigid_pose, // The used inlier-outlier threshold in GC-RANSAC.
			spatial_coherence_weight, // The weight of the spatial coherence term in the graph-cut energy minimization.
			20.0); // The radius of the neighborhood ball for determining the neighborhoods.
	}

	printf("------------------------------------------------------------\n6D pose fitting by the PnP algorithm\n------------------------------------------------------------\n");
	for (const std::string& scene : getAvailableTestScenes(Problem::PerspectiveNPointFitting))
	{
		printf("Processed scene = '%s'\n", scene.c_str());
		std::string points_path, // Path of the image and world points
			intrinsics_path, // Path where the intrinsics camera matrix 
			ground_truth_pose_path, // Path where the ground truth pose is found
			inlier_points_path; // Path where the inlier points are saved

		// Initializing the paths
		initializeScenePnP(scene,
			intrinsics_path,
			ground_truth_pose_path,
			points_path,
			inlier_points_path,
			data_directory);

		// Estimating the fundamental matrix by the Graph-Cut RANSAC algorithm
		test6DPoseFitting(
			intrinsics_path, // The path where the intrinsic camera matrix can be found
			ground_truth_pose_path, // Path where the ground truth pose is found
			points_path, // The path where the image and world points can be found
			inlier_points_path, // The path where the inlier points should be saved
			confidence, // The RANSAC confidence value
			inlier_outlier_threshold_pnp, // The used inlier-outlier threshold in GC-RANSAC.
			spatial_coherence_weight, // The weight of the spatial coherence term in the graph-cut energy minimization.
			20.0, // The radius of the neighborhood ball for determining the neighborhoods.
			fps); // The required FPS limit. If it is set to -1, the algorithm will not be interrupted before finishing.
		printf("\n------------------------------------------------------------\n");
	}

	printf("------------------------------------------------------------\nFundamental matrix fitting\n------------------------------------------------------------\n");
	for (const std::string& scene : getAvailableTestScenes(Problem::FundamentalMatrixFitting))
	{
		printf("Processed scene = '%s'\n", scene.c_str());
		std::string src_image_path, // Path of the source image
			dst_image_path, // Path of the destination image
			input_correspondence_path, // Path where the detected correspondences are saved
			output_correspondence_path, // Path where the inlier correspondences are saved
			output_matched_image_path; // Path where the matched image is saved

		// Initializing the paths
		initializeScene(scene,
			src_image_path,
			dst_image_path,
			input_correspondence_path,
			output_correspondence_path,
			output_matched_image_path,
			data_directory);

		// Estimating the fundamental matrix by the Graph-Cut RANSAC algorithm
		testFundamentalMatrixFitting(
			src_image_path, // The source image's path
			dst_image_path, // The destination image's path
			input_correspondence_path, // The path where the detected correspondences (before the robust estimation) will be saved (or loaded from if exists)
			output_correspondence_path, // The path where the inliers of the estimated fundamental matrices will be saved
			output_matched_image_path, // The path where the matched image pair will be saved
			confidence, // The RANSAC confidence value
			inlier_outlier_threshold_fundamental_matrix, // The used inlier-outlier threshold in GC-RANSAC.
			spatial_coherence_weight, // The weight of the spatial coherence term in the graph-cut energy minimization.
			cell_number_in_neighborhood_graph, // The radius of the neighborhood ball for determining the neighborhoods.
			fps); // The required FPS limit. If it is set to -1, the algorithm will not be interrupted before finishing.
		printf("\n------------------------------------------------------------\n");
	}

	printf("------------------------------------------------------------\nEssential matrix fitting\n------------------------------------------------------------\n");
	for (const std::string& scene : getAvailableTestScenes(Problem::EssentialMatrixFitting))
	{
		printf("Processed scene = '%s'\n", scene.c_str());
		std::string src_image_path, // Path of the source image
			dst_image_path, // Path of the destination image
			input_correspondence_path, // Path where the detected correspondences are saved
			output_correspondence_path, // Path where the inlier correspondences are saved
			output_matched_image_path, // Path where the matched image is saved
			src_intrinsics_path, // Path where the intrinsics camera matrix of the source image is
			dst_intrinsics_path; // Path where the intrinsics camera matrix of the destination image is

		// Initializing the paths
		initializeScene(scene,
			src_image_path,
			dst_image_path,
			src_intrinsics_path,
			dst_intrinsics_path,
			input_correspondence_path,
			output_correspondence_path,
			output_matched_image_path,
			data_directory);

		// Estimating the fundamental matrix by the Graph-Cut RANSAC algorithm
		testEssentialMatrixFitting(
			src_image_path, // The source image's path
			dst_image_path, // The destination image's path
			src_intrinsics_path, // Path where the intrinsics camera matrix of the source image is
			dst_intrinsics_path, // Path where the intrinsics camera matrix of the destination image is
			input_correspondence_path, // The path where the detected correspondences (before the robust estimation) will be saved (or loaded from if exists)
			output_correspondence_path, // The path where the inliers of the estimated fundamental matrices will be saved
			output_matched_image_path, // The path where the matched image pair will be saved
			confidence, // The RANSAC confidence value
			inlier_outlier_threshold_essential_matrix, // The used inlier-outlier threshold in GC-RANSAC.
			spatial_coherence_weight, // The weight of the spatial coherence term in the graph-cut energy minimization.
			cell_number_in_neighborhood_graph, // The radius of the neighborhood ball for determining the neighborhoods.
			fps); // The required FPS limit. If it is set to -1, the algorithm will not be interrupted before finishing.
		printf("\n------------------------------------------------------------\n");
	}

	printf("------------------------------------------------------------\nHomography fitting\n------------------------------------------------------------\n");
	for (const std::string& scene : getAvailableTestScenes(Problem::HomographyFitting))
	{
		printf("Processed scene = '%s'\n", scene.c_str());
		std::string src_image_path, // Path of the source image
			dst_image_path, // Path of the destination image
			input_correspondence_path, // Path where the detected correspondences are saved
			output_correspondence_path, // Path where the inlier correspondences are saved
			output_matched_image_path; // Path where the matched image is saved

		// Initializing the paths
		initializeScene(scene,
			src_image_path,
			dst_image_path,
			input_correspondence_path,
			output_correspondence_path,
			output_matched_image_path,
			data_directory);

		// Estimating the fundamental matrix by the Graph-Cut RANSAC algorithm
		testHomographyFitting(
			src_image_path, // The source image's path
			dst_image_path, // The destination image's path
			input_correspondence_path, // The path where the detected correspondences (before the robust estimation) will be saved (or loaded from if exists)
			output_correspondence_path, // The path where the inliers of the estimated fundamental matrices will be saved
			output_matched_image_path, // The path where the matched image pair will be saved
			confidence, // The RANSAC confidence value
			inlier_outlier_threshold_homography, // The used inlier-outlier threshold in GC-RANSAC.
			spatial_coherence_weight, // The weight of the spatial coherence term in the graph-cut energy minimization.
			cell_number_in_neighborhood_graph, // The radius of the neighborhood ball for determining the neighborhoods.
			fps); // The required FPS limit. If it is set to -1, the algorithm will not be interrupted before finishing.
		printf("\n------------------------------------------------------------\n");
	}

	return 0;
}

std::vector<std::string> getAvailableTestScenes(Problem problem_)
{
	switch (problem_)
	{
	case Problem::PerspectiveNPointFitting:
		return { "pose6dscene" };
	case Problem::LineFitting:
		return { "adam" };
	case Problem::FundamentalMatrixFitting:
		return { "head", "johnssona", "Kyoto" };
	case Problem::HomographyFitting:
		return { "graf", "Eiffel", "adam" };
	case Problem::RigidTransformationFitting:
		return { "kitchen" };
	default:
		return { "fountain" };
	}
}

bool initializeSceneRigidPose(
	const std::string& scene_name_,
	std::string& ground_truth_pose_path_,
	std::string& points_path_,
	std::string& inlier_image_points_path_,
	const std::string root_directory_)
{
	// The directory to which the results will be saved
	std::string results_dir = root_directory_ + "results";

	// Create the task directory if it doesn't exist
	if (stat(results_dir.c_str(), &info) != 0) // Check if exists
	{
#ifdef _WIN32 // Create a directory on Windows
		if (_mkdir(results_dir.c_str()) != 0) // Create it, if not
		{
			fprintf(stderr, "Error while creating folder 'results'\n");
			return false;
		}
#else // Create a directory on Linux
		if (mkdir(results_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
		{
			fprintf(stderr, "Error while creating a new folder in 'results'\n");
			return false;
		}
#endif
	}

	// The directory to which the results will be saved
	std::string dir = root_directory_ + "results/" + scene_name_;

	// Create the task directory if it doesn't exist
	if (stat(dir.c_str(), &info) != 0) // Check if exists
	{
#ifdef _WIN32 // Create a directory on Windows
		if (_mkdir(dir.c_str()) != 0) // Create it, if not
		{
			fprintf(stderr, "Error while creating a new folder in 'results'\n");
			return false;
		}
#else // Create a directory on Linux
		if (mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
		{
			fprintf(stderr, "Error while creating a new folder in 'results'\n");
			return false;
		}
#endif
	}

	// The path where the ground truth pose can be found
	ground_truth_pose_path_ =
		root_directory_ + "data/" + scene_name_ + "/" + scene_name_ + "_gt.txt";
	// The path where the 3D point cloud can be found
	points_path_ =
		root_directory_ + "data/" + scene_name_ + "/" + scene_name_ + "_points.txt";
	// The path where the inliers of the estimated fundamental matrices will be saved
	inlier_image_points_path_ =
		root_directory_ + "results/" + scene_name_ + "/result_" + scene_name_ + ".txt";

	return true;
}

bool initializeScenePnP(
	const std::string& scene_name_,
	std::string& intrinsics_path_,
	std::string& ground_truth_pose_path_,
	std::string& points_path_,
	std::string& inlier_image_points_path_,
	const std::string root_directory_) // The root directory where the "results" and "data" folder are
{
	// The directory to which the results will be saved
	std::string results_dir = root_directory_ + "results";

	// Create the task directory if it doesn't exist
	if (stat(results_dir.c_str(), &info) != 0) // Check if exists
	{
#ifdef _WIN32 // Create a directory on Windows
		if (_mkdir(results_dir.c_str()) != 0) // Create it, if not
		{
			fprintf(stderr, "Error while creating folder 'results'\n");
			return false;
		}
#else // Create a directory on Linux
		if (mkdir(results_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
		{
			fprintf(stderr, "Error while creating a new folder in 'results'\n");
			return false;
		}
#endif
	}

	// The directory to which the results will be saved
	std::string dir = root_directory_ + "results/" + scene_name_;

	// Create the task directory if it doesn't exist
	if (stat(dir.c_str(), &info) != 0) // Check if exists
	{
#ifdef _WIN32 // Create a directory on Windows
		if (_mkdir(dir.c_str()) != 0) // Create it, if not
		{
			fprintf(stderr, "Error while creating a new folder in 'results'\n");
			return false;
		}
#else // Create a directory on Linux
		if (mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
		{
			fprintf(stderr, "Error while creating a new folder in 'results'\n");
			return false;
		}
#endif
	}

	// The path where the intrinsics camera matrix of the source camera can be found
	intrinsics_path_ =
		root_directory_ + "data/" + scene_name_ + "/" + scene_name_ + ".K";
	// The path where the ground truth pose can be found
	ground_truth_pose_path_ =
		root_directory_ + "data/" + scene_name_ + "/" + scene_name_ + "_gt.txt";
	// The path where the 3D point cloud can be found
	points_path_ =
		root_directory_ + "data/" + scene_name_ + "/" + scene_name_ + "_points.txt";
	// The path where the inliers of the estimated fundamental matrices will be saved
	inlier_image_points_path_ =
		root_directory_ + "results/" + scene_name_ + "/result_" + scene_name_ + ".txt";

	return true;
}

bool initializeScene(
	const std::string& scene_name_, // The name of the scene
	std::string& src_image_path_, // The path to the source image
	std::string& dst_image_path_, // The path to the destination image
	std::string& input_correspondence_path_, // The path to the correspondences
	std::string& output_correspondence_path_, // The path where the found inliers should be saved
	std::string& output_matched_image_path_, // The path where the matched image should be saved,
	const std::string root_directory_) // The root directory where the "results" and "data" folder are
{
	// The directory to which the results will be saved
	std::string results_dir = root_directory_ + "results";

	// Create the task directory if it doesn't exist
	if (stat(results_dir.c_str(), &info) != 0) // Check if exists
	{
#ifdef _WIN32 // Create a directory on Windows
		if (_mkdir(results_dir.c_str()) != 0) // Create it, if not
		{
			fprintf(stderr, "Error while creating folder 'results'\n");
			return false;
		}
#else // Create a directory on Linux
		if (mkdir(results_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
		{
			fprintf(stderr, "Error while creating a new folder in 'results'\n");
			return false;
		}
#endif
	}

	// The directory to which the results will be saved
	std::string dir = root_directory_ + "results/" + scene_name_;

	// Create the task directory if it doesn't exist
	if (stat(dir.c_str(), &info) != 0) // Check if exists
	{
#ifdef _WIN32 // Create a directory on Windows
		if (_mkdir(dir.c_str()) != 0) // Create it, if not
		{
			fprintf(stderr, "Error while creating a new folder in 'results'\n");
			return false;
		}
#else // Create a directory on Linux
		if (mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
		{
			fprintf(stderr, "Error while creating a new folder in 'results'\n");
			return false;
		}
#endif
	}

	// The source image's path
	src_image_path_ =
		root_directory_ + "data/" + scene_name_ + "/" + scene_name_ + "1.jpg";
	if (cv::imread(src_image_path_).empty())
		src_image_path_ = root_directory_ + "data/" + scene_name_ + "/" + scene_name_ + "1.png";
	if (cv::imread(src_image_path_).empty())
	{
		fprintf(stderr, "Error while loading image '%s'\n", src_image_path_.c_str());
		return false;
	}

	// The destination image's path
	dst_image_path_ =
		root_directory_ + "data/" + scene_name_ + "/" + scene_name_ + "2.jpg";
	if (cv::imread(dst_image_path_).empty())
		dst_image_path_ = root_directory_ + "data/" + scene_name_ + "/" + scene_name_ + "2.png";
	if (cv::imread(dst_image_path_).empty())
	{
		fprintf(stderr, "Error while loading image '%s'\n", dst_image_path_.c_str());
		return false;
	}

	// The path where the detected correspondences (before the robust estimation) will be saved (or loaded from if exists)
	input_correspondence_path_ =
		root_directory_ + "results/" + scene_name_ + "/" + scene_name_ + "_points_with_no_annotation.txt";
	// The path where the inliers of the estimated fundamental matrices will be saved
	output_correspondence_path_ =
		root_directory_ + "results/" + scene_name_ + "/result_" + scene_name_ + ".txt";
	// The path where the matched image pair will be saved
	output_matched_image_path_ =
		root_directory_ + "results/" + scene_name_ + "/matches_" + scene_name_ + ".png";

	return true;
}

bool initializeScene(const std::string& scene_name_,
	std::string& src_image_path_,
	std::string& dst_image_path_,
	std::string& src_intrinsics_path_,
	std::string& dst_intrinsics_path_,
	std::string& input_correspondence_path_,
	std::string& output_correspondence_path_,
	std::string& output_matched_image_path_,
	const std::string root_directory_) // The root directory where the "results" and "data" folder are
{
	// The directory to which the results will be saved
	std::string results_dir = root_directory_ + "results";

	// Create the task directory if it doesn't exist
	if (stat(results_dir.c_str(), &info) != 0) // Check if exists
	{
#ifdef _WIN32 // Create a directory on Windows
		if (_mkdir(results_dir.c_str()) != 0) // Create it, if not
		{
			fprintf(stderr, "Error while creating folder 'results'\n");
			return false;
		}
#else // Create a directory on Linux
		if (mkdir(results_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
		{
			fprintf(stderr, "Error while creating a new folder in 'results'\n");
			return false;
		}
#endif
	}

	// The directory to which the results will be saved
	std::string dir = root_directory_ + "results/" + scene_name_;

	// Create the task directory if it doesn't exist
	if (stat(dir.c_str(), &info) != 0) // Check if exists
	{
#ifdef _WIN32 // Create a directory on Windows
		if (_mkdir(dir.c_str()) != 0) // Create it, if not
		{
			fprintf(stderr, "Error while creating a new folder in 'results'\n");
			return false;
		}
#else // Create a directory on Linux
		if (mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
		{
			fprintf(stderr, "Error while creating a new folder in 'results'\n");
			return false;
		}
#endif
	}

	// The source image's path
	src_image_path_ =
		root_directory_ + "data/" + scene_name_ + "/" + scene_name_ + "1.jpg";
	if (cv::imread(src_image_path_).empty())
		src_image_path_ = root_directory_ + "data/" + scene_name_ + "/" + scene_name_ + "1.png";
	if (cv::imread(src_image_path_).empty())
	{
		fprintf(stderr, "Error while loading image '%s'\n", src_image_path_.c_str());
		return false;
	}

	// The destination image's path
	dst_image_path_ =
		root_directory_ + "data/" + scene_name_ + "/" + scene_name_ + "2.jpg";
	if (cv::imread(dst_image_path_).empty())
		dst_image_path_ = root_directory_ + "data/" + scene_name_ + "/" + scene_name_ + "2.png";
	if (cv::imread(dst_image_path_).empty())
	{
		fprintf(stderr, "Error while loading image '%s'\n", dst_image_path_.c_str());
		return false;
	}

	// The path where the intrinsics camera matrix of the source camera can be found
	src_intrinsics_path_ =
		root_directory_ + "data/" + scene_name_ + "/" + scene_name_ + "1.K";
	// The path where the intrinsics camera matrix of the destination camera can be found
	dst_intrinsics_path_ =
		root_directory_ + "data/" + scene_name_ + "/" + scene_name_ + "2.K";
	// The path where the detected correspondences (before the robust estimation) will be saved (or loaded from if exists)
	input_correspondence_path_ =
		root_directory_ + "results/" + scene_name_ + "/" + scene_name_ + "_points_with_no_annotation.txt";
	// The path where the inliers of the estimated fundamental matrices will be saved
	output_correspondence_path_ =
		root_directory_ + "results/" + scene_name_ + "/result_" + scene_name_ + ".txt";
	// The path where the matched image pair will be saved
	output_matched_image_path_ =
		root_directory_ + "results/" + scene_name_ + "/matches_" + scene_name_ + ".png";

	return true;
}

// An example function showing how to fit calculate a 2D line to a set of 2D points by Graph-Cut RANSAC
void test2DLineFitting(
	const std::string& image_path_, // The path where the ground truth pose can be found
	const double confidence_, // The RANSAC confidence value
	const double inlier_outlier_threshold_, // The used inlier-outlier threshold in GC-RANSAC.
	const double spatial_coherence_weight_, // The weight of the spatial coherence term in the graph-cut energy minimization.
	const double sphere_radius_,  // The radius of the sphere used for determining the neighborhood-graph
	const double minimum_inlier_ratio_for_sprt_) // An assumption about the minimum inlier ratio used for the SPRT test
{
	// Load the image
	cv::Mat image = cv::imread(image_path_),
		image_gray;

	// Checking if the image has been loaded
	if (image.empty())
	{
		fprintf(stderr, "Image '%s' has not been loaded correctly.\n", image_path_.c_str());
		return;
	}

	// Applying Canny edge detection
	constexpr double
		low_threshold = 50,
		max_low_threshold = 100,
		kernel_size = 3,
		ratio = 3;

	cv::Mat detected_edges;
	cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
	cv::blur(image_gray, detected_edges, cv::Size(3, 3));

	cv::Canny(detected_edges,
		detected_edges,
		low_threshold,
		low_threshold * ratio,
		kernel_size);

	// Storing the points for line fitting
	cv::Mat points(0, 2, CV_64F),
		point(1, 2, CV_64F);
	for (size_t row = 0; row < detected_edges.rows; row += 1)
		for (size_t col = 0; col < detected_edges.cols; col += 1)
			if (detected_edges.at<uchar>(row, col) > std::numeric_limits<int>::epsilon())
			{
				point.at<double>(0) = col;
				point.at<double>(1) = row;
				points.push_back(point);
			}

	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	neighborhood::GridNeighborhoodGraph<2> neighborhood(&points, // The data points
		{ image.cols / 16.0,
			image.rows / 16.0 },
		16); // The radius of the neighborhood sphere used for determining the neighborhood structure
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
	printf("Neighborhood calculation time = %f secs\n", elapsed_seconds.count());

	// Apply Graph-cut RANSAC
	utils::Default2DLineEstimator estimator; // The estimator used for the pose fitting
	Line2D model; // The estimated model parameters

	// Initialize the samplers
	// The main sampler is used inside the local optimization
	// The main sampler is used inside the local optimization
	sampler::ProgressiveNapsacSampler<2> main_sampler(&points,
		{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
							// (image_width / 16) * (image_height / 16), etc.
		estimator.sampleSize(), // The size of a minimal sample
		{ static_cast<double>(image.cols), // The width of the image
			static_cast<double>(image.rows) });  // The height of the image
	sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!main_sampler.isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return;
	}

	// Initializing SPRT test
	preemption::SPRTPreemptiveVerfication<utils::Default2DLineEstimator> preemptive_verification(
		points, // The set of 2D points
		estimator, // The line estimator object
		minimum_inlier_ratio_for_sprt_); // The minimum acceptable inlier ratio. Models with fewer inliers will not be accepted.

	// Initializing the fast inlier selector object
	inlier_selector::EmptyInlierSelector<utils::Default2DLineEstimator, 
		neighborhood::GridNeighborhoodGraph<2>> inlier_selector(&neighborhood);

	GCRANSAC<utils::Default2DLineEstimator,
		neighborhood::GridNeighborhoodGraph<2>,
		MSACScoringFunction<utils::Default2DLineEstimator>,
		preemption::SPRTPreemptiveVerfication<utils::Default2DLineEstimator>> gcransac;
	gcransac.settings.threshold = inlier_outlier_threshold_; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = spatial_coherence_weight_; // The weight of the spatial coherence term
	gcransac.settings.confidence = confidence_; // The required confidence in the results
	gcransac.settings.max_iteration_number = 5000; // The maximum number of iterations
	gcransac.settings.min_iteration_number = 20; // The minimum number of iterations

	// Start GC-RANSAC
	gcransac.run(points, // The normalized points
		estimator,  // The estimator
		&main_sampler, // The sample used for selecting minimal samples in the main iteration
		&local_optimization_sampler, // The sampler used for selecting a minimal sample when doing the local optimization
		&neighborhood, // The neighborhood-graph
		model, // The obtained model parameters
		preemptive_verification,
		inlier_selector);

	// Get the statistics of the results
	const utils::RANSACStatistics& statistics = gcransac.getRansacStatistics();

	printf("Elapsed time = %f secs\n", statistics.processing_time);
	printf("Inlier number before = %d\n", static_cast<int>(statistics.inliers.size()));
	printf("Applied number of local optimizations = %d\n", static_cast<int>(statistics.local_optimization_number));
	printf("Applied number of graph-cuts = %d\n", static_cast<int>(statistics.graph_cut_number));
	printf("Number of iterations = %d\n\n", static_cast<int>(statistics.iteration_number));

	// Draw the inlier to the image
	for (const auto& inlierIdx : statistics.inliers)
		cv::circle(image,
			cv::Point(points.at<double>(inlierIdx, 0), points.at<double>(inlierIdx, 1)),
			2,
			cv::Scalar(0, 255, 0),
			-1);

	// Draw the line to the image
	double x1 = 0,
		x2 = image.cols;
	double y1 = -(model.descriptor(0) * x1 + model.descriptor(2)) / model.descriptor(1),
		y2 = -(model.descriptor(0) * x2 + model.descriptor(2)) / model.descriptor(1);
	cv::line(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2);

	printf("Press a button to continue.\n");
	cv::imshow("Detected edges", detected_edges);
	cv::imshow("Found line", image);
	cv::waitKey(0);

	// Cleaning up
	cv::destroyWindow("Detected edges");
	cv::destroyWindow("Found line");
	image.release();
}

// An example function showing how to fit calculate a rigid transformation from a set of 3D-3D correspondences by Graph-Cut RANSAC
void testRigidTransformFitting(
	const std::string& ground_truth_pose_path_, // The path where the ground truth pose can be found
	const std::string& points_path_, // The path of the points 
	const std::string& inliers_point_path_, // The path where the inlier correspondences are saved
	const double confidence_, // The RANSAC confidence value
	const double inlier_outlier_threshold_, // The used inlier-outlier threshold in GC-RANSAC.
	const double spatial_coherence_weight_, // The weight of the spatial coherence term in the graph-cut energy minimization.
	const double sphere_radius_,  // The radius of the sphere used for determining the neighborhood-graph
	const double minimum_inlier_ratio_for_sprt_) // An assumption about the minimum inlier ratio used for the SPRT test
{
	// The image and world points stored as rows in the matrix. Each row is of format 'u v x y z'.
	cv::Mat points;

	// Loading the 3D-3D correspondences 
	gcransac::utils::loadPointsFromFile<6, 1, false>(points, // The points in the image
		points_path_.c_str()); // The path where the image points are stored

	// Load the ground truth pose
	Eigen::Matrix<double, 8, 4> reference_pose;
	if (!utils::loadMatrix<double, 8, 4>(ground_truth_pose_path_,
		reference_pose))
	{
		printf("An error occured when loading the reference camera pose from '%s'\n",
			ground_truth_pose_path_.c_str());
		return;
	}

	const Eigen::Matrix4d& initial_T =
		reference_pose.block<4, 4>(0, 0);
	const Eigen::Matrix4d& ground_truth_T =
		reference_pose.block<4, 4>(4, 0);

	// Transform the point by the initial transformation
	for (size_t point_idx = 0; point_idx < points.rows; ++point_idx)
	{
		const double
			x = points.at<double>(point_idx, 0),
			y = points.at<double>(point_idx, 1),
			z = points.at<double>(point_idx, 2);

		points.at<double>(point_idx, 0) = initial_T(0, 0) * x + initial_T(1, 0) * y + initial_T(2, 0) * z + initial_T(3, 0);
		points.at<double>(point_idx, 1) = initial_T(0, 1) * x + initial_T(1, 1) * y + initial_T(2, 1) * z + initial_T(3, 1);
		points.at<double>(point_idx, 2) = initial_T(0, 2) * x + initial_T(1, 2) * y + initial_T(2, 2) * z + initial_T(3, 2);
	}

	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	// TODO: generalize the grid class to handle not only point correspondences
	neighborhood::FlannNeighborhoodGraph neighborhood(&points, // The data points
		sphere_radius_); // The radius of the neighborhood sphere used for determining the neighborhood structure
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
	printf("Neighborhood calculation time = %f secs\n", elapsed_seconds.count());

	// Apply Graph-cut RANSAC
	utils::DefaultRigidTransformationEstimator estimator; // The estimator used for the pose fitting
	RigidTransformation model; // The estimated model parameters

	// Initialize the samplers
	// The main sampler is used inside the local optimization
	sampler::UniformSampler main_sampler(&points); // The data points
	sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!main_sampler.isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return;
	}

	// Initializing SPRT test
	preemption::SPRTPreemptiveVerfication<utils::DefaultRigidTransformationEstimator> preemptive_verification(
		points,
		estimator,
		minimum_inlier_ratio_for_sprt_);

	// Initializing the fast inlier selector object
	inlier_selector::EmptyInlierSelector<utils::DefaultRigidTransformationEstimator, 
		neighborhood::FlannNeighborhoodGraph> inlier_selector(&neighborhood);

	GCRANSAC<utils::DefaultRigidTransformationEstimator,
		neighborhood::FlannNeighborhoodGraph,
		MSACScoringFunction<utils::DefaultRigidTransformationEstimator>,
		preemption::SPRTPreemptiveVerfication<utils::DefaultRigidTransformationEstimator>> gcransac;
	gcransac.settings.threshold = inlier_outlier_threshold_; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = spatial_coherence_weight_; // The weight of the spatial coherence term
	gcransac.settings.confidence = confidence_; // The required confidence in the results
	gcransac.settings.max_iteration_number = 5000; // The maximum number of iterations
	gcransac.settings.min_iteration_number = 20; // The minimum number of iterations
	gcransac.settings.neighborhood_sphere_radius = sphere_radius_; // The radius of the neighborhood ball

	// Start GC-RANSAC
	gcransac.run(points, // The normalized points
		estimator,  // The estimator
		&main_sampler, // The sample used for selecting minimal samples in the main iteration
		&local_optimization_sampler, // The sampler used for selecting a minimal sample when doing the local optimization
		&neighborhood, // The neighborhood-graph
		model, // The obtained model parameters
		preemptive_verification,
		inlier_selector);

	// Get the statistics of the results
	const utils::RANSACStatistics& statistics = gcransac.getRansacStatistics();

	printf("Elapsed time = %f secs\n", statistics.processing_time);
	printf("Inlier number before = %d\n", static_cast<int>(statistics.inliers.size()));
	printf("Applied number of local optimizations = %d\n", static_cast<int>(statistics.local_optimization_number));
	printf("Applied number of graph-cuts = %d\n", static_cast<int>(statistics.graph_cut_number));
	printf("Number of iterations = %d\n\n", static_cast<int>(statistics.iteration_number));

	// Save the inliers to file
	utils::savePointsToFile(points, // The loaded data points
		inliers_point_path_.c_str(), // The path where the results should be saved
		&statistics.inliers); // The set of inlier found

	model.descriptor =
		initial_T * model.descriptor;

	// The estimated rotation matrix
	Eigen::Matrix3d rotation =
		model.descriptor.block<3, 3>(0, 0);
	// The estimated translation
	Eigen::Vector3d translation =
		model.descriptor.block<1, 3>(3, 0);

	// The number of inliers found
	const size_t inlier_number = statistics.inliers.size();

	// Calculate the estimation error
	constexpr double radian_to_degree_multiplier = 180.0 / M_PI;
	const double angular_error = radian_to_degree_multiplier * (Eigen::Quaterniond(
		rotation).angularDistance(Eigen::Quaterniond(ground_truth_T.block<3, 3>(0, 0))));
	const double translation_error = (ground_truth_T.block<1, 3>(3, 0) - translation.transpose()).norm();

	printf("The error in the rotation matrix = %f degrees\n", angular_error);
	printf("The error in the translation = %f\n", translation_error / 10.0);
}

void testHomographyFitting(
	const std::string& source_path_,
	const std::string& destination_path_,
	const std::string& out_correspondence_path_,
	const std::string& in_correspondence_path_,
	const std::string& output_match_image_path_,
	const double confidence_,
	const double inlier_outlier_threshold_,
	const double spatial_coherence_weight_,
	const size_t cell_number_in_neighborhood_graph_,
	const int fps_,
	const double minimum_inlier_ratio_for_sprt_) // An assumption about the minimum inlier ratio used for the SPRT test
{
	// Read the images
	cv::Mat source_image = cv::imread(source_path_); // The source image
	cv::Mat destination_image = cv::imread(destination_path_); // The destination image

	if (source_image.empty()) // Check if the source image is loaded successfully
	{
		printf("An error occured while loading image '%s'\n",
			source_path_.c_str());
		return;
	}

	if (destination_image.empty()) // Check if the destination image is loaded successfully
	{
		printf("An error occured while loading image '%s'\n",
			destination_path_.c_str());
		return;
	}

	// Detect or load point correspondences using AKAZE 
	cv::Mat points;
	utils::detectFeatures(
		in_correspondence_path_, // The path where the correspondences are read from or saved to.
		source_image, // The source image
		destination_image, // The destination image
		points); // The detected point correspondences. Each row is of format "x1 y1 x2 y2"

	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	neighborhood::GridNeighborhoodGraph<4> neighborhood(&points,
		{ source_image.cols / static_cast<double>(cell_number_in_neighborhood_graph_),
			source_image.rows / static_cast<double>(cell_number_in_neighborhood_graph_),
			destination_image.cols / static_cast<double>(cell_number_in_neighborhood_graph_),
			destination_image.rows / static_cast<double>(cell_number_in_neighborhood_graph_) },
		cell_number_in_neighborhood_graph_);
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
	printf("Neighborhood calculation time = %f secs\n", elapsed_seconds.count());

	// Checking if the neighborhood graph is initialized successfully.
	if (!neighborhood.isInitialized())
	{
		fprintf(stderr, "The neighborhood graph is not initialized successfully.\n");
		return;
	}

	// Apply Graph-cut RANSAC
	utils::DefaultHomographyEstimator estimator;
	std::vector<int> inliers;
	Model model;

	// Initializing SPRT test
	/*preemption::SPRTPreemptiveVerfication<utils::DefaultHomographyEstimator> preemptive_verification(
		points,
		estimator,
		minimum_inlier_ratio_for_sprt_);*/
	preemption::EmptyPreemptiveVerfication<utils::DefaultHomographyEstimator> preemptive_verification;

	// Initializing the fast inlier selector object
	inlier_selector::SpacePartitioningRANSAC<utils::DefaultHomographyEstimator, 
		neighborhood::GridNeighborhoodGraph<4>> inlier_selector(&neighborhood);

	GCRANSAC<utils::DefaultHomographyEstimator,
		neighborhood::GridNeighborhoodGraph<4>,
		MSACScoringFunction<utils::DefaultHomographyEstimator>,
		preemption::EmptyPreemptiveVerfication<utils::DefaultHomographyEstimator>,
		inlier_selector::SpacePartitioningRANSAC<utils::DefaultHomographyEstimator, neighborhood::GridNeighborhoodGraph<4>>> gcransac;
	gcransac.setFPS(fps_); // Set the desired FPS (-1 means no limit)
	gcransac.settings.threshold = inlier_outlier_threshold_; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = spatial_coherence_weight_; // The weight of the spatial coherence term
	gcransac.settings.confidence = confidence_; // The required confidence in the results
	gcransac.settings.max_local_optimization_number = 50; // The maximm number of local optimizations
	gcransac.settings.max_iteration_number = 5000; // The maximum number of iterations
	gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
	gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball
	gcransac.settings.core_number = std::thread::hardware_concurrency(); // The number of parallel processes

	// Initialize the samplers
	// The main sampler is used inside the local optimization
	sampler::ProgressiveNapsacSampler<4> main_sampler(&points,
		{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
							// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
		estimator.sampleSize(), // The size of a minimal sample
		{ static_cast<double>(source_image.cols), // The width of the source image
			static_cast<double>(source_image.rows), // The height of the source image
			static_cast<double>(destination_image.cols), // The width of the destination image
			static_cast<double>(destination_image.rows) },  // The height of the destination image
		0.5); // The length (i.e., 0.5 * <point number> iterations) of fully blending to global sampling 

	sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!main_sampler.isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return;
	}

	// Start GC-RANSAC
	gcransac.run(points,
		estimator,
		&main_sampler,
		&local_optimization_sampler,
		&neighborhood,
		model,
		preemptive_verification,
		inlier_selector);

	// Get the statistics of the results
	const utils::RANSACStatistics& statistics = gcransac.getRansacStatistics();

	// Write statistics
	printf("Elapsed time = %f secs\n", statistics.processing_time);
	printf("Inlier number = %d\n", static_cast<int>(statistics.inliers.size()));
	printf("Applied number of local optimizations = %d\n", static_cast<int>(statistics.local_optimization_number));
	printf("Applied number of graph-cuts = %d\n", static_cast<int>(statistics.graph_cut_number));
	printf("Number of iterations = %d\n\n", static_cast<int>(statistics.iteration_number));

	// Draw the inlier matches to the images
	cv::Mat match_image;
	utils::drawMatches(points,
		statistics.inliers,
		source_image,
		destination_image,
		match_image);

	printf("Saving the matched images to file '%s'.\n", output_match_image_path_.c_str());
	imwrite(output_match_image_path_, match_image); // Save the matched image to file
	printf("Saving the inlier correspondences to file '%s'.\n", out_correspondence_path_.c_str());
	utils::savePointsToFile(points, out_correspondence_path_.c_str(), &statistics.inliers); // Save the inliers to file

	printf("Press a button to continue...\n");

	// Showing the image
	utils::showImage(match_image,
		"Inlier correspondences",
		1600,
		1200,
		true);
}

void testFundamentalMatrixFitting(
	const std::string& source_path_,
	const std::string& destination_path_,
	const std::string& out_correspondence_path_,
	const std::string& in_correspondence_path_,
	const std::string& output_match_image_path_,
	const double confidence_,
	const double inlier_outlier_threshold_,
	const double spatial_coherence_weight_,
	const size_t cell_number_in_neighborhood_graph_,
	const int fps_,
	const double minimum_inlier_ratio_for_sprt_) // An assumption about the minimum inlier ratio used for the SPRT test
{
	// Read the images
	cv::Mat source_image = cv::imread(source_path_);
	cv::Mat destination_image = cv::imread(destination_path_);

	if (source_image.empty()) // Check if the source image is loaded successfully
	{
		printf("An error occured while loading image '%s'\n",
			source_path_.c_str());
		return;
	}

	if (destination_image.empty()) // Check if the destination image is loaded successfully
	{
		printf("An error occured while loading image '%s'\n",
			destination_path_.c_str());
		return;
	}

	// Detect or load point correspondences using AKAZE 
	cv::Mat points;
	utils::detectFeatures(
		in_correspondence_path_, // The path where the correspondences are read from or saved to.
		source_image, // The source image
		destination_image, // The destination image
		points); // The detected point correspondences. Each row is of format "x1 y1 x2 y2"

	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	neighborhood::GridNeighborhoodGraph<4> neighborhood(&points,
		{ source_image.cols / static_cast<double>(cell_number_in_neighborhood_graph_),
			source_image.rows / static_cast<double>(cell_number_in_neighborhood_graph_),
			destination_image.cols / static_cast<double>(cell_number_in_neighborhood_graph_),
			destination_image.rows / static_cast<double>(cell_number_in_neighborhood_graph_) },
		cell_number_in_neighborhood_graph_);
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
	printf("Neighborhood calculation time = %f secs\n", elapsed_seconds.count());

	// Checking if the neighborhood graph is initialized successfully.
	if (!neighborhood.isInitialized())
	{
		fprintf(stderr, "The neighborhood graph is not initialized successfully.\n");
		return;
	}

	// Calculating the maximum image diagonal to be used for setting the threshold
	// adaptively for each image pair. 
	const double max_image_diagonal =
		sqrt(pow(MAX(source_image.cols, destination_image.cols), 2) + pow(MAX(source_image.rows, destination_image.rows), 2));

	// Apply Graph-cut RANSAC
	utils::DefaultFundamentalMatrixEstimator estimator;
	std::vector<int> inliers;
	FundamentalMatrix model;

	// Initializing SPRT test
	preemption::SPRTPreemptiveVerfication<utils::DefaultFundamentalMatrixEstimator> preemptive_verification(
		points,
		estimator,
		minimum_inlier_ratio_for_sprt_);
	
	// Initializing the fast inlier selector object
	inlier_selector::EmptyInlierSelector<utils::DefaultFundamentalMatrixEstimator, 
		neighborhood::GridNeighborhoodGraph<4>> inlier_selector(&neighborhood);

	// Initialize the samplers
	// The main sampler is used inside the local optimization
	sampler::ProgressiveNapsacSampler<4> main_sampler(&points,
		{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
							// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
		estimator.sampleSize(), // The size of a minimal sample
		{ static_cast<double>(source_image.cols), // The width of the source image
			static_cast<double>(source_image.rows), // The height of the source image
			static_cast<double>(destination_image.cols), // The width of the destination image
			static_cast<double>(destination_image.rows) });  // The height of the destination image
	sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!main_sampler.isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return;
	}

	GCRANSAC<utils::DefaultFundamentalMatrixEstimator,
		neighborhood::GridNeighborhoodGraph<4>,
		MSACScoringFunction<utils::DefaultFundamentalMatrixEstimator>,
		preemption::SPRTPreemptiveVerfication<utils::DefaultFundamentalMatrixEstimator>> gcransac;
	gcransac.settings.threshold = inlier_outlier_threshold_ * max_image_diagonal; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = spatial_coherence_weight_; // The weight of the spatial coherence term
	gcransac.settings.confidence = confidence_; // The required confidence in the results
	gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
	gcransac.settings.max_iteration_number = 5000; // The maximum number of iterations
	gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
	gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball

	// Printinf the actually used threshold
	printf("Used threshold is %.2f pixels (%.2f%% of the image diagonal)\n",
		gcransac.settings.threshold, 100.0 * inlier_outlier_threshold_);

	// Start GC-RANSAC
	gcransac.run(points,
		estimator,
		&main_sampler,
		&local_optimization_sampler,
		&neighborhood,
		model,
		preemptive_verification,
		inlier_selector);

	// Get the statistics of the results
	const utils::RANSACStatistics& statistics = gcransac.getRansacStatistics();

	// Write statistics
	printf("Elapsed time = %f secs\n", statistics.processing_time);
	printf("Inlier number = %d\n", static_cast<int>(statistics.inliers.size()));
	printf("Applied number of local optimizations = %d\n", static_cast<int>(statistics.local_optimization_number));
	printf("Applied number of graph-cuts = %d\n", static_cast<int>(statistics.graph_cut_number));
	printf("Number of iterations = %d\n\n", static_cast<int>(statistics.iteration_number));

	// Draw the inlier matches to the images
	cv::Mat match_image;
	utils::drawMatches(points,
		statistics.inliers,
		source_image,
		destination_image,
		match_image);

	printf("Saving the matched images to file '%s'.\n",
		output_match_image_path_.c_str());
	imwrite(output_match_image_path_, match_image); // Save the matched image to file
	printf("Saving the inlier correspondences to file '%s'.\n",
		out_correspondence_path_.c_str());
	utils::savePointsToFile(points, out_correspondence_path_.c_str(), &statistics.inliers); // Save the inliers to file

	printf("Press a button to continue...\n");

	// Showing the image
	utils::showImage(match_image,
		"Inlier correspondences",
		1600,
		1200,
		true);
}

void test6DPoseFitting(
	const std::string& intrinsics_path_,
	const std::string& ground_truth_pose_path_,
	const std::string& points_path_,
	const std::string& inlier_image_points_path_,
	const double confidence_,
	const double inlier_outlier_threshold_,
	const double spatial_coherence_weight_,
	const double sphere_radius_,
	const int fps_,
	const bool numerical_optimization_, // A flag to decide if numerical optimization should be applied as a post-processing step)
	const double minimum_inlier_ratio_for_sprt_) // An assumption about the minimum inlier ratio used for the SPRT test
{
	// The image and world points stored as rows in the matrix. Each row is of format 'u v x y z'.
	cv::Mat points;

	// Loading the image and world points
	gcransac::utils::loadPointsFromFile<5>(points, // The points in the image
		points_path_.c_str()); // The path where the image points are stored

	// Load the intrinsic camera matrix
	Eigen::Matrix3d intrinsics;

	if (!utils::loadMatrix<double, 3, 3>(intrinsics_path_,
		intrinsics))
	{
		printf("An error occured when loading the intrinsics camera matrix from '%s'\n",
			intrinsics_path_.c_str());
		return;
	}

	// Load the ground truth pose
	Eigen::Matrix<double, 3, 4> reference_pose;
	if (!utils::loadMatrix<double, 3, 4>(ground_truth_pose_path_,
		reference_pose))
	{
		printf("An error occured when loading the reference camera pose from '%s'\n",
			ground_truth_pose_path_.c_str());
		return;
	}

	// The ground truth rotation matrix
	const Eigen::Matrix3d& gt_rotation =
		reference_pose.leftCols<3>();

	// The ground truth translation matrix
	const Eigen::Vector3d& gt_translation =
		reference_pose.rightCols<1>();

	// Normalize the point coordinate by the intrinsic matrix
	cv::Rect roi(0, 0, 2, points.rows); // A rectangle covering the 2D image points in the matrix
	cv::Mat normalized_points = points.clone(); // The 2D image points normalized by the intrinsic camera matrix
	// Normalizing the image points by the camera matrix
	utils::normalizeImagePoints(
		points(roi), // The loaded image points
		intrinsics, // The intrinsic camera matrix
		normalized_points); // The normalized points

	// Normalize the threshold by the average of the focal lengths
	const double avg_focal_length =
		(intrinsics(0, 0) + intrinsics(1, 1)) / 2.0;
	const double normalized_threshold =
		inlier_outlier_threshold_ / avg_focal_length;

	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	// TODO: generalize the grid class to handle not only point correspondences
	neighborhood::FlannNeighborhoodGraph neighborhood(&points, // The data points
		sphere_radius_); // The radius of the neighborhood sphere used for determining the neighborhood structure
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
	printf("Neighborhood calculation time = %f secs\n", elapsed_seconds.count());

	// Apply Graph-cut RANSAC
	utils::DefaultPnPEstimator estimator; // The estimator used for the pose fitting
	Pose6D model; // The estimated model parameters

	// Initialize the samplers
	// The main sampler is used inside the local optimization
	sampler::ProsacSampler main_sampler(&points, // The data points
		estimator.sampleSize()); // The size of a minimal sample
	sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!main_sampler.isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return;
	}

	// Initializing SPRT test
	preemption::SPRTPreemptiveVerfication<utils::DefaultPnPEstimator> preemptive_verification(
		points,
		estimator,
		minimum_inlier_ratio_for_sprt_);
	
	// Initializing the fast inlier selector object
	inlier_selector::EmptyInlierSelector<utils::DefaultPnPEstimator, 
		neighborhood::FlannNeighborhoodGraph> inlier_selector(&neighborhood);

	GCRANSAC<utils::DefaultPnPEstimator,
		neighborhood::FlannNeighborhoodGraph,
		MSACScoringFunction<utils::DefaultPnPEstimator>,
		preemption::SPRTPreemptiveVerfication<utils::DefaultPnPEstimator>> gcransac;
	gcransac.setFPS(fps_); // Set the desired FPS (-1 means no limit)
	gcransac.settings.threshold = normalized_threshold; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = spatial_coherence_weight_; // The weight of the spatial coherence term
	gcransac.settings.confidence = confidence_; // The required confidence in the results
	gcransac.settings.max_iteration_number = 5000; // The maximum number of iterations
	gcransac.settings.min_iteration_number = 20; // The minimum number of iterations
	gcransac.settings.neighborhood_sphere_radius = sphere_radius_; // The radius of the neighborhood ball
	gcransac.settings.core_number = std::thread::hardware_concurrency(); // The number of parallel processes

	// Start GC-RANSAC
	gcransac.run(normalized_points, // The normalized points
		estimator,  // The estimator
		&main_sampler, // The sample used for selecting minimal samples in the main iteration
		&local_optimization_sampler, // The sampler used for selecting a minimal sample when doing the local optimization
		&neighborhood, // The neighborhood-graph
		model, // The obtained model parameters
		preemptive_verification,
		inlier_selector);

	// Get the statistics of the results
	const utils::RANSACStatistics& statistics = gcransac.getRansacStatistics();

	printf("Elapsed time = %f secs\n", statistics.processing_time);
	printf("Inlier number before = %d\n", static_cast<int>(statistics.inliers.size()));
	printf("Applied number of local optimizations = %d\n", static_cast<int>(statistics.local_optimization_number));
	printf("Applied number of graph-cuts = %d\n", static_cast<int>(statistics.graph_cut_number));
	printf("Number of iterations = %d\n\n", static_cast<int>(statistics.iteration_number));

	// Save the inliers to file
	utils::savePointsToFile(points, // The loaded data points
		inlier_image_points_path_.c_str(), // The path where the results should be saved
		&statistics.inliers); // The set of inlier found

	// The estimated rotation matrix
	Eigen::Matrix3d rotation =
		model.descriptor.leftCols<3>();
	// The estimated translation
	Eigen::Vector3d translation =
		model.descriptor.rightCols<1>();

	// The number of inliers found
	const size_t inlier_number = statistics.inliers.size();

	// Calculate the estimation error
	constexpr double radian_to_degree_multiplier = 180.0 / M_PI;
	const double angular_error = radian_to_degree_multiplier * (Eigen::Quaterniond(
		rotation).angularDistance(Eigen::Quaterniond(gt_rotation)));
	const double translation_error = (gt_translation - translation).norm();

	printf("The error in the rotation matrix = %f degrees\n", angular_error);
	printf("The error in the translation = %f cm\n", translation_error / 10.0);

	// If numerical optimization is needed, apply the Levenberg-Marquardt 
	// implementation of OpenCV.
	if (numerical_optimization_ && inlier_number >= 3)
	{
		// Copy the data into two matrices containing the image and object points. 
		// This would not be necessary, but selecting the submatrices by cv::Rect
		// leads to an error in cv::solvePnP().
		cv::Mat inlier_image_points(inlier_number, 2, CV_64F),
			inlier_object_points(inlier_number, 3, CV_64F);

		for (size_t i = 0; i < inlier_number; ++i)
		{
			const size_t& idx = statistics.inliers[i];
			inlier_image_points.at<double>(i, 0) = normalized_points.at<double>(idx, 0);
			inlier_image_points.at<double>(i, 1) = normalized_points.at<double>(idx, 1);
			inlier_object_points.at<double>(i, 0) = points.at<double>(idx, 2);
			inlier_object_points.at<double>(i, 1) = points.at<double>(idx, 3);
			inlier_object_points.at<double>(i, 2) = points.at<double>(idx, 4);
		}

		// Converting the estimated pose parameters OpenCV format
		cv::Mat cv_rotation(3, 3, CV_64F, rotation.data()), // The estimated rotation matrix converted to OpenCV format
			cv_translation(3, 1, CV_64F, translation.data()); // The estimated translation converted to OpenCV format

		// Convert the rotation matrix by the rodrigues formula
		cv::Mat cv_rodrigues(3, 1, CV_64F);
		cv::Rodrigues(cv_rotation.t(), cv_rodrigues);

		// Applying numerical optimization to the estimated pose parameters
		cv::solvePnP(inlier_object_points, // The object points
			inlier_image_points, // The image points
			cv::Mat::eye(3, 3, CV_64F), // The camera's intrinsic parameters 
			cv::Mat(), // An empty vector since the radial distortion is not known
			cv_rodrigues, // The initial rotation
			cv_translation, // The initial translation
			true, // Use the initial values
			cv::SOLVEPNP_ITERATIVE); // Apply numerical refinement

		// Convert the rotation vector back to a rotation matrix
		cv::Rodrigues(cv_rodrigues, cv_rotation);

		// Transpose the rotation matrix back
		cv_rotation = cv_rotation.t();

		// Calculate the error of the refined pose
		const double angular_error_refined = radian_to_degree_multiplier * (Eigen::Quaterniond(
			rotation).angularDistance(Eigen::Quaterniond(gt_rotation)));
		const double translation_error_refined = (gt_translation - translation).norm();

		printf("The error in the rotation matrix after numerical refinement = %f degrees\n", angular_error_refined);
		printf("The error in the translation after numerical refinement = %f cm\n", translation_error_refined / 10.0);
	}
}

void testEssentialMatrixFitting(
	const std::string& source_path_,
	const std::string& destination_path_,
	const std::string& source_intrinsics_path_,
	const std::string& destination_intrinsics_path_,
	const std::string& out_correspondence_path_,
	const std::string& in_correspondence_path_,
	const std::string& output_match_image_path_,
	const double confidence_,
	const double inlier_outlier_threshold_,
	const double spatial_coherence_weight_,
	const size_t cell_number_in_neighborhood_graph_,
	const int fps_,
	const double minimum_inlier_ratio_for_sprt_) // An assumption about the minimum inlier ratio used for the SPRT test
{
	// Read the images
	cv::Mat source_image = cv::imread(source_path_);
	cv::Mat destination_image = cv::imread(destination_path_);

	if (source_image.empty()) // Check if the source image is loaded successfully
	{
		printf("An error occured while loading image '%s'\n",
			source_path_.c_str());
		return;
	}

	if (destination_image.empty()) // Check if the destination image is loaded successfully
	{
		printf("An error occured while loading image '%s'\n",
			destination_path_.c_str());
		return;
	}

	// Detect or load point correspondences using AKAZE 
	cv::Mat points;
	utils::detectFeatures(
		in_correspondence_path_, // The path where the correspondences are read from or saved to.
		source_image, // The source image
		destination_image, // The destination image
		points); // The detected point correspondences. Each row is of format "x1 y1 x2 y2"

	// Load the intrinsic camera matrices
	Eigen::Matrix3d intrinsics_src,
		intrinsics_dst;

	if (!utils::loadMatrix<double, 3, 3>(source_intrinsics_path_,
		intrinsics_src))
	{
		printf("An error occured when loading the intrinsics camera matrix from '%s'\n",
			source_intrinsics_path_.c_str());
		return;
	}

	if (!utils::loadMatrix<double, 3, 3>(destination_intrinsics_path_,
		intrinsics_dst))
	{
		printf("An error occured when loading the intrinsics camera matrix from '%s'\n",
			destination_intrinsics_path_.c_str());
		return;
	}

	// Normalize the point coordinate by the intrinsic matrices
	cv::Mat normalized_points(points.size(), CV_64F);
	utils::normalizeCorrespondences(points,
		intrinsics_src,
		intrinsics_dst,
		normalized_points);

	// Normalize the threshold by the average of the focal lengths
	const double normalized_threshold =
		inlier_outlier_threshold_ / ((intrinsics_src(0, 0) + intrinsics_src(1, 1) +
			intrinsics_dst(0, 0) + intrinsics_dst(1, 1)) / 4.0);

	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	neighborhood::GridNeighborhoodGraph<4> neighborhood(&points,
		{ source_image.cols / static_cast<double>(cell_number_in_neighborhood_graph_),
			source_image.rows / static_cast<double>(cell_number_in_neighborhood_graph_),
			destination_image.cols / static_cast<double>(cell_number_in_neighborhood_graph_),
			destination_image.rows / static_cast<double>(cell_number_in_neighborhood_graph_) },
		cell_number_in_neighborhood_graph_);
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
	printf("Neighborhood calculation time = %f secs\n", elapsed_seconds.count());

	// Checking if the neighborhood graph is initialized successfully.
	if (!neighborhood.isInitialized())
	{
		fprintf(stderr, "The neighborhood graph is not initialized successfully.\n");
		return;
	}

	// Apply Graph-cut RANSAC
	utils::DefaultEssentialMatrixEstimator estimator(intrinsics_src,
		intrinsics_dst);
	std::vector<int> inliers;
	EssentialMatrix model;

	// Initializing SPRT test
	preemption::SPRTPreemptiveVerfication<utils::DefaultEssentialMatrixEstimator> preemptive_verification(
		points,
		estimator,
		minimum_inlier_ratio_for_sprt_);

	// Initializing the fast inlier selector object
	inlier_selector::EmptyInlierSelector<utils::DefaultEssentialMatrixEstimator, 
		neighborhood::GridNeighborhoodGraph<4>> inlier_selector(&neighborhood);

	// Initialize the samplers
	// The main sampler is used inside the local optimization
	sampler::ProgressiveNapsacSampler<4> main_sampler(&points,
		{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
							// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
		estimator.sampleSize(), // The size of a minimal sample
		{ static_cast<double>(source_image.cols), // The width of the source image
			static_cast<double>(source_image.rows), // The height of the source image
			static_cast<double>(destination_image.cols), // The width of the destination image
			static_cast<double>(destination_image.rows) });  // The height of the destination image
	sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!main_sampler.isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return;
	}

	GCRANSAC<utils::DefaultEssentialMatrixEstimator,
		neighborhood::GridNeighborhoodGraph<4>,
		MSACScoringFunction<utils::DefaultEssentialMatrixEstimator>,
		preemption::SPRTPreemptiveVerfication<utils::DefaultEssentialMatrixEstimator>> gcransac;
	gcransac.setFPS(fps_); // Set the desired FPS (-1 means no limit)
	gcransac.settings.threshold = normalized_threshold; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = spatial_coherence_weight_; // The weight of the spatial coherence term
	gcransac.settings.confidence = confidence_; // The required confidence in the results
	gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
	gcransac.settings.max_iteration_number = 5000; // The maximum number of iterations
	gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
	gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball
	gcransac.settings.core_number = std::thread::hardware_concurrency(); // The number of parallel processes

	// Start GC-RANSAC
	gcransac.run(normalized_points,
		estimator,
		&main_sampler,
		&local_optimization_sampler,
		&neighborhood,
		model,
		preemptive_verification,
		inlier_selector);

	// Get the statistics of the results
	const utils::RANSACStatistics& statistics = gcransac.getRansacStatistics();

	// Print the statistics
	printf("Elapsed time = %f secs\n", statistics.processing_time);
	printf("Inlier number = %d\n", static_cast<int>(statistics.inliers.size()));
	printf("Applied number of local optimizations = %d\n", static_cast<int>(statistics.local_optimization_number));
	printf("Applied number of graph-cuts = %d\n", static_cast<int>(statistics.graph_cut_number));
	printf("Number of iterations = %d\n\n", static_cast<int>(statistics.iteration_number));

	// Draw the inlier matches to the images
	cv::Mat match_image;
	utils::drawMatches(points,
		statistics.inliers,
		source_image,
		destination_image,
		match_image);

	printf("Saving the matched images to file '%s'.\n", output_match_image_path_.c_str());
	imwrite(output_match_image_path_, match_image); // Save the matched image to file
	printf("Saving the inlier correspondences to file '%s'.\n", out_correspondence_path_.c_str());
	utils::savePointsToFile(points, out_correspondence_path_.c_str(), &statistics.inliers); // Save the inliers to file

	printf("Press a button to continue...\n");

	// Showing the image
	utils::showImage(match_image,
		"Inlier correspondences",
		1600,
		1200,
		true);
}