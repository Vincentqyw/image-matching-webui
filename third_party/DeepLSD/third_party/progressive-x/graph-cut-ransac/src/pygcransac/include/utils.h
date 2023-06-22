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

#include <algorithm>
#include <fstream>
#include <functional>
#include <sstream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Eigen>

#include "types.h"
#include "statistics.h"

namespace gcransac
{
	namespace utils
	{
		/*
			Function declaration
		*/
		void drawMatches(
			const cv::Mat &points_,
			const std::vector<size_t> &inliers_,
			const cv::Mat &image1_,
			const cv::Mat &image2_,
			cv::Mat &out_image_,
			int circle_radius_ = 4);

		void drawImagePoints(
			const cv::Mat &points_,
			const std::vector<size_t> &inliers_,
			const cv::Scalar &color_,
			cv::Mat &image_,
			int circle_radius_ = 10);

		bool savePointsToFile(
			const cv::Mat &points_,
			const char* file_,
			const std::vector<size_t> *inliers_ = NULL);

		template<size_t _Dimensions = 4, 
			size_t _StoreEveryKth = 1, 
			bool _PointNumberInHeader = true>
		bool loadPointsFromFile(
			cv::Mat &points_,
			const char* file_);

		void detectFeatures(
			std::string name_,
			cv::Mat image1_,
			cv::Mat image2_,
			cv::Mat &points_);

		void showImage(
			const cv::Mat &image_,
			std::string window_name_,
			size_t max_width_,
			size_t max_height_,
			bool wait_);

		template<typename T, size_t N, size_t M>
		bool loadMatrix(const std::string &path_,
			Eigen::Matrix<T, N, M> &matrix_);

		void normalizeCorrespondences(const cv::Mat &points_,
			const Eigen::Matrix3d &intrinsics_src_,
			const Eigen::Matrix3d &intrinsics_dst_,
			cv::Mat &normalized_points_);

		void saveStatisticsToFile(const gcransac::utils::RANSACStatistics &statistics_,
			const std::string &source_path_,
			const std::string &destination_path_,
			const std::string &filename_,
#ifdef _WIN32
			const int mode_ = std::fstream::app
#else
			const std::ios_base::openmode mode_ = std::fstream::app
#endif
		);

		/*
			Function definition
		*/
		void drawImagePoints(
			const cv::Mat &points_,
			const std::vector<size_t> &inliers_,
			const cv::Scalar &color_,
			cv::Mat &image_,
			int circle_radius_)
		{
			for (const auto &idx : inliers_)
			{
				cv::Point2d point(points_.at<double>(idx, 0),
					points_.at<double>(idx, 1));
				
				cv::circle(image_, 
					point, 
					circle_radius_, 
					color_, 
					static_cast<int>(circle_radius_));
			}
		}

		void drawMatches(
			const cv::Mat &points_,
			const std::vector<size_t> &inliers_,
			const cv::Mat &image_src_,
			const cv::Mat &image_dst_,
			cv::Mat &out_image_,
			int circle_radius_)
		{
			// Final image
			out_image_.create(image_src_.rows, // Height
				2 * image_src_.cols, // Width
				image_src_.type()); // Type

			cv::Mat roi_img_result_left =
				out_image_(cv::Rect(0, 0, image_src_.cols, image_src_.rows)); // Img1 will be on the left part
			cv::Mat roi_img_result_right =
				out_image_(cv::Rect(image_src_.cols, 0, image_dst_.cols, image_dst_.rows)); // Img2 will be on the right part, we shift the roi of img1.cols on the right

			cv::Mat roi_image_src = image_src_;
			cv::Mat roi_image_dst = image_dst_;

			roi_image_src.copyTo(roi_img_result_left); //Img1 will be on the left of imgResult
			roi_image_dst.copyTo(roi_img_result_right); //Img2 will be on the right of imgResult

			for (const auto &idx : inliers_)
			{
				cv::Point2d pt1(points_.at<double>(idx, 0),
					points_.at<double>(idx, 1));
				cv::Point2d pt2(image_dst_.cols + points_.at<double>(idx, 2),
					points_.at<double>(idx, 3));

				cv::Scalar color(255 * static_cast<double>(rand()) / RAND_MAX,
					255 * static_cast<double>(rand()) / RAND_MAX,
					255 * static_cast<double>(rand()) / RAND_MAX);

				cv::circle(out_image_, pt1, circle_radius_, color, static_cast<int>(circle_radius_ * 0.4));
				cv::circle(out_image_, pt2, circle_radius_, color, static_cast<int>(circle_radius_ * 0.4));
				cv::line(out_image_, pt1, pt2, color, 2);
			}
		}

		void detectFeatures(std::string scene_name_,
			cv::Mat image1_,
			cv::Mat image2_,
			cv::Mat &points_)
		{
			if (loadPointsFromFile(points_,
				scene_name_.c_str()))
			{
				printf("Match number: %d\n", points_.rows);
				return;
			}

			printf("Detect AKAZE features\n");
			cv::Mat descriptors1, descriptors2;
			std::vector<cv::KeyPoint> keypoints1, keypoints2;

			cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
			detector->detect(image1_, keypoints1);
			detector->compute(image1_, keypoints1, descriptors1);
			printf("Features found in the first image: %d\n", static_cast<int>(keypoints1.size()));

			detector->detect(image2_, keypoints2);
			detector->compute(image2_, keypoints2, descriptors2);
			printf("Features found in the second image: %d\n", static_cast<int>(keypoints2.size()));

			cv::BFMatcher matcher(cv::NORM_HAMMING);
			std::vector< std::vector< cv::DMatch >> matches_vector;
			matcher.knnMatch(descriptors1, descriptors2, matches_vector, 2);

			std::vector<std::tuple<double, cv::Point2d, cv::Point2d>> correspondences;
			for (auto match : matches_vector)
			{
				if (match.size() == 2 && match[0].distance < match[1].distance * 0.8)
				{
					auto& kp1 = keypoints1[match[0].queryIdx];
					auto& kp2 = keypoints2[match[0].trainIdx];
					correspondences.emplace_back(std::make_tuple<double, cv::Point2d, cv::Point2d>(match[0].distance / match[1].distance, (cv::Point2d)kp1.pt, (cv::Point2d)kp2.pt));
				}
			}

			// Sort the points for PROSAC
			std::sort(correspondences.begin(), correspondences.end(), [](const std::tuple<double, cv::Point2d, cv::Point2d>& correspondence_1_,
				const std::tuple<double, cv::Point2d, cv::Point2d>& correspondence_2_) -> bool
			{
				return std::get<0>(correspondence_1_) < std::get<0>(correspondence_2_);
			});

			points_ = cv::Mat(static_cast<int>(correspondences.size()), 4, CV_64F);
			double *points_ptr = reinterpret_cast<double*>(points_.data);

			for (auto[distance_ratio, point_1, point_2] : correspondences)
			{
				*(points_ptr++) = point_1.x;
				*(points_ptr++) = point_1.y;
				*(points_ptr++) = point_2.x;
				*(points_ptr++) = point_2.y;
			}

			savePointsToFile(points_, scene_name_.c_str());
			printf("Match number: %d\n", static_cast<int>(points_.rows));
		}

		template<size_t _Dimensions, size_t _StoreEveryKth, bool _PointNumberInHeader>
		bool loadPointsFromFile(cv::Mat &points,
			const char* file)
		{
			std::ifstream infile(file);

			if (!infile.is_open())
				return false;

			int N;
			std::string line;
			int line_idx = 0, 
				loaded_lines = 0;
			double *points_ptr = NULL;
			cv::Mat point;

			while (getline(infile, line))
			{
				// If the number of points to be loaded is stored in the file's header
				if (_PointNumberInHeader)
				{
					if (line_idx++ == 0)
					{
						N = atoi(line.c_str()) / _StoreEveryKth;
						points = cv::Mat(N, _Dimensions, CV_64F);
						points_ptr = reinterpret_cast<double*>(points.data);
						continue;
					}

					if (line_idx % _StoreEveryKth)
						continue;

					std::istringstream split(line);
					for (size_t dim = 0; dim < _Dimensions; ++dim)
						split >> *(points_ptr++);

					if (++loaded_lines >= N)
						break;
				}
				else // Otherwise
				{
					if (++loaded_lines % _StoreEveryKth)
						continue;
					
					if (point.rows != 1 || point.cols != _Dimensions)
					{
						point.create(1, _Dimensions, CV_64F);
						points_ptr = reinterpret_cast<double*>(point.data);
					}

					std::istringstream split(line);
					for (size_t dim = 0; dim < _Dimensions; ++dim)
						split >> points_ptr[dim];

					// Add the new point as the last row
					if (points.cols != _Dimensions)
						points.create(0, _Dimensions, CV_64F);
					points.push_back(point);
				}
			}

			infile.close();
			return true;
		}

		bool savePointsToFile(const cv::Mat &points,
			const char* file,
			const std::vector<size_t> *inliers)
		{
			std::ofstream outfile(file, std::ios::out);

			double *points_ptr = reinterpret_cast<double*>(points.data);
			const size_t M = points.cols;

			if (inliers == NULL)
			{
				outfile << points.rows << std::endl;
				for (auto i = 0; i < points.rows; ++i)
				{
					for (auto j = 0; j < M; ++j)
						outfile << *(points_ptr++) << " ";
					outfile << std::endl;
				}
			}
			else
			{
				outfile << inliers->size() << std::endl;
				for (size_t i = 0; i < inliers->size(); ++i)
				{
					const size_t offset = inliers->at(i) * M;
					for (auto j = 0; j < M; ++j)
						outfile << *(points_ptr + offset + j) << " ";
					outfile << std::endl;
				}
			}

			outfile.close();

			return true;
		}

		template<typename T, size_t N, size_t M>
		bool loadMatrix(const std::string &path_,
			Eigen::Matrix<T, N, M> &matrix_)
		{
			std::ifstream infile(path_);

			if (!infile.is_open())
				return false;

			size_t row = 0,
				column = 0;
			double element;

			while (infile >> element)
			{
				matrix_(row, column) = element;
				++column;
				if (column >= M)
				{
					column = 0;
					++row;
				}
			}

			infile.close();

			return row == N &&
				column == 0;
		}

		void showImage(const cv::Mat &image_,
			std::string window_name_,
			size_t max_width_,
			size_t max_height_,
			bool wait_)
		{
			// Resizing the window to fit into the screen if needed
			int window_width = image_.cols,
				window_height = image_.rows;
			if (static_cast<double>(image_.cols) / max_width_ > 1.0 &&
				static_cast<double>(image_.cols) / max_width_ >
				static_cast<double>(image_.rows) / max_height_)
			{
				window_width = max_width_;
				window_height = static_cast<int>(window_width * static_cast<double>(image_.rows) / static_cast<double>(image_.cols));
			}
			else if (static_cast<double>(image_.rows) / max_height_ > 1.0 &&
				static_cast<double>(image_.cols) / max_width_ <
				static_cast<double>(image_.rows) / max_height_)
			{
				window_height = max_height_;
				window_width = static_cast<int>(window_height * static_cast<double>(image_.cols) / static_cast<double>(image_.rows));
			}

			cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
			cv::resizeWindow(window_name_, window_width, window_height);
			cv::imshow(window_name_, image_);
			if (wait_)
				cv::waitKey(0);
		}

		void normalizeCorrespondences(const cv::Mat &points_,
			const Eigen::Matrix3d &intrinsics_src_,
			const Eigen::Matrix3d &intrinsics_dst_,
			cv::Mat &normalized_points_)
		{
			const double *points_ptr = reinterpret_cast<double *>(points_.data);
			double *normalized_points_ptr = reinterpret_cast<double *>(normalized_points_.data);
			const Eigen::Matrix3d inverse_intrinsics_src = intrinsics_src_.inverse(),
				inverse_intrinsics_dst = intrinsics_dst_.inverse();

			// Most likely, this is not the fastest solution, but it does
			// not affect the speed of Graph-cut RANSAC, so not a crucial part of
			// this example.
			double x0, y0, x1, y1;
			for (auto r = 0; r < points_.rows; ++r)
			{
				Eigen::Vector3d point_src,
					point_dst,
					normalized_point_src,
					normalized_point_dst;

				x0 = *(points_ptr++);
				y0 = *(points_ptr++);
				x1 = *(points_ptr++);
				y1 = *(points_ptr++);

				point_src << x0, y0, 1.0; // Homogeneous point in the first image
				point_dst << x1, y1, 1.0; // Homogeneous point in the second image

				// Normalized homogeneous point in the first image
				normalized_point_src =
					inverse_intrinsics_src * point_src;
				// Normalized homogeneous point in the second image
				normalized_point_dst =
					inverse_intrinsics_dst * point_dst;

				// The second four columns contain the normalized coordinates.
				*(normalized_points_ptr++) = normalized_point_src(0);
				*(normalized_points_ptr++) = normalized_point_src(1);
				*(normalized_points_ptr++) = normalized_point_dst(0);
				*(normalized_points_ptr++) = normalized_point_dst(1);
			}
		}

		void normalizeImagePoints(const cv::Mat &points_,
			const Eigen::Matrix3d &intrinsics_,
			cv::Mat &normalized_points_)
		{
			const Eigen::Matrix3d inverse_intrinsics = intrinsics_.inverse();

			// Most likely, this is not the fastest solution, but it does
			// not affect the speed of Graph-cut RANSAC, so not a crucial part of
			// this example.
			double x0, y0, x1, y1;
			for (auto r = 0; r < points_.rows; ++r)
			{
				Eigen::Vector3d point,
					normalized_point;

				x0 = points_.at<double>(r, 0);
				y0 = points_.at<double>(r, 1);

				point << x0, y0, 1.0; // Homogeneous point in the image

				// Normalized homogeneous point in the first image
				normalized_point =
					inverse_intrinsics * point;

				// The second four columns contain the normalized coordinates.
				normalized_points_.at<double>(r, 0) = normalized_point(0);
				normalized_points_.at<double>(r, 1) = normalized_point(1);
			}
		}

		void saveStatisticsToFile(
			const gcransac::utils::RANSACStatistics& statistics_,
			const std::string& source_path_,
			const std::string& destination_path_,
			const std::string& filename_,
#ifdef _WIN32
			const int mode_
#else
			const std::ios_base::openmode mode_
#endif
		)
		{
			std::ofstream file(filename_, mode_);

			if (!file.is_open())
			{
				fprintf(stderr, "A problem occured when saving the statistics to file '%s'.\n", filename_.c_str());
				return;
			}

			file << source_path_ << ";"
				<< destination_path_ << ";"
				<< statistics_.iteration_number << ";"
				<< statistics_.processing_time << ";"
				<< statistics_.inliers.size() << ";"
				<< statistics_.main_sampler_name << ";"
				<< statistics_.local_optimizer_sampler_name
				<< std::endl;
			file.close();
		}
	}
}
