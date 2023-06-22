cmake_minimum_required(VERSION 3.6)
project(LineRefinement)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
if(${CERES_VERSION} VERSION_LESS "2.2.0")
    # ceres 2.2.0 changed the interface of local parameterization
    add_definitions("-DCERES_PARAMETERIZATION_ENABLED")
endif()

add_subdirectory(pybind11)
include_directories(../third_party/progressive-x/graph-cut-ransac/src/pygcransac/include
					../third_party/progressive-x/src/pyprogressivex/include
					../third_party/progressive-x/src/pyprogressivex/src)

# Add GC-RANSAC
find_library(GraphCutRANSAC_LIB
			 NAMES GraphCutRANSAC
			 PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/progressive-x/
			 PATH_SUFFIXES build)
if(NOT GraphCutRANSAC_LIB)
  message(FATAL_ERROR "GraphCutRANSAC library not found")
endif()

# Add Progressive-X
find_library(ProgressiveX_LIB
			 NAMES ProgressiveX
			 PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/progressive-x/
			 PATH_SUFFIXES build)
if(NOT ProgressiveX_LIB)
  message(FATAL_ERROR "ProgressiveX library not found")
endif()

pybind11_add_module(line_refinement line_vp_optim.cpp cost_functions.h vp_det.h)
target_link_libraries(line_refinement PUBLIC ceres stdc++fs
					  ${OpenCV_LIBS} "${GraphCutRANSAC_LIB}"
					  "${ProgressiveX_LIB}")
