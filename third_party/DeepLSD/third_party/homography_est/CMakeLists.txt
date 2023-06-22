cmake_minimum_required(VERSION 3.11)

project(homography_est)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

include_directories(${CMAKE_SOURCE_DIR})

add_compile_options(-std=c++17 -Wall -Werror -Wno-unused-variable)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(pybind11)
pybind11_add_module(homography_est python_interface.cc hest.cc Estimators.cc)
target_link_libraries(homography_est PRIVATE Eigen3::Eigen ${CERES_LIBRARIES})

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24.0)
    cmake_policy(SET CMP0135 OLD)
endif()
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

file(GLOB TEST_SOURCES "tests/*.cpp" "tests/*.cc")
# file(GLOB TEST_SOURCES "tests/test_hest.cc")
add_executable(all_tests hest.cc Estimators.cc ${TEST_SOURCES})
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( all_tests PRIVATE Eigen3::Eigen ${CERES_LIBRARIES} GTest::gtest_main)
