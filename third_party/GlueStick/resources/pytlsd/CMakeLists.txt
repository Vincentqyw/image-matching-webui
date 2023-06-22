cmake_minimum_required(VERSION 3.4...3.18)
project(pytlsd)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -std=c++17")

add_subdirectory(pybind11)

###################################### OpenCV ######################################
find_package(OpenCV QUIET)
message( STATUS "    OpenCV_DIR: " ${OpenCV_DIR})
message( STATUS "    OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
message( STATUS "    OpenCV_LIBS: " ${OpenCV_LIBS})
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Declare the library containing the Line segment Detection (LSD) code
include_directories(src)
add_library(tlsd src/lsd.cpp)

# Create the python bindings
pybind11_add_module(pytlsd src/PYAPI.cpp)

# Add the dependency between the LSD code and the bindings
target_link_libraries(pytlsd PRIVATE tlsd ${OpenCV_LIBS})

target_compile_definitions(pytlsd PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

enable_testing()

find_package(OpenCV QUIET)
if(${OpenCV_FOUND})
    message( STATUS "    OpenCV_DIR: " ${OpenCV_DIR})
    message( STATUS "    OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
    message( STATUS "    OpenCV_LIBS: " ${OpenCV_LIBS})

    add_executable(test_tlsd tests/test_tlsd.cpp)
    target_include_directories(test_tlsd PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(test_tlsd PRIVATE tlsd ${OpenCV_LIBS})
    add_test(test_tlsd test_tlsd)
else(${OpenCV_FOUND})
    message( STATUS "OpenCV Not Found -- Tests disabled")
endif(${OpenCV_FOUND})
