cmake_minimum_required(VERSION 3.4...3.18)
project(pytlbd)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(pybind11)

###################################### OpenCV ######################################
find_package(OpenCV)
message( STATUS "    OpenCV_DIR: " ${OpenCV_DIR})
message( STATUS "    OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
message( STATUS "    OpenCV_LIBS: " ${OpenCV_LIBS})

###################################### BLAS & ARPACK ######################################
IF (${WITH_ARPACK})
    ADD_DEFINITIONS("-DWITH_ARPACK")
    FIND_PACKAGE(BLAS)
    MESSAGE(STATUS "    BLAS_FOUND: ${BLAS_FOUND}")
    MESSAGE(STATUS "    BLAS_LIBRARIES: ${BLAS_LIBRARIES}")
    INCLUDE_DIRECTORIES(/usr/include/arpack++)
    SET(ARPACK_LIBRARIES ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} arpack arpack++ superlu)
ENDIF ()


include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(src/include)
set( LIB_SOURCES
        src/LineBandDescriptor.cpp
        src/PairwiseLineMatching.cpp
        src/MultiOctaveSegmentDetector.cpp
        src/EDLineDetector.cpp
        src/MultiScaleMatching.cpp
        )
# file(GLOB_RECURSE LIB_SOURCES "src/*.cpp")
add_library(tlbd ${LIB_SOURCES})

# Create the python bindings
pybind11_add_module(pytlbd src/PYAPI.cpp)


# Add the dependency between the LBD code and the bindings
target_link_libraries(pytlbd PRIVATE tlbd ${OpenCV_LIBS} ${ARPACK_LIBRARIES})

target_compile_definitions(pytlbd PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

add_subdirectory(googletest)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

enable_testing()
add_executable(test_tlbd tests/test_tlbd.cpp)
target_link_libraries(test_tlbd PRIVATE tlbd ${OpenCV_LIBS} ${ARPACK_LIBRARIES} gtest)
add_test(test_tlbd test_tlbd)
