cmake_minimum_required(VERSION 3.10)
project(imatchui)

set(OpenCV_DIR /usr/include/opencv4)
find_package(OpenCV REQUIRED)

find_package(Boost REQUIRED COMPONENTS system)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

add_executable(client client.cpp)

target_include_directories(client PRIVATE ${Boost_LIBRARIES}
                                          ${OpenCV_INCLUDE_DIRS})

target_link_libraries(client PRIVATE curl jsoncpp b64 ${OpenCV_LIBS})
