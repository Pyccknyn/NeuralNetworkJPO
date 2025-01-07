cmake_minimum_required(VERSION 3.10)

project(NeuralNetworkJPO LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include headers and find Eigen
include_directories(include)
find_package(Eigen3 REQUIRED)

# Collect source files
file(GLOB_RECURSE SOURCES src/*.cpp)

# Define executable
add_executable(NeuralNetworkJPO ${SOURCES})

# Include Eigen headers
target_include_directories(NeuralNetworkJPO PRIVATE ${EIGEN3_INCLUDE_DIR})

# Set compile options based on the compiler
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(NeuralNetworkJPO PRIVATE -Wall -Wextra -pedantic)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(NeuralNetworkJPO PRIVATE /W4)
endif()

# Set build type to Debug if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the build type (Debug or Release)" FORCE)
endif()

# Print configuration summary
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Using Eigen at: ${EIGEN3_INCLUDE_DIR}")
