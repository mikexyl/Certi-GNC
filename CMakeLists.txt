cmake_minimum_required(VERSION 3.9)
project(StiefelManifoldExample)

# --- GLOBAL SETTINGS ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# --- POLICIES ---
# CMP0167: FindBoost should not use deprecated variables
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif()

# CMP0060: Link to libraries by full path to avoid RPATH conflicts (e.g. Anaconda)
if(POLICY CMP0060)
    cmake_policy(SET CMP0060 NEW)
endif()

# --- OPTIONS ---
option(ENABLE_PROFILING "Enable code profiling using gperftools" OFF)

if(ENABLE_PROFILING)
    message(STATUS "Enabling code profiling using Google Performance Tools")
    find_library(PROFILER_LIB profiler REQUIRED)
endif()

# --- OUTPUT PATHS ---
set(LIBRARY_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/lib" CACHE PATH "Directory for libraries")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/bin" CACHE PATH "Directory for executables")

# --- DEPENDENCIES ---

# BLAS & SuiteSparse
find_package(BLAS REQUIRED)
# Prioritize system SuiteSparse to avoid conflicts with Anaconda
include_directories(SYSTEM /usr/include/suitesparse)

# Eigen3
find_package(Eigen3 3.3.7 REQUIRED)
if(EIGEN3_FOUND)
    message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")
endif()

# GTSAM. By default we let CMake find a system install (e.g. /usr/local
# after `sudo make install`). To use a non-installed GTSAM build tree, pass
#   -DGTSAM_DIR=/path/to/gtsam/build -DGTSAMCMakeTools_DIR=/path/to/gtsam/cmake

find_package(GTSAM REQUIRED)
if(GTSAM_FOUND)
    message(STATUS "Found GTSAM: ${GTSAM_INCLUDE_DIR}")
    include_directories(${GTSAM_INCLUDE_DIR})
endif()

find_package(GTSAMCMakeTools)
if(GTSAMCMakeTools_FOUND)
    message(STATUS "Found GTSAMCMakeTools: ${GTSAMCMakeTools_DIR}")
    set(GTSAM_BUILD_WITH_MARCH_NATIVE OFF CACHE BOOL "Disable -march=native" FORCE)
    include(GtsamMakeConfigFile)
    include(GtsamBuildTypes)
    include(GtsamTesting)
    set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${GTSAM_DIR}/../GTSAMCMakeTools")
endif()

# External Libraries
set(EIGEN_USE_BLAS ON)

# GFlags
find_package(gflags REQUIRED)
if(gflags_FOUND)
    message(STATUS "Found gflags: ${gflags_VERSION}")
endif()

# GLog
find_package(glog REQUIRED)
if(glog_FOUND)
    message(STATUS "Found glog: ${glog_VERSION}")
endif()

# GTest
find_package(GTest REQUIRED)
if(GTest_FOUND)
    message(STATUS "Found GTest: ${GTest_VERSION}")
endif()

# --- SUBDIRECTORIES ---
add_subdirectory(cpp)

# Optimization library
set(OPTIMIZATION_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Optimization" CACHE PATH "Optimization library path")
add_subdirectory("${OPTIMIZATION_DIR}")

# Preconditioners library (provides ILDL). The submodule also defines an
# LSChol target that fails to compile against newer SuiteSparseQR headers;
# we don't use it, so exclude it from the default build target.
set(PRECONDITIONERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Preconditioners" CACHE PATH "Preconditioners library path")
add_subdirectory("${PRECONDITIONERS_DIR}")
if(TARGET LSChol)
    set_target_properties(LSChol PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
