# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 80 89 90)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

project(transformer_engine LANGUAGES CUDA CXX)

list(APPEND CMAKE_CUDA_FLAGS "--threads 4")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")
find_package(CUDAToolkit REQUIRED cublas nvToolsExt)
find_package(CUDNN REQUIRED cudnn)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(common)
if(NVTE_WITH_USERBUFFERS)
    message(STATUS "userbuffers support enabled")
    add_subdirectory(pytorch/csrc/userbuffers)
endif()


option(ENABLE_JAX "Enable JAX in the building workflow." OFF)
message(STATUS "JAX support: ${ENABLE_JAX}")
if(ENABLE_JAX)
  find_package(pybind11 CONFIG REQUIRED)
  add_subdirectory(jax)
endif()

option(ENABLE_TENSORFLOW "Enable TensorFlow in the building workflow." OFF)
message(STATUS "TensorFlow support: ${ENABLE_TENSORFLOW}")
if(ENABLE_TENSORFLOW)
  find_package(pybind11 CONFIG REQUIRED)
  add_subdirectory(tensorflow)
endif()
