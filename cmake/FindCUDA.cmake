# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
# Licensed under the BSD 2-Clause License.

# - Find the CUDA libraries
#
# Following variables can be set and are optional:
#
#  CUDA_VERSION          - version of the CUDA
#  CUDA_VERSION_MAJOR    - the major version number of CUDA
#  CUDA_VERSION_MINOR    - the minor version number of CUDA
#  CUDA_TOOLKIT_ROOT_DIR - path to the CUDA toolkit
#  CUDA_INCLUDE_DIR      - path to the CUDA headers
#  CUDA_LIBRARIES_SHARED - path to the CUDA shared library
#  CUDA_LIBRARIES_STATIC - path to the CUDA static library
#

find_package(CUDA REQUIRED)
include(${PROJECT_SOURCE_DIR}/cmake/SelectCudaArch.cmake)

# Set NVCC flags.
CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH ${CUDA_ARCH})
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH} -Wno-deprecated-gpu-targets")
if (MSVC)
  # Suppress all warnings for msvc compiler.
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -w -std=c++14")
else()
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options -Wno-attributes")
endif()
if (CUDA_VERSION VERSION_GREATER "10.5" AND CUDA_VERSION VERSION_LESS "11.5")
  # Use custom CUB (>=1.13.0) for bfloat16 features with CUDA (<11.5)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -I ${THIRD_PARTY_DIR}/cub")
  add_definitions(-DTHRUST_IGNORE_CUB_VERSION_CHECK)
endif()

# Set include directory.
set(CUDA_INCLUDE_DIR ${CUDA_INCLUDE_DIRS})

# Set libraries.
if (EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
  list(APPEND THIRD_PARTY_LIBRARY_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
elseif (EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
  list(APPEND THIRD_PARTY_LIBRARY_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
endif()
set(CUDA_LIBRARIES_SHARED cudart cublas curand)
set(CUDA_LIBRARIES_STATIC culibos cudart_static cublas_static curand_static)
if (CUDA_VERSION VERSION_GREATER "10.0")
  set(CUDA_LIBRARIES_SHARED ${CUDA_LIBRARIES_SHARED} cublasLt)
  set(CUDA_LIBRARIES_STATIC ${CUDA_LIBRARIES_STATIC} cublasLt_static)
endif()
