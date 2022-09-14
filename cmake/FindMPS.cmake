# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
# Licensed under the BSD 2-Clause License.

# - Find the MPS libraries
#
# Following variables can be set and are optional:
#
#  FRAMEWORK_FOUNDATION - path to the Foundation.framework
#  FRAMEWORK_METAL      - path to the Metal.framework
#  FRAMEWORK_MPS        - path to the MetalPerformanceShaders.framework
#  FRAMEWORK_MPSGRAPH   - path to the MetalPerformanceShadersGraph.framework
#  MPS_OSX_VERSION      - osx version of the MPS library
#  MPS_LIBRARIES        - path to the MPS library
#

# Check frameworks.
set(FRAMEWORK_FOUNDATION ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/Foundation.framework)
set(FRAMEWORK_METAL ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/Metal.framework)
set(FRAMEWORK_MPS ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/MetalPerformanceShaders.framework)
set(FRAMEWORK_MPSGRAPH ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/MetalPerformanceShadersGraph.framework)

if (NOT EXISTS ${FRAMEWORK_FOUNDATION})
  message(FATAL_ERROR "Foundation is not found.")
else()
  get_filename_component(_dir "${FRAMEWORK_FOUNDATION}" ABSOLUTE)
  message(STATUS "Found Foundation: ${_dir}")
endif()
if (NOT EXISTS ${FRAMEWORK_METAL})
  message(FATAL_ERROR "Metal is not found.")
else()
  get_filename_component(_dir "${FRAMEWORK_METAL}" ABSOLUTE)
  message(STATUS "Found Metal: ${_dir}")
endif()
if (NOT EXISTS ${FRAMEWORK_MPS})
  message(FATAL_ERROR "MPS is not found.")
else()
  get_filename_component(_dir "${FRAMEWORK_MPS}" ABSOLUTE)
  message(STATUS "Found MPS: ${_dir}")
endif()
if (NOT EXISTS ${FRAMEWORK_MPSGRAPH})
  message(FATAL_ERROR "MPSGraph is not found.")
else()
  get_filename_component(_dir "${FRAMEWORK_MPSGRAPH}" ABSOLUTE)
  message(STATUS "Found MPSGraph: ${_dir}")
endif()

# Set defines.
string(REGEX MATCH "([0-9]+)\\.([0-9]+)" _version "${FRAMEWORK_MPS}")
set(MPS_OSX_VERSION_DEFINES "MPS_OSX_VERSION_MAJOR=${CMAKE_MATCH_1} "
                            "MPS_OSX_VERSION_MINOR=${CMAKE_MATCH_2}")

# Set libraries.
set(MPS_LIBRARIES "-weak_framework Foundation \
                   -weak_framework Metal \
                   -weak_framework MetalPerformanceShaders \
                   -weak_framework MetalPerformanceShadersGraph")
