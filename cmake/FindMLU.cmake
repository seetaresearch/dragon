# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
# Licensed under the BSD 2-Clause License.

# - Find the MLU libraries
#
# Following variables can be set and are optional:
#
#  CNRT_VERSION          - version of the CNRT
#  CNNL_VERSION          - version of the CNNL
#  CNCL_VERSION          - version of the CNCL
#  NEUWARE_ROOT_DIR      - path to the neuware toolkit
#  NEUWARE_INCLUDE_DIR   - path to the neuware headers
#

if (NOT NEUWARE_ROOT_DIR)
  set(NEUWARE_ROOT_DIR /usr/local/neuware)
endif()
include(${NEUWARE_ROOT_DIR}/cmake/modules/FindBANG.cmake)

# Set include directory.
set(NEUWARE_INCLUDE_DIR ${NEUWARE_ROOT_DIR}/include)

# Set libraries.
link_directories(${NEUWARE_ROOT_DIR}/lib64)
list(APPEND THIRD_PARTY_LIBRARY_DIRS ${NEUWARE_ROOT_DIR}/lib64)
set(MLU_LIBRARIES cnrt cnnl)

# Check CNRT version.
if (BANG_FOUND)
  set(_file ${NEUWARE_INCLUDE_DIR}/cnrt.h)
  file(READ ${_file} tmp)
  string(REGEX MATCH "define CNRT_MAJOR_VERSION * +([0-9]+)" _major "${tmp}")
  string(REGEX REPLACE "define CNRT_MAJOR_VERSION * +([0-9]+)" "\\1" _major "${_major}")
  string(REGEX MATCH "define CNRT_MINOR_VERSION * +([0-9]+)" _minor "${tmp}")
  string(REGEX REPLACE "define CNRT_MINOR_VERSION * +([0-9]+)" "\\1" _minor "${_minor}")
  string(REGEX MATCH "define CNRT_PATCH_VERSION * +([0-9]+)" _patch "${tmp}")
  string(REGEX REPLACE "define CNRT_PATCH_VERSION * +([0-9]+)" "\\1" _patch "${_patch}")  
  set(CNRT_VERSION ${_major}.${_minor}.${_patch})
  set(CNRT_VERSION_MAJOR ${_major})
  set(CNRT_VERSION_MINOR ${_minor})
  set(CNRT_VERSION_PATCH ${_patch})
  get_filename_component(_dir "${NEUWARE_ROOT_DIR}/include" ABSOLUTE)
  message(STATUS "Found CNRT: ${_dir} (found version \"${CNRT_VERSION}\")")
else()
  message(FATAL_ERROR "CNRT is not found.")
endif()

# Check CNNL version.
if (EXISTS "${NEUWARE_INCLUDE_DIR}/cnnl.h")
  set(_file ${NEUWARE_INCLUDE_DIR}/cnnl.h)
  file(READ ${_file} tmp)
  string(REGEX MATCH "define CNNL_MAJOR * +([0-9]+)" _major "${tmp}")
  string(REGEX REPLACE "define CNNL_MAJOR * +([0-9]+)" "\\1" _major "${_major}")
  string(REGEX MATCH "define CNNL_MINOR * +([0-9]+)" _minor "${tmp}")
  string(REGEX REPLACE "define CNNL_MINOR * +([0-9]+)" "\\1" _minor "${_minor}")
  string(REGEX MATCH "define CNNL_PATCHLEVEL * +([0-9]+)" _patch "${tmp}")
  string(REGEX REPLACE "define CNNL_PATCHLEVEL * +([0-9]+)" "\\1" _patch "${_patch}")  
  set(CNNL_VERSION ${_major}.${_minor}.${_patch})
  set(CNNL_VERSION_MAJOR ${_major})
  set(CNNL_VERSION_MINOR ${_minor})
  set(CNNL_VERSION_PATCH ${_patch})
  get_filename_component(_dir "${NEUWARE_ROOT_DIR}/include" ABSOLUTE)
  message(STATUS "Found CNNL: ${_dir} (found version \"${CNNL_VERSION}\")")
else()
  message(FATAL_ERROR "CNNL is not found.")
endif()

# Check CNCL version.
if (EXISTS "${NEUWARE_INCLUDE_DIR}/cncl.h" AND USE_MPI)
  set(_file ${NEUWARE_INCLUDE_DIR}/cncl.h)
  file(READ ${_file} tmp)
  string(REGEX MATCH "define CNCL_MAJOR_VERSION * +([0-9]+)" _major "${tmp}")
  string(REGEX REPLACE "define CNCL_MAJOR_VERSION * +([0-9]+)" "\\1" _major "${_major}")
  string(REGEX MATCH "define CNCL_MINOR_VERSION * +([0-9]+)" _minor "${tmp}")
  string(REGEX REPLACE "define CNCL_MINOR_VERSION * +([0-9]+)" "\\1" _minor "${_minor}")
  string(REGEX MATCH "define CNCL_PATCH_VERSION * +([0-9]+)" _patch "${tmp}")
  string(REGEX REPLACE "define CNCL_PATCH_VERSION * +([0-9]+)" "\\1" _patch "${_patch}")  
  set(CNCL_VERSION ${_major}.${_minor}.${_patch})
  set(CNCL_VERSION_MAJOR ${_major})
  set(CNCL_VERSION_MINOR ${_minor})
  set(CNCL_VERSION_PATCH ${_patch})
  list(APPEND MLU_LIBRARIES cncl)
  get_filename_component(_dir "${NEUWARE_ROOT_DIR}/include" ABSOLUTE)
  message(STATUS "Found CNCL: ${_dir} (found version \"${CNCL_VERSION}\")")
elseif (USE_MPI)
  message(FATAL_ERROR "CNNL is not found.")
endif()

# Set CNCC flags.
set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} ${MLU_ARCH}")
set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -fPIC -std=c++14 -pthread -O3")
