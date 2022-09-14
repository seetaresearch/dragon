# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
# Licensed under the BSD 2-Clause License.

# - Find the CUDNN libraries
#
# Following variables can be set and are optional:
#
#  CUDNN_VERSION          - version of the CUDNN
#  CUDNN_VERSION_MAJOR    - the major version number of CUDNN
#  CUDNN_VERSION_MINOR    - the minor version number of CUDNN
#  CUDNN_VERSION_PATCH    - the patch version number of CUDNN
#  CUDNN_INCLUDE_DIR      - path to the CUDNN headers
#  CUDNN_LIBRARIES_SHARED - path to the CUDNN shared library
#  CUDNN_LIBRARIES_STATIC - path to the CUDNN static library
#

# Set include directory.
if (EXISTS "${CUDA_INCLUDE_DIRS}/cudnn.h")
  set(CUDNN_INCLUDE_DIR ${CUDA_INCLUDE_DIRS})
elseif (EXISTS "${THIRD_PARTY_DIR}/cudnn/include/cudnn.h")
  set(CUDNN_INCLUDE_DIR ${THIRD_PARTY_DIR}/cudnn/include)
endif()

# Set version.
if (CUDNN_INCLUDE_DIR)
  set(_file ${CUDNN_INCLUDE_DIR}/cudnn.h)
  if (EXISTS "${CUDNN_INCLUDE_DIR}/cudnn_version.h")
    set(_file ${CUDNN_INCLUDE_DIR}/cudnn_version.h)
  endif()
  file(READ ${_file} tmp)
  string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)" _major "${tmp}")
  string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1" _major "${_major}")
  string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)" _minor "${tmp}")
  string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1" _minor "${_minor}")
  string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)" _patch "${tmp}")
  string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1" _patch "${_patch}")  
  set(CUDNN_VERSION ${_major}.${_minor}.${_patch})
  set(CUDNN_VERSION_MAJOR ${_major})
  set(CUDNN_VERSION_MINOR ${_minor})
  set(CUDNN_VERSION_PATCH ${_patch})
endif()

# Check version.
if (CUDNN_VERSION VERSION_LESS "7.0.0")
  message(FATAL_ERROR "CuDNN ${CUDNN_VERSION} is not supported. (Required: >= 7.0.0)")
else()
  get_filename_component(_dir "${CUDNN_INCLUDE_DIR}" ABSOLUTE)
  message(STATUS "Found CUDNN: ${_dir} (found version \"${CUDNN_VERSION}\")")
endif()

# Set libraries.
if (EXISTS "${CUDNN_INCLUDE_DIR}/../lib64")
  list(APPEND THIRD_PARTY_LIBRARY_DIRS ${CUDNN_INCLUDE_DIR}/../lib64)
elseif (EXISTS "${CUDNN_INCLUDE_DIR}/../lib/x64")
  list(APPEND THIRD_PARTY_LIBRARY_DIRS ${CUDNN_INCLUDE_DIR}/../lib/x64)
elseif (EXISTS "${CUDNN_INCLUDE_DIR}/../lib")
  list(APPEND THIRD_PARTY_LIBRARY_DIRS ${CUDNN_INCLUDE_DIR}/../lib)
endif()
set(CUDNN_LIBRARIES_SHARED cudnn)
set(CUDNN_LIBRARIES_STATIC cudnn_static)
if (CUDNN_VERSION VERSION_GREATER "7.6.5")
  set(CUDNN_LIBRARIES_SHARED ${CUDNN_LIBRARIES_SHARED}
                             cudnn_adv_infer cudnn_adv_train
                             cudnn_cnn_infer cudnn_cnn_train
                             cudnn_ops_infer cudnn_ops_train)
endif()
if (CUDNN_VERSION VERSION_GREATER "8.2.4")
  set(CUDNN_LIBRARIES_STATIC cudnn_adv_infer_static cudnn_adv_train_static
                             cudnn_cnn_infer_static cudnn_cnn_train_static
                             cudnn_ops_infer_static cudnn_ops_train_static)
endif()
