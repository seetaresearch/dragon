# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
# Licensed under the BSD 2-Clause License.

# - Find the OpenMP libraries
#
# Following variables can be set and are optional:
#
#  OPENMP_INCLUDE_DIR      - path to the OpenMP headers
#  OPENMP_LIBRARIES - path to the OpenMP library
#

# Set include directory.
if (EXISTS "${THIRD_PARTY_DIR}/openmp/include/omp.h")
  set(OPENMP_INCLUDE_DIR ${THIRD_PARTY_DIR}/openmp/include)
endif()

# Set libraries.
if (EXISTS "${OPENMP_INCLUDE_DIR}/../lib")
  list(APPEND THIRD_PARTY_LIBRARY_DIRS ${OPENMP_INCLUDE_DIR}/../lib)
endif()

set(OPENMP_LIBRARIES omp)
