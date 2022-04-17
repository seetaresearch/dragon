# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
# Licensed under the BSD 2-Clause License.

# Following variables can be set and are optional:
#
#  MPI_INCLUDE_DIR      - path to the MPI headers
#  MPI_LIBRARIES        - path to the MPI library
#  MPI_LIBRARIES_SHARED - path to the MPI shared library
#  MPI_LIBRARIES_STATIC - path to the MPI static library
#

# Set include directory.
set(MPI_INCLUDE_DIR ${THIRD_PARTY_DIR}/mpi/include)

# Set libraries.
if (EXISTS "${THIRD_PARTY_DIR}/mpi/lib")
  list(APPEND THIRD_PARTY_LIBRARY_DIRS ${THIRD_PARTY_DIR}/mpi/lib)
endif()
set(MPI_LIBRARIES z)
if (UNIX AND (NOT APPLE))
  set(MPI_LIBRARIES ${MPI_LIBRARIES} udev)
endif()
set(MPI_LIBRARIES_SHARED mpi)
set(MPI_LIBRARIES_STATIC mpi open-rte open-pal)
