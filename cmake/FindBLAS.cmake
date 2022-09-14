# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
# Licensed under the BSD 2-Clause License.

# - Find the BLAS libraries
#
# Following variables can be set and are optional:
#
#  BLAS_VENDOR       - search for the specific BLAS vendor
#  BLAS_FOUND_VENDOR - vendor implementing the BLAS interface is found
#  BLAS_LIBRARIES    - path to the BLAS library
#

if(CMAKE_Fortran_COMPILER_LOADED)
  include(CheckFortranFunctionExists)
else()
  include(CheckFunctionExists)
endif()

if(NOT $ENV{BLAS_VENDOR} STREQUAL "")
  set(BLAS_VENDOR $ENV{BLAS_VENDOR})
else()
  if(NOT BLAS_VENDOR)
    set(BLAS_VENDOR "All")
  endif()
endif()
set(BLAS_FOUND_VENDOR "")

function(CHECK_BLAS_LIBRARIES LIBRARIES _prefix _name _flags _list _deps _addlibdir _subdirs)
  # This function checks for the existence of the combination of libraries
  # given by _list.  If the combination is found, this checks whether can link
  # against that library combination using the name of a routine given by _name
  # using the linker flags given by _flags.  If the combination of libraries is
  # found and passes the link test, ${LIBRARIES} is set to the list of complete
  # library paths that have been found.  Otherwise, ${LIBRARIES} is set to FALSE.
  set(_libraries_work TRUE)
  set(_libraries)
  set(_combined_name)

  if(NOT USE_SHARED_LIBS)
    if(WIN32)
      set(CMAKE_FIND_LIBRARY_SUFFIXES .lib ${CMAKE_FIND_LIBRARY_SUFFIXES})
    else()
      set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
    endif()
  else()
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      # for ubuntu's libblas3gf and liblapack3gf packages
      set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES} .so.3gf)
    endif()
  endif()

  set(_extaddlibdir "${_addlibdir}")
  if(WIN32)
    list(APPEND _extaddlibdir ENV LIB)
  elseif(APPLE)
    list(APPEND _extaddlibdir ENV DYLD_LIBRARY_PATH)
  else()
    list(APPEND _extaddlibdir ENV LD_LIBRARY_PATH)
  endif()
  list(APPEND _extaddlibdir "${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}")

  foreach(_library ${_list})
    if(_library MATCHES "^-")
      # Respect linker flags as-is (required by MKL)
      list(APPEND _libraries "${_library}")
    else()
      string(REGEX REPLACE "[^A-Za-z0-9]" "_" _lib_var "${_library}")
      string(APPEND _combined_name "_${_lib_var}")
      if(NOT "${_deps}" STREQUAL "")
        string(APPEND _combined_name "_deps")
      endif()
      if(_libraries_work)
        find_library(${_prefix}_${_lib_var}_LIBRARY
          NAMES ${_library}
          NAMES_PER_DIR
          PATHS ${_extaddlibdir}
          PATH_SUFFIXES ${_subdirs})
        mark_as_advanced(${_prefix}_${_lib_var}_LIBRARY)
        list(APPEND _libraries ${${_prefix}_${_lib_var}_LIBRARY})
        set(_libraries_work ${${_prefix}_${_lib_var}_LIBRARY})
      endif()
    endif()
  endforeach()

  foreach(_flag ${_flags})
    string(REGEX REPLACE "[^A-Za-z0-9]" "_" _flag_var "${_flag}")
    string(APPEND _combined_name "_${_flag_var}")
  endforeach()

  if(_libraries_work AND USE_SHARED_LIBS)
    # Test this combination of libraries.
    set(CMAKE_REQUIRED_LIBRARIES ${_flags} ${_libraries} ${_deps})
    set(CMAKE_REQUIRED_QUIET ${BLAS_FIND_QUIETLY})
    if(CMAKE_Fortran_COMPILER_LOADED)
      check_fortran_function_exists("${_name}" ${_prefix}${_combined_name}_WORKS)
    else()
      check_function_exists("${_name}_" ${_prefix}${_combined_name}_WORKS)
    endif()
    set(CMAKE_REQUIRED_LIBRARIES)
    set(_libraries_work ${${_prefix}${_combined_name}_WORKS})
  endif()

  if(_libraries_work)
    if("${_list}" STREQUAL "")
      set(_libraries "${LIBRARIES}-PLACEHOLDER-FOR-EMPTY-LIBRARIES")
    else()
      list(APPEND _libraries ${_deps})
    endif()
  else()
    set(_libraries FALSE)
  endif()
  set(${LIBRARIES} "${_libraries}" PARENT_SCOPE)
endfunction()

# OpenBLAS? (http://www.openblas.net)
if(BLAS_VENDOR STREQUAL "OpenBLAS" OR BLAS_VENDOR STREQUAL "All")
  if(NOT BLAS_LIBRARIES)
    check_blas_libraries(BLAS_LIBRARIES BLAS sgemm "" "openblas" "" "" "")
    if(BLAS_LIBRARIES)
      set(BLAS_FOUND_VENDOR "OpenBLAS")
    endif()
  endif()
  if(NOT BLAS_LIBRARIES)
    check_blas_libraries(BLAS_LIBRARIES BLAS sgemm "" "openblas;pthread;m" "" "" "")
    if(BLAS_LIBRARIES)
      set(BLAS_FOUND_VENDOR "OpenBLAS")
    endif()
  endif()
  if(NOT BLAS_LIBRARIES)
    check_blas_libraries(BLAS_LIBRARIES BLAS sgemm "" "openblas;pthread;m;gomp" "" "" "")
    if(BLAS_LIBRARIES)
      set(BLAS_FOUND_VENDOR "OpenBLAS")
    endif()
  endif()
endif()

# Apple BLAS library?
if(BLAS_VENDOR STREQUAL "Apple" OR BLAS_VENDOR STREQUAL "All")
  if(NOT BLAS_LIBRARIES)
    check_blas_libraries(BLAS_LIBRARIES BLAS sgemm "" "Accelerate" "" "" "")
    if(BLAS_LIBRARIES)
      set(BLAS_FOUND_VENDOR "Apple")
    endif()
  endif()
endif()

# BLAS in acml library?
if(BLAS_VENDOR MATCHES "ACML" OR BLAS_VENDOR STREQUAL "All")
  if(NOT BLAS_LIBRARIES)
    check_blas_libraries(BLAS_LIBRARIES BLAS sgemm "" "acml;gfortran")
    if(BLAS_LIBRARIES)
      set(BLAS_FOUND_VENDOR "ACML")
    endif()
  endif()
endif()

# BLAS in the ATLAS library? (http://math-atlas.sourceforge.net/)
if(BLAS_VENDOR STREQUAL "ATLAS" OR BLAS_VENDOR STREQUAL "All")
  if(NOT BLAS_LIBRARIES)
    check_blas_libraries(BLAS_LIBRARIES BLAS sgemm "" "ptf77blas;atlas;gfortran" "" "" "")
    if(BLAS_LIBRARIES)
      set(BLAS_FOUND_VENDOR "ATLAS")
    endif()
  endif()
endif()

# Generic BLAS library?
if(BLAS_VENDOR STREQUAL "Generic" OR BLAS_VENDOR STREQUAL "All")
  if(NOT BLAS_LIBRARIES)
    check_blas_libraries(BLAS_LIBRARIES BLAS sgemm "" "blas" "" "" "")
    if(BLAS_LIBRARIES)
      set(BLAS_FOUND_VENDOR "Generic")
    endif()
  endif()
endif()

# Check libraries.
if (NOT BLAS_FOUND_VENDOR)
  message(FATAL_ERROR "Search for BLAS vendor: ${BLAS_VENDOR}. Not found.")
else()
  message(STATUS "Found BLAS: ${BLAS_LIBRARIES} (found vendor: ${BLAS_FOUND_VENDOR})")
endif()
