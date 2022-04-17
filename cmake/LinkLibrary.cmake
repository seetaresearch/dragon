# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
# Licensed under the BSD 2-Clause License.

# - Link the libraries according to the project suffix hint
#
# Following function are defined:
#
#  target_link_libraries_v2(<target> <item>...]) - Link the libraries to target
#  target_get_libraries(<out_variable> <target> [<keyword>...]) - Query the libraries of target
#

################################################################################################

function(target_link_libraries_v2)

# Parse the arguments
cmake_parse_arguments("" "" "" "" ${ARGN})
list(GET ARGN 0 _target)
list(REMOVE_AT ARGN 0)

# Temporally disable the cmake default suffix
set(_cmake_default_suffix ${CMAKE_FIND_LIBRARY_SUFFIXES})
if (MSVC)
  set(${CMAKE_FIND_LIBRARY_SUFFIXES} .lib)
else()
  if (USE_SHARED_LIBS)
    if (APPLE)
      set(CMAKE_FIND_LIBRARY_SUFFIXES .dylib .so)
    else()
      set(CMAKE_FIND_LIBRARY_SUFFIXES .so)
    endif()
  else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
  endif()
endif()

foreach(_plain_name ${ARGN})
  # Filter the linker option
  string(FIND "${_plain_name}" "-Wl" _is_linker_option)
  if (${_is_linker_option} GREATER -1)
    list(APPEND _libraries ${_plain_name})
    continue()
  endif()
  # Firstly, search in the third party
  set(LIB_VAR "${_target}/lib/${_plain_name}")
  find_library(
      ${LIB_VAR} ${_plain_name}
      PATHS ${THIRD_PARTY_LIBRARY_DIRS} 
      NO_DEFAULT_PATH)
  # If not, search in the default path
  if (NOT ${LIB_VAR})
    find_library(${LIB_VAR} ${_plain_name})
  endif()
  # Finally, throw out a NOT-FOUND error
  if (NOT ${LIB_VAR})
    message(FATAL_ERROR
            "Failed to find the library for <${_plain_name}>. \
            (Suffixes required: ${CMAKE_FIND_LIBRARY_SUFFIXES})")
  endif()
  list(APPEND _libraries ${${LIB_VAR}})
endforeach()

# Call the builtin cmake link function
target_link_libraries(${_target} ${_libraries})

# Restore the cmake default suffix
set(CMAKE_FIND_LIBRARY_SUFFIXES ${_cmake_default_suffix})

endfunction()

################################################################################################

function(target_get_libraries out_variable)

# Parse the arguments
cmake_parse_arguments("" "" "" "" ${ARGN})
list(GET ARGN 0 _target)
list(REMOVE_AT ARGN 0)
get_target_property(_libraries ${_target} LINK_LIBRARIES)
set(_matches "")

foreach(_keyword ${ARGN})
  foreach(_file ${_libraries})
    if (_file MATCHES ${_keyword})
      list(APPEND _matches ${_file})
    endif()
  endforeach()
endforeach()

set(${out_variable} ${_matches} PARENT_SCOPE)

endfunction()
