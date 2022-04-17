# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
# Licensed under the BSD 2-Clause License.

# - Strip debug information from a c/c++ target
#
# Following function are defined:
#
#  strip_debug_symbol(<target>) - Strip the debug symbol in the target
#
# References:
#
#  https://github.com/GerbilSoft/mcrecover/blob/master/cmake/macros/SplitDebugInformation.cmake
#

function(strip_debug_symbol _target)

# Detect the "strip" status
set(STRIP_OK 1)
if(MSVC)
  # MSVC strips information by itself.
  set(STRIP_OK 0)
elseif(APPLE)
  # OSX strips information by itself.
  set(STRIP_OK 0)
elseif(NOT CMAKE_STRIP)
  # command "strip" is missing.
  set(STRIP_OK 0)
endif()

if(STRIP_OK)
  # Handle target prefixes if not overridden.
  # NOTE: Cannot easily use the TYPE property in a generator expression...
  get_property(TARGET_TYPE TARGET ${_target} PROPERTY TYPE)
  set(PREFIX_EXPR_1 "$<$<STREQUAL:$<TARGET_PROPERTY:${_target},PREFIX>,>:${CMAKE_${TARGET_TYPE}_PREFIX}>")
  set(PREFIX_EXPR_2 "$<$<NOT:$<STREQUAL:$<TARGET_PROPERTY:${_target},PREFIX>,>>:$<TARGET_PROPERTY:${_target},PREFIX>>")
  set(PREFIX_EXPR_FULL "${PREFIX_EXPR_1}${PREFIX_EXPR_2}")

  # If a custom OUTPUT_NAME was specified, use it.
  set(OUTPUT_NAME_EXPR_1 "$<$<STREQUAL:$<TARGET_PROPERTY:${_target},OUTPUT_NAME>,>:${_target}>")
  set(OUTPUT_NAME_EXPR_2 "$<$<NOT:$<STREQUAL:$<TARGET_PROPERTY:${_target},OUTPUT_NAME>,>>:$<TARGET_PROPERTY:${_target},OUTPUT_NAME>>")
  set(OUTPUT_NAME_EXPR "${OUTPUT_NAME_EXPR_1}${OUTPUT_NAME_EXPR_2}")
  set(OUTPUT_NAME_FULL "${PREFIX_EXPR_FULL}${OUTPUT_NAME_EXPR}$<TARGET_PROPERTY:${_target},POSTFIX>")
  set(STRIP_SOURCE "$<TARGET_FILE:${_target}>")

  add_custom_command(
      TARGET ${_target} POST_BUILD
      COMMAND ${CMAKE_STRIP} -d ${STRIP_SOURCE})
endif()

endfunction()
