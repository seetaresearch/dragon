# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
# Licensed under the BSD 2-Clause License.

# - Find the protobuf libraries
#
# The Following variables can be set and are optional:
#
#  PROTOBUF_SDK_ROOT_DIR      - The root dir of protobuf sdk
#  PROTOBUF_PROTOC_EXECUTABLE - The protoc compiler
# 
# The following function are defined:
#
#  protobuf_generate_cpp(<proto_file>...)    - Process the proto to C++ sources
#  protobuf_generate_lite(<proto_file>...)   - Process the proto to Lite sources
#  protobuf_generate_python(<proto_file>...) - Process the proto to Python sources
#

# Set the protobuf sdk root dir
# If not specified, directly set to "third_party/protobuf"
if (NOT PROTOBUF_SDK_ROOT_DIR)
  set(PROTOBUF_SDK_ROOT_DIR ${THIRD_PARTY_DIR}/protobuf)
endif()

# Set the protobuf compiler(i.e., protoc)
if (NOT PROTOBUF_PROTOC_EXECUTABLE)
  if (EXISTS "${PROTOBUF_SDK_ROOT_DIR}/bin/protoc")
    # Use the "protoc" in SDK if necessary
    # For cross compiling, the explicit executable is required
    set(PROTOBUF_PROTOC_EXECUTABLE ${PROTOBUF_SDK_ROOT_DIR}/bin/protoc)
  else()
    # Otherwise, try the global binary executable
    set(PROTOBUF_PROTOC_EXECUTABLE protoc)
  endif()
endif()

# Set the dllexport string
if (NOT PROTOBUF_DLLEXPORT_STRING)
  set(PROTOBUF_DLLEXPORT_STRING "dllexport_decl=DRAGON_API:")
endif()

# Process the proto to C++ sources
function(protobuf_generate_cpp)
cmake_parse_arguments("" "" "" "" ${ARGN})
foreach(_proto ${ARGN})
  get_filename_component(_proto_dir ${_proto} DIRECTORY)
  execute_process(
      COMMAND
      ${PROTOBUF_PROTOC_EXECUTABLE}
      -I=${_proto_dir}
      --cpp_out=${PROTOBUF_DLLEXPORT_STRING}${_proto_dir}
      ${_proto})
endforeach()
endfunction()

# Process the proto to C++ sources optimized for lite
function(protobuf_generate_lite)
cmake_parse_arguments("" "" "" "" ${ARGN})
foreach(_proto ${ARGN})
  file(APPEND ${_proto} "option optimize_for = LITE_RUNTIME;")
  get_filename_component(_proto_dir ${_proto} DIRECTORY)
  execute_process(
      COMMAND
      ${PROTOBUF_PROTOC_EXECUTABLE}
      -I=${_proto_dir}
      --cpp_out=${PROTOBUF_DLLEXPORT_STRING}${_proto_dir}
      ${_proto})
endforeach()
endfunction()

# Process the proto to Python sources
function(protobuf_generate_python)
cmake_parse_arguments("" "" "" "" ${ARGN})
foreach(_proto ${ARGN})
  get_filename_component(_proto_dir ${_proto} DIRECTORY)
  execute_process(
      COMMAND
      ${PROTOBUF_PROTOC_EXECUTABLE}
      -I=${_proto_dir}
      --python_out=${_proto_dir}
      ${_proto})
endforeach()
endfunction()

# Remove the constexpr for NVCC under Windows
function(protobuf_remove_constexpr)
cmake_parse_arguments("" "" "" "" ${ARGN})
foreach(_file ${ARGN})
  file(READ ${_file} tmp)
  string(REPLACE "PROTOBUF_CONSTEXPR" "" tmp "${tmp}")
  string(REPLACE "constexpr" "const" tmp "${tmp}")
  file(WRITE ${_file} "${tmp}")
endforeach()
endfunction()
