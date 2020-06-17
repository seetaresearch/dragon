project(dragon)
cmake_minimum_required(VERSION 3.0.2)

# ------------------------------------------------- #
#                                                   #
#                Custom Configuration               #
#                                                   #
# ------------------------------------------------- #

# Set the optional buildings
option(BUILD_PYTHON      "Build Python binding library" ON)
option(BUILD_RUNTIME     "build C++ runtime library"    OFF)
option(BUILD_SHARED_LIBS "Build shared libraries"       ON)

# Set the optional libraries
option(USE_CUDA          "Use CUDA"                    ON)
option(USE_CUDNN         "Use CUDNN"                   ON)
option(USE_OPENMP        "Use OpenMP"                  ON)
option(USE_MPI           "Use MPI"                     OFF)
option(USE_NCCL          "Use NCCL"                    OFF)
option(USE_AVX           "Use AVX instructions"        ON)
option(USE_AVX2          "Use AVX2 instructions"       ON)
option(USE_FMA           "Use FMA instructions"        ON)
option(USE_NATIVE_ARCH   "Use all native instructions" OFF)
option(USE_SHARED_LIBS   "Use shared libraries"        ON)

# Set the optional directory of third party
if (NOT THIRD_PARTY_DIR)
  set(THIRD_PARTY_DIR ${PROJECT_SOURCE_DIR}/../third_party)
endif()

# Set the custom protobuf sdk if necessary
# If not, "${THIRD_PARTY_DIR}/protobuf" will be used
# set(PROTOBUF_SDK_ROOT_DIR <sdk_root_dir>)

# Set the protobuf compiler(i.e., protoc) if necessary
# If not, a compiler in the sdk or environment will be used
# set(PROTOBUF_PROTOC_EXECUTABLE <executable>)

# Set the python interpreter if necessary
# If not, a searched interpreter will be used
# set(PYTHON_EXECUTABLE <executable>)

# Set the CUDA target architectures
# If not, common architectures (>= 3.5) will be used
if (NOT CUDA_ARCH)
  set(CUDA_ARCH Common)
endif()

# ------------------------------------------------- #
#                                                   #
#                 CMake Configuration               #
#                                                   #
# ------------------------------------------------- #

# ---[ CMake modules
include(${PROJECT_SOURCE_DIR}/../cmake/MiscCheck.cmake)
include(${PROJECT_SOURCE_DIR}/../cmake/LinkLibrary.cmake)
include(${PROJECT_SOURCE_DIR}/../cmake/StripDebugInfo.cmake)
include(${PROJECT_SOURCE_DIR}/../cmake/Dependencies.cmake)

# ---[ CMake settings
set(CMAKE_BUILD_TYPE Release CACHE INTERNAL "" FORCE)
set(CMAKE_CONFIGURATION_TYPES Release CACHE INTERNAL "" FORCE)
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/../targets/native
      CACHE INTERNAL "" FORCE)
endif()
if (NOT LIBRARY_INSTALL_PREFIX)
  set(LIBRARY_INSTALL_PREFIX "")
endif()

# ---[ Command - Protobuf
file(GLOB PROTO_FILES ${PROJECT_SOURCE_DIR}/proto/*.proto)
protobuf_generate_cpp(${PROTO_FILES})

# ---[ Command - CodeGen
if (PYTHON_EXECUTABLE AND BUILD_RUNTIME)
  set(HAS_RUNTIME_CODEGEN ON)
  execute_process(
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${PROJECT_SOURCE_DIR}/../tools/codegen_runtime.py
      ${PROJECT_SOURCE_DIR} "REMOVE_GRADIENT")
else()
  set(HAS_RUNTIME_CODEGEN OFF)
endif()

# ---[ Subdirectories
if (BUILD_PYTHON)
  add_subdirectory(modules/python)
endif()
if (BUILD_RUNTIME)
  add_subdirectory(modules/runtime)
endif()
