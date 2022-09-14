project(dragon)
cmake_minimum_required(VERSION 3.0.2)

# ---[ Build Options
option(BUILD_PYTHON "Build Python binding library" ON)
option(BUILD_RUNTIME "Build C++ runtime library" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# ---[ Library Options
option(USE_CUDA "Use CUDA" ON)
option(USE_CUDNN "Use CUDNN" ON)
option(USE_MPS "Use MPS" OFF)
option(USE_BLAS "Use BLAS" OFF)
option(USE_OPENMP "Use OpenMP" ON)
option(USE_MPI "Use MPI" OFF)
option(USE_NCCL "Use NCCL" OFF)
option(USE_AVX "Use AVX instructions" ON)
option(USE_AVX2 "Use AVX2 instructions" ON)
option(USE_FMA "Use FMA instructions" ON)
option(USE_NATIVE_ARCH "Use all native instructions" OFF)
option(USE_SHARED_LIBS "Use shared libraries" ON)

# ---[ Project Variables
# Set the directory of third party.
if (NOT THIRD_PARTY_DIR)
  set(THIRD_PARTY_DIR ${PROJECT_SOURCE_DIR}/third_party)
endif()

# Set the CUDA target architectures.
# If not, common architectures (>= 5.0) will be used.
if (NOT CUDA_ARCH)
  set(CUDA_ARCH Common)
endif()

# Set the custom protobuf sdk if necessary.
# If not, "${THIRD_PARTY_DIR}/protobuf" will be used.
# set(PROTOBUF_SDK_ROOT_DIR <sdk_root_dir>)

# Set the protobuf compiler (i.e., protoc) if necessary.
# If not, a compiler in the sdk or environment will be used.
# set(PROTOBUF_PROTOC_EXECUTABLE <executable>)

# Set the python interpreter if necessary.
# If not, a searched interpreter will be used.
# set(PYTHON_EXECUTABLE <executable>)

# ---[ CMake Modules
include(${PROJECT_SOURCE_DIR}/cmake/MiscCheck.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/LinkLibrary.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/StripDebugInfo.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/Dependencies.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/Codegen.cmake)

# ---[ CMake Variables
set(CMAKE_BUILD_TYPE Release CACHE INTERNAL "" FORCE)
set(CMAKE_CONFIGURATION_TYPES Release CACHE INTERNAL "" FORCE)
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/../targets/native
      CACHE INTERNAL "" FORCE)
endif()
if (NOT LIBRARY_INSTALL_PREFIX)
  set(LIBRARY_INSTALL_PREFIX "")
endif()

# ---[ Subdirectories
if (BUILD_PYTHON)
  add_subdirectory(dragon/modules/python)
endif()
if (BUILD_RUNTIME)
  add_subdirectory(dragon/modules/runtime)
endif()
