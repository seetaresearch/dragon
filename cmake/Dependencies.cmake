# ---[ Portable origin rpath
if (APPLE)
  set(CMAKE_MACOSX_RPATH ON)
  set(RPATH_PORTABLE_ORIGIN "@loader_path")
else()
  set(RPATH_PORTABLE_ORIGIN $ORIGIN)
endif()

# ---[ Packages
include(${PROJECT_SOURCE_DIR}/../cmake/FindProtobuf.cmake)
if (BUILD_PYTHON)
  include(${PROJECT_SOURCE_DIR}/../cmake/FindPythonLibs.cmake)
  include(${PROJECT_SOURCE_DIR}/../cmake/FindNumPy.cmake)
endif()
if (USE_CUDA)
  find_package(CUDA REQUIRED)
  include(${PROJECT_SOURCE_DIR}/../cmake/SelectCudaArch.cmake)
  CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH ${CUDA_ARCH})
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH}")
  if (MSVC)
    # Suppress all warnings for msvc compiler
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -w")
  else()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14")
  endif()
endif()
if (USE_TENSORRT)
  if (NOT TENSORRT_SDK_ROOT_DIR)
    set(TENSORRT_SDK_ROOT_DIR ${THIRD_PARTY_DIR}/tensorrt)
  endif()
endif()

# ---[ Include directories
include_directories(${PROJECT_SOURCE_DIR}/../)
include_directories(${THIRD_PARTY_DIR}/eigen)
include_directories(${PROTOBUF_SDK_ROOT_DIR}/include)
if(APPLE)
  include_directories(/usr/local/include)
endif()
if (BUILD_PYTHON)
  include_directories(${PYTHON_INCLUDE_DIRS})
  include_directories(${NUMPY_INCLUDE_DIR})
  include_directories(${THIRD_PARTY_DIR}/pybind11/include)
endif()
if (USE_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
  include_directories(${THIRD_PARTY_DIR}/cub)
endif()
if (USE_CUDNN)
  include_directories(${THIRD_PARTY_DIR}/cudnn/include)
endif()
if (USE_MPI)
  include_directories(${THIRD_PARTY_DIR}/mpi/include)
endif()
if (USE_TENSORRT)
  include_directories(${TENSORRT_SDK_ROOT_DIR}/include)
endif()

# ---[ Library directories
list(APPEND THIRD_PARTY_LIBRARY_DIRS ${PROTOBUF_SDK_ROOT_DIR}/lib)
if (USE_MPI)
  list(APPEND THIRD_PARTY_LIBRARY_DIRS ${THIRD_PARTY_DIR}/mpi/lib)
endif()
if (USE_CUDNN)
  list(APPEND THIRD_PARTY_LIBRARY_DIRS ${THIRD_PARTY_DIR}/cudnn/lib)
  list(APPEND THIRD_PARTY_LIBRARY_DIRS ${THIRD_PARTY_DIR}/cudnn/lib64)
  list(APPEND THIRD_PARTY_LIBRARY_DIRS ${THIRD_PARTY_DIR}/cudnn/lib/x64)
endif()
if (USE_TENSORRT)
  list(APPEND THIRD_PARTY_LIBRARY_DIRS ${TENSORRT_SDK_ROOT_DIR}/lib)
endif()
if (USE_CUDA)
  list(APPEND THIRD_PARTY_LIBRARY_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  list(APPEND THIRD_PARTY_LIBRARY_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
endif()

# ---[ Defines
if (BUILD_PYTHON)
  add_definitions(-DUSE_PYTHON)
  if (${PYTHON_VERSION_MAJOR} STREQUAL "2")
    message(STATUS "Use Python2.")
  elseif (${PYTHON_VERSION_MAJOR} STREQUAL "3")
    message(STATUS "Use Python3.")
    add_definitions(-DUSE_PYTHON3)
  else()
    message("Invalid version of Python(Detected ${PYTHON_VERSION_STRING})")
    message(FATAL_ERROR "Do you set <PYTHON_EXECUTABLE> correctly?")
  endif()
endif()
if (USE_CUDA)
  add_definitions(-DUSE_CUDA)
  message(STATUS "Use CUDA.")
endif()
if (USE_CUDNN)
  add_definitions(-DUSE_CUDNN)
  message(STATUS "Use CUDNN.")
endif()
if (USE_OPENMP)
  add_definitions(-DUSE_OPENMP)
  message(STATUS "Use OpenMP.")
endif() 
if (USE_MPI)
  add_definitions(-DUSE_MPI)
  message(STATUS "Use MPI.")
endif()
if (USE_NCCL)
  add_definitions(-DUSE_NCCL)
  message(STATUS "Use NCCL.")
endif()
if (USE_TENSORRT)
  add_definitions(-DUSE_TENSORRT)
  message(STATUS "Use TensorRT.")
endif()
if (USE_NATIVE_ARCH)
  message(STATUS "Use all native instructions.")
else()
  if (USE_AVX)
    add_definitions(-DUSE_AVX)
    message(STATUS "Use AVX.")
  endif()
  if (USE_AVX2)
    add_definitions(-DUSE_AVX2)
    message(STATUS "Use AVX2.")
  endif()
  if (USE_FMA)
    add_definitions(-DUSE_FMA)
    message(STATUS "Use FMA.")
  endif()
endif()
if (USE_SHARED_LIBS)
  message(STATUS "Use shared libraries.")
endif()
