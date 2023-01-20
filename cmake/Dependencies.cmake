# ---[ Portable origin rpath
if (APPLE)
  set(CMAKE_MACOSX_RPATH ON)
  set(RPATH_PORTABLE_ORIGIN "@loader_path")
else()
  set(RPATH_PORTABLE_ORIGIN $ORIGIN)
endif()

# ---[ Packages
include(${PROJECT_SOURCE_DIR}/cmake/FindProtobuf.cmake)
if (BUILD_PYTHON)
  include(${PROJECT_SOURCE_DIR}/cmake/FindPythonLibs.cmake)
  include(${PROJECT_SOURCE_DIR}/cmake/FindNumPy.cmake)
endif()
if (USE_BLAS)
  include(${PROJECT_SOURCE_DIR}/cmake/FindBLAS.cmake)
endif()
if (USE_OPENMP)
  include(${PROJECT_SOURCE_DIR}/cmake/FindOpenMP.cmake)
endif()
if (USE_MPI)
  include(${PROJECT_SOURCE_DIR}/cmake/FindMPI.cmake)
endif()
if (USE_CUDA)
  include(${PROJECT_SOURCE_DIR}/cmake/FindCUDA.cmake)
endif()
if (USE_CUDNN)
  include(${PROJECT_SOURCE_DIR}/cmake/FindCUDNN.cmake)
endif()
if (USE_MPS)
  include(${PROJECT_SOURCE_DIR}/cmake/FindMPS.cmake)
endif()
if (USE_MLU)
  include(${PROJECT_SOURCE_DIR}/cmake/FindMLU.cmake)
endif()
if (USE_TENSORRT)
  if (NOT TENSORRT_SDK_ROOT_DIR)
    set(TENSORRT_SDK_ROOT_DIR ${THIRD_PARTY_DIR}/tensorrt)
  endif()
endif()

# ---[ Directories
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${THIRD_PARTY_DIR}/eigen)
include_directories(${PROTOBUF_SDK_ROOT_DIR}/include)
list(APPEND THIRD_PARTY_LIBRARY_DIRS ${PROTOBUF_SDK_ROOT_DIR}/lib)
if(APPLE)
  include_directories(/usr/local/include)
endif()
if (BUILD_PYTHON)
  include_directories(${PYTHON_INCLUDE_DIRS})
  include_directories(${NUMPY_INCLUDE_DIR})
  include_directories(${THIRD_PARTY_DIR}/pybind11/include)
endif()
if (USE_CUDA)
  include_directories(${CUDA_INCLUDE_DIR})
endif()
if (USE_CUDNN)
  include_directories(${CUDNN_INCLUDE_DIR})
endif()
if (USE_MLU)
  include_directories(${NEUWARE_INCLUDE_DIR})
endif()
if (USE_OPENMP)
  include_directories(${OPENMP_INCLUDE_DIR})
endif()
if (USE_MPI)
  include_directories(${MPI_INCLUDE_DIR})
endif()
if (USE_TENSORRT)
  include_directories(${TENSORRT_SDK_ROOT_DIR}/include)
  list(APPEND THIRD_PARTY_LIBRARY_DIRS ${TENSORRT_SDK_ROOT_DIR}/lib)
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
if (USE_MPS)
  add_definitions(-DUSE_MPS)
  message(STATUS "Use MPS.")
endif()
if (USE_MLU)
  add_definitions(-DUSE_MLU)
  message(STATUS "Use MLU.")
endif()
if (USE_BLAS)
  add_definitions(-DEIGEN_USE_BLAS)
  message(STATUS "Use BLAS.")
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
