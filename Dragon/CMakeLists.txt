# ---------------- Welcom To Use Dragon  ----------------

PROJECT(dragon)
CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)

# ---------------- Welcom To Use Dragon   ----------------

# ---------------- User Config ----------------

# set optional libraries
option(WITH_PYTHON3                "Set ON to use PYTHON3 otherwise PYTHON2"  OFF)
option(WITH_CUDA                   "Set ON to use CUDA"  ON)
option(WITH_CUDNN                  "Set ON to use CUDNN" OFF)
option(WITH_BLAS                   "Set ON to use BLAS"  OFF)
option(WITH_OMP                    "Set ON to use OpenMP"  ON)
option(WITH_SSE                    "Set ON to use SSE 4.1"  ON)
option(WITH_MPI                    "Set ON to use MPI"  OFF)
option(WITH_MPI_CUDA               "Set ON to use MPI-CUDA"  OFF)
option(WITH_MPI_NCCL               "Set ON to use MPI-NCCL"  OFF)
option(WITH_CUDA_FP16              "Set ON to use FP16"  ON)

# set your 3rdparty
set(3RDPARTY_DIR  ${PROJECT_SOURCE_DIR}/../3rdparty)

# set your python environment
set(PYTHON_INCLUDE_DIR /usr/include/python2.7)  # preferred
#set(PYTHON_INCLUDE_DIR /usr/include/python3.x)  # optional, set specific version
#set(ANACONDA_ROOT_DIR /xxx/anaconda)  # optional, preset for 2.7, 3.5, and 3.6
set(NUMPY_ROOT_DIR /xxx/numpy)  # required

# set CUDA compiling architecture
set(CUDA_ARCH     -gencode arch=compute_30,code=sm_30
                  -gencode arch=compute_35,code=sm_35
                  -gencode arch=compute_50,code=sm_50
                  -gencode arch=compute_60,code=sm_60)

# ---------------- User Config ----------------


# ---------------- Do Not Edit Following Items ----------------
#                                                    __----~~~~~~~~~~~------___
#                                   .  .   ~~//====......          __--~ ~~
#                   -.            \_|//     |||\\  ~~~~~~::::... /~
#                ___-==_       _-~o~  \/    |||  \\            _/~~-
#        __---~~~.==~||\=_    -_--~/_-~|-   |\\   \\        _/~
#    _-~~     .=~    |  \\-_    '-~7  /-   /  ||    \      /
#  .~       .~       |   \\ -_    /  /-   /   ||      \   /
# /  ____  /         |     \\ ~-_/  /|- _/   .||       \ /
# |~~    ~~|--~~~~--_ \     ~==-/   | \~--===~~        .\
#          '         ~-|      /|    |-~\~~       __--~~
#                      |-~~-_/ |    |   ~\_   _-~            /\
#                           /  \     \__   \/~                \__
#                      _--~ _/ | .-~~____--~-/                  ~~==.
#                     ((->/~   '.|||' -_|    ~~-/ ,              . _||
#                                -_     ~\      ~~---l__i__i__i--~~_/
#                                _-~-__   ~)  \--______________--~~
#                               //.-~~~-~_--~- |-------~~~~~~~~
#                                     //.-~~~--\
#
# ---------------- If You Are Not Goot At CMake ----------------


# ---[ Dependencies
if (WITH_CUDA) 
    FIND_PACKAGE(CUDA REQUIRED)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++11 support has been enabled by default.")

# ---[ Config types
set(CMAKE_BUILD_TYPE Release CACHE STRING "set build type to release")
set(CMAKE_CONFIGURATION_TYPES  Release CACHE STRING "set build type to release" FORCE)

# ---[ Includes
set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
include_directories(${INCLUDE_DIR})
include_directories(${3RDPARTY_DIR}/include)
include_directories(${3RDPARTY_DIR}/include/mpi)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${ANACONDA_ROOT_DIR}/include)
include_directories(${ANACONDA_ROOT_DIR}/include/python2.7)
include_directories(${ANACONDA_ROOT_DIR}/include/python3.5)
include_directories(${ANACONDA_ROOT_DIR}/include/python3.6)
include_directories(${NUMPY_ROOT_DIR}/core/include)
include_directories(${NUMPY_ROOT_DIR}/include)
include_directories(${NUMPY_ROOT_DIR})

# ---[ libs
set(3RDPARTY_LIBS ${3RDPARTY_DIR}/lib)
set(UINX_CUDNN_LIBS /usr/local/cuda/lib64)
link_directories(${3RDPARTY_LIBS})
link_directories(${UINX_CUDNN_LIBS})

# ---[ Install
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR} CACHE STRING "set install prefix" FORCE)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${3RDPARTY_LIBS})

# ---[ defines
if (WITH_PYTHON3) 
    ADD_DEFINITIONS(-DWITH_PYTHON3)
    message(STATUS "Use PYTHON3 [Optional]")
endif()
if (WITH_CUDA) 
    ADD_DEFINITIONS(-DWITH_CUDA)
    message(STATUS "Use CUDA [Optional]")
endif()
if (WITH_CUDNN)
    ADD_DEFINITIONS(-DWITH_CUDNN)
    message(STATUS "Use CUDNN [Optional]")
endif()
if (WITH_BLAS) 
    ADD_DEFINITIONS(-DWITH_BLAS)
    message(STATUS "Use BLAS [Optional]")
else()
    message(STATUS "Unuse BLAS [Optional]"
        "\n -- > GEMM/GEMV is disabled"
        "\n -- > prefer not to run as CPU Mode")
endif()
if (WITH_OMP)
    ADD_DEFINITIONS(-DWITH_OMP)
    message(STATUS "Use OpenMP [Optional]")
endif() 
if (WITH_SSE) 
    ADD_DEFINITIONS(-DWITH_SSE)
    message(STATUS "Use SSE [Optional]")
    if(UNIX)
         set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
    endif()
endif()
if (WITH_MPI)
    ADD_DEFINITIONS(-DWITH_MPI)
    message(STATUS "Use MPI [Optional]")
endif()
if (WITH_MPI_CUDA)
    ADD_DEFINITIONS(-DWITH_MPI_CUDA)
    message(STATUS "Use MPI-CUDA [Optional]")
endif()
if (WITH_MPI_NCCL)
    ADD_DEFINITIONS(-DWITH_MPI_NCCL)
    message(STATUS "Use MPI-NCCL [Optional]")
endif()
if (WITH_CUDA_FP16)
    ADD_DEFINITIONS(-DWITH_CUDA_FP16)
    message(STATUS "Use CUDA FP16 [Optional]")
endif()

# ---[ Flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH}")
if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /O2")
    if (WITH_OMP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
    endif()
endif()
if(UNIX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O2 -m64 -fpermissive -std=c++11")
    if (WITH_OMP)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif()
endif()

# ---[ Warnings

# ---[ execute
set (PROTOS_DIR ${PROJECT_SOURCE_DIR}/src/protos)
message(STATUS "Generate Protobuf Files")
execute_process(COMMAND protoc -I=${PROTOS_DIR} --cpp_out=${PROTOS_DIR} ${PROTOS_DIR}/caffemodel.proto)
execute_process(COMMAND protoc -I=${PROTOS_DIR} --cpp_out=${PROTOS_DIR} ${PROTOS_DIR}/dragon.proto)

# ---[ Subdirectories
add_subdirectory(modules/python)

# ---[ Utils
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/../lib)