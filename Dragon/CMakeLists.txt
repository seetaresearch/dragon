# ---------------- Welcom To Use Dragon  ----------------

PROJECT(dragon)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0)

# ---------------- Welcom To Use Dragon   ----------------

# ---------------- User Config ----------------

# set optional libraries
option(WITH_CUDA                   "Set to ON use CUDA"  ON)
option(WITH_CUDNN                  "Set to ON use CUDNN" OFF)
option(WITH_BLAS                   "Set to ON to use BLAS"  OFF)
option(WITH_SSE                    "Set to ON to use SSE 4.1"  ON)
option(WITH_MPI                    "Set to ON to use MPI"  OFF)
option(WITH_MPI_CUDA               "Set to ON to use MPI_CUDA_AWARE"  OFF)
option(WITH_CUDA_FP16              "Set to ON to use FP16"  ON)

# set your 3rdparty
set(3RDPARTY_DIR  ${PROJECT_SOURCE_DIR}/../3rdparty)

# set your py27
set(PYTHON_DIR /usr/include/python2.7)  # prefer
#set(ANACONDA_DIR /xxx/anaconda)  # optional
set(NUMPY_DIR /xxx/numpy)  # require

# set CUDA compiling architecture
set(CUDA_ARCH     -gencode arch=compute_20,code=sm_20
                  -gencode arch=compute_30,code=sm_30
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
include_directories(${NUMPY_DIR}/core/include)
include_directories(${ANACONDA_DIR}/include/python2.7)
include_directories(${PYTHON_DIR})
include_directories(${ANACONDA_DIR}/include)

# ---[ libs
set(3RDPARTY_LIBS ${3RDPARTY_DIR}/lib)
link_directories(${3RDPARTY_LIBS})
link_directories(/usr/local/cuda/lib64)

# ---[ Install
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR} CACHE STRING "set install prefix" FORCE)

# ---[ defines
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
	ADD_DEFINITIONS(-DWITH_CUDA_AWARE)
              message(STATUS "Use MPI-CUDA [Optional]")
endif()
if (WITH_CUDA_FP16)
	ADD_DEFINITIONS(-DWITH_CUDA_FP16)
              message(STATUS "Use CUDA FP16 [Optional]")
endif()

# ---[ Flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH}")
if(WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()
if(UNIX)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O2 -m64 -fpermissive -std=c++11")
endif()

# ---[ Warnings

# ---[ Subdirectories
add_subdirectory(modules/python)

# ---[ Utils
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/../lib)
