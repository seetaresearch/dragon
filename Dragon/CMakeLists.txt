# ---------------- Welcom To Use Dragon  ----------------

project(dragon)
cmake_minimum_required(VERSION 3.0.0)

# ---------------- Welcom To Use Dragon   ----------------

# ---------------- User Config ----------------

# Set optional buildings
option(BUILD_PYTHON_API        "Set ON to build PYTHON API"    ON)
option(BUILD_CXX_API           "Set ON to build CXX API"       OFF)

# Set optional libraries
option(WITH_CUDA               "Set ON to use CUDA"            ON)
option(WITH_CUDNN              "Set ON to use CUDNN"           ON)
option(WITH_BLAS               "Set ON to use BLAS"            ON)
option(WITH_OMP                "Set ON to use OpenMP"          ON)
option(WITH_SSE                "Set ON to use SSE 4.1"         ON)
option(WITH_MPI                "Set ON to use MPI"             OFF)
option(WITH_MPI_CUDA           "Set ON to use MPI-CUDA"        OFF)
option(WITH_MPI_NCCL           "Set ON to use MPI-NCCL"        OFF)

# Set your 3rdparty
if (NOT 3RDPARTY_DIR)
    set(3RDPARTY_DIR  ${PROJECT_SOURCE_DIR}/../3rdparty)
endif()

# Set your python "interpreter" if necessary
# if not, a default interpreter will be used
# here, provide several examples:
if (NOT PYTHON_EXECUTABLE)
    # set(PYTHON_EXECUTABLE  /usr/bin/python) # Linux && OSX, Builtin Python
    # set(PYTHON_EXECUTABLE  /X/anaconda/bin/python) # Linux && OSX, Anaconda
    # set(PYTHON_EXECUTABLE  X:/Anaconda/python) # Win, Anaconda
endif()

# Set CUDA compiling architecture
# Remove "compute_70/sm_70" if using CUDA 8.0
set(CUDA_ARCH  -gencode arch=compute_30,code=sm_30
               -gencode arch=compute_35,code=sm_35
               -gencode arch=compute_50,code=sm_50
               -gencode arch=compute_60,code=sm_60
               -gencode arch=compute_70,code=sm_70)

# Set CUDNN Library Dir if necessary (Linux/OSX Only)
# For Win, Recommend to use ``3RDPARTY_DIR/lib``
if (NOT CUDNN_LIBRARY_DIR)
    set(CUDNN_LIBRARY_DIR  /usr/local/cuda/lib64) # Linux
    # set(CUDNN_LIBRARY_DIR  /usr/local/cuda/lib) # OSX
endif()

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
if (BUILD_PYTHON_API)
    include(${PROJECT_SOURCE_DIR}/../CMake/FindPythonLibs.cmake)
    include(${PROJECT_SOURCE_DIR}/../CMake/FindNumPy.cmake)
endif()
if (WITH_CUDA) 
    find_package(CUDA REQUIRED)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++11 support has been enabled by default.")

# ---[ Config types
set(CMAKE_BUILD_TYPE Release CACHE STRING "set build type to release")
set(CMAKE_CONFIGURATION_TYPES  Release CACHE STRING "set build type to release" FORCE)

# ---[ Includes
include_directories(${3RDPARTY_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
if (BUILD_PYTHON_API)
    include_directories(${PYTHON_INCLUDE_DIRS})
    include_directories(${NUMPY_INCLUDE_DIR})
endif()
if (WITH_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()
if (WITH_MPI)
    include_directories(${3RDPARTY_DIR}/include/mpi)
endif()

# ---[ Lib Directories
set(3RDPARTY_LIBS ${3RDPARTY_DIR}/lib)
link_directories(${3RDPARTY_LIBS})
if (WITH_CUDNN)
    link_directories(${CUDNN_LIBRARY_DIR})
endif()

# ---[ Install
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR} CACHE STRING "set install prefix" FORCE)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${3RDPARTY_LIBS})

# ---[ Defines
if (BUILD_PYTHON_API)
    ADD_DEFINITIONS(-DWITH_PYTHON)
    if (${PYTHON_VERSION_MAJOR} STREQUAL "2")
        message(STATUS "Use Python2 [Optional]")
    elseif (${PYTHON_VERSION_MAJOR} STREQUAL "3")
        message(STATUS "Use Python3 [Optional]")
        ADD_DEFINITIONS(-DWITH_PYTHON3)
    else()
        message("Invalid version of Python(Detected ${PYTHON_VERSION_STRING})")
        message(FATAL_ERROR "Do you set PYTHON_EXECUTABLE correctly?")
    endif()
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

# ---[ Flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH}")
if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /wd4819 /wd4244")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler \"/wd 4819\"")
    string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "/O2" "/Ox" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    if (WITH_OMP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
    endif()
endif()
if(UNIX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s -w -fPIC -O3 -m64 -std=c++11")
    if (WITH_OMP AND (NOT APPLE))
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif()
endif()

# ---[ Warnings

# ---[ Commands
set (PROTOS_DIR ${PROJECT_SOURCE_DIR}/src/protos)
message(STATUS "Generate Protobuf Files")
execute_process(COMMAND protoc -I=${PROTOS_DIR} --cpp_out=${PROTOS_DIR} ${PROTOS_DIR}/caffemodel.proto)
execute_process(COMMAND protoc -I=${PROTOS_DIR} --cpp_out=${PROTOS_DIR} ${PROTOS_DIR}/dragon.proto)

# ---[ Subdirectories
if (BUILD_PYTHON_API)
    add_subdirectory(modules/python)
endif()
if (BUILD_CXX_API)
    add_subdirectory(modules/cxx)
endif()

# ---[ Utils
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/../lib)