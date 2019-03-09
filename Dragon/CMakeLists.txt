# ---------------- Welcom To Use Dragon  ----------------

project(dragon)
cmake_minimum_required(VERSION 3.0.2)

# ---------------- Welcom To Use Dragon   ----------------

# ---------------- User Config ----------------

# Set optional buildings
option(BUILD_PYTHON_API        "Set ON to build PYTHON API"    ON)
option(BUILD_CXX_API           "Set ON to build CXX API"       OFF)

# Set optional libraries
option(WITH_CUDA               "Set ON to use CUDA"            ON)
option(WITH_CUDNN              "Set ON to use CUDNN"           ON)
option(WITH_OMP                "Set ON to use OpenMP"          ON)
option(WITH_MPI                "Set ON to use MPI"             OFF)
option(WITH_NCCL               "Set ON to use NCCL"            OFF)

# Set your 3rdparty
if (NOT THIRD_PARTY_DIR)
    set(THIRD_PARTY_DIR  ${PROJECT_SOURCE_DIR}/../ThirdParty)
endif()

# Set your protobuf compiler(protc) if necessary
# if not, a default "protoc" in the environment path will be used
if (NOT PROTOC_EXECUTABLE)
   set(PROTOC_EXECUTABLE protoc)
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
include_directories(${THIRD_PARTY_DIR}/eigen)
include_directories(${THIRD_PARTY_DIR}/protobuf/include)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
if (BUILD_PYTHON_API)
    include_directories(${PYTHON_INCLUDE_DIRS})
    include_directories(${NUMPY_INCLUDE_DIR})
    include_directories(${THIRD_PARTY_DIR}/pybind11/include)
endif()
if (WITH_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
    include_directories(${THIRD_PARTY_DIR}/cub)
endif()
if (WITH_CUDNN)
    include_directories(${THIRD_PARTY_DIR}/cudnn/include)
endif()
if (WITH_MPI)
    include_directories(${THIRD_PARTY_DIR}/mpi/include)
endif()

# ---[ Lib Directories
list(APPEND THIRD_PARTY_LIBRARY_DIRS ${THIRD_PARTY_DIR}/protobuf/lib)
if (WITH_CUDA)
    list(APPEND THIRD_PARTY_LIBRARY_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib)
    list(APPEND THIRD_PARTY_LIBRARY_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    list(APPEND THIRD_PARTY_LIBRARY_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
endif()
if (WITH_CUDNN)
    list(APPEND THIRD_PARTY_LIBRARY_DIRS ${THIRD_PARTY_DIR}/cudnn/lib)
    list(APPEND THIRD_PARTY_LIBRARY_DIRS ${THIRD_PARTY_DIR}/cudnn/lib64)
    list(APPEND THIRD_PARTY_LIBRARY_DIRS ${THIRD_PARTY_DIR}/cudnn/lib/x64)
endif()
if (WITH_MPI)
    list(APPEND THIRD_PARTY_LIBRARY_DIRS ${THIRD_PARTY_DIR}/mpi/lib)
endif()
link_directories(${THIRD_PARTY_LIBRARY_DIRS})

# ---[ Install
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR} CACHE STRING "set install prefix" FORCE)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${THIRD_PARTY_LIBRARY_DIRS})

# ---[ Defines
if (BUILD_PYTHON_API)
    ADD_DEFINITIONS(-DWITH_PYTHON)
    if (${PYTHON_VERSION_MAJOR} STREQUAL "2")
        message(STATUS "Use Python2 [Optional]")
    elseif (${PYTHON_VERSION_MAJOR} STREQUAL "3")
        message(STATUS "Use Python3 [Optional]")
        add_definitions(-DWITH_PYTHON3)
    else()
        message("Invalid version of Python(Detected ${PYTHON_VERSION_STRING})")
        message(FATAL_ERROR "Do you set PYTHON_EXECUTABLE correctly?")
    endif()
endif()
if (WITH_CUDA) 
    add_definitions(-DWITH_CUDA)
    message(STATUS "Use CUDA [Optional]")
endif()
if (WITH_CUDNN)
    add_definitions(-DWITH_CUDNN)
    message(STATUS "Use CUDNN [Optional]")
endif()
if (WITH_OMP)
    ADD_DEFINITIONS(-DWITH_OMP)
    message(STATUS "Use OpenMP [Optional]")
endif() 
if (WITH_MPI)
    add_definitions(-DWITH_MPI)
    message(STATUS "Use MPI [Optional]")
endif()
if (WITH_NCCL)
    add_definitions(-DWITH_NCCL)
    message(STATUS "Use NCCL [Optional]")
endif()

# ---[ Flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH}")
if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /wd4244 /wd4800 /wd4819 /wd4996")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler \"/wd 4800 /wd 4819\"")
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

# ~ Protobuf
set(PROTO_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src/proto)
file(GLOB_RECURSE PROTO_FILES ${PROTO_SOURCE_DIR}/*.proto)
foreach(PROTO_FILE ${PROTO_FILES})
    message(STATUS "Generate Proto Files (ref: " ${PROTO_FILE} ")")
    execute_process(COMMAND ${PROTOC_EXECUTABLE} -I=${PROTO_SOURCE_DIR} --cpp_out=${PROTO_SOURCE_DIR} ${PROTO_FILE})
endforeach()

# ---[ Subdirectories
if (BUILD_PYTHON_API)
    add_subdirectory(modules/python)
endif()
if (BUILD_CXX_API)
    add_subdirectory(modules/cxx)
endif()
