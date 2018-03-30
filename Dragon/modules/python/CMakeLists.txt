message(STATUS "Found Python Module: ${CMAKE_CURRENT_LIST_DIR}")

if (NOT WITH_PYTHON)
    message(FATAL_ERROR "Set WITH_PYTHON as ON for this module.")
endif()

FILE(GLOB_RECURSE MODULE_FILES *.h *.hpp *.c *.cpp *.cu *.cc)
FILE(GLOB_RECURSE SRC_FILES ../../src/*.c ../../src/*.cpp ../../src/*.cu ../../src/*.cc)

# ---[ complier
if (WITH_CUDA) 
    CUDA_ADD_LIBRARY(${PROJECT_NAME}_python SHARED ${MODULE_FILES} ${SRC_FILES})
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}_python  ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY})
else ()
    ADD_LIBRARY(${PROJECT_NAME}_python SHARED ${MODULE_FILES} ${SRC_FILES})
endif()


# ---[ link basics
FILE(GLOB targets ${3RDPARTY_LIBS}/*.so ${3RDPARTY_LIBS}/*.lib) 
foreach(target ${targets})
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}_python  ${target})
endforeach()

# ---[ link optional libs
if (UNIX AND WITH_CUDNN)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}_python cudnn)
endif()
if (UNIX AND WITH_BLAS)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}_python openblas)
endif()
if (UNIX AND WITH_MPI_NCCL)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}_python nccl)
endif()

# ---[ link platforms
if(UNIX)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}_python protobuf pthread)
endif()
if(WIN32)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}_python  shlwapi.lib)
endif()

set_target_properties(${PROJECT_NAME}_python PROPERTIES OUTPUT_NAME dragon)

# ---[ install
install (TARGETS ${PROJECT_NAME}_python DESTINATION ${PROJECT_BINARY_DIR}/../lib)