message(STATUS "Found CXX Module: ${CMAKE_CURRENT_LIST_DIR}")

file(GLOB_RECURSE MODULE_FILES *.h *.cc)
file(GLOB_RECURSE SRC_FILES ../../src/*.cc)
list(REMOVE_ITEM SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../src/operators/misc/python_op.cc)

# ---[ Target
if (WITH_CUDA)
    file(GLOB_RECURSE CUCXX_IMPL_FILES ../../src/utils/*.cu)
    file(GLOB_RECURSE CUOPS_IMPL_FILES ../../src/kernels/*.cu ../../src/contrib/*.cu)
    file(WRITE cuda_ops_impl.cu "")
    set(CUCXX_IMPL_FILES ${CUCXX_IMPL_FILES} cuda_ops_impl.cu)
    foreach(CU_FILE ${CUOPS_IMPL_FILES})
        file(STRINGS ${CU_FILE} CU_CONTENT NEWLINE_CONSUME)
        file(APPEND cuda_ops_impl.cu ${CU_CONTENT} "\n\n")
    endforeach()
    cuda_add_library(${PROJECT_NAME}_cxx SHARED ${MODULE_FILES} ${SRC_FILES} ${CUCXX_IMPL_FILES})
    target_link_libraries(${PROJECT_NAME}_cxx  ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY})
else ()
    add_library(${PROJECT_NAME}_cxx SHARED ${MODULE_FILES} ${SRC_FILES})
endif()

# ---[ Linker(Basics)
target_link_libraries(${PROJECT_NAME}_cxx protobuf)

# ---[ Linker(OptionalLibs)
if (WITH_CUDNN)
    target_link_libraries(${PROJECT_NAME}_cxx cudnn)
endif()
if (WITH_NCCL)
    target_link_libraries(${PROJECT_NAME}_cxx nccl)
endif()
if (WITH_MPI)
    target_link_libraries(${PROJECT_NAME}_cxx mpi)
endif()

# ---[ Linker(Platforms)
if(UNIX)
    target_link_libraries(${PROJECT_NAME}_cxx pthread)
endif()
if(WIN32)
    target_link_libraries(${PROJECT_NAME}_cxx shlwapi.lib)
endif()

# ---[ Properties
set_target_properties(${PROJECT_NAME}_cxx PROPERTIES OUTPUT_NAME dragon)
set_target_properties(${PROJECT_NAME}_cxx PROPERTIES DEFINE_SYMBOL DRAGON_CXX_EXPORTS)

# ---[ Install
install(TARGETS ${PROJECT_NAME}_cxx DESTINATION ${PROJECT_SOURCE_DIR}/api/lib)
file(INSTALL dragon.h DESTINATION ${PROJECT_SOURCE_DIR}/api/include)