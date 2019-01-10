message(STATUS "Found Python Module: ${CMAKE_CURRENT_LIST_DIR}")

file(GLOB_RECURSE MODULE_FILES *.h *.cc)
file(GLOB_RECURSE SRC_FILES ../../src/*.cc)

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
    cuda_add_library(${PROJECT_NAME}_python SHARED ${MODULE_FILES} ${SRC_FILES} ${CUCXX_IMPL_FILES})
    target_link_libraries(${PROJECT_NAME}_python ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY})
else ()
    add_library(${PROJECT_NAME}_python SHARED ${MODULE_FILES} ${SRC_FILES})
endif()

# ---[ Linker(Basics)
target_link_libraries(${PROJECT_NAME}_python protobuf)

# ---[ Linker(OptionalLibs)
if (WITH_CUDNN)
    target_link_libraries(${PROJECT_NAME}_python cudnn)
endif()
if (WITH_NCCL)
    target_link_libraries(${PROJECT_NAME}_python nccl)
endif()
if (WITH_MPI)
    target_link_libraries(${PROJECT_NAME}_python mpi)
endif()

# ---[ Linker(Platforms)
if(UNIX)
    target_link_libraries(${PROJECT_NAME}_python pthread)
endif()
if(WIN32)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}_python shlwapi.lib ${PYTHON_LIBRARIES})
endif()

# ---[ Properties
set_target_properties(${PROJECT_NAME}_python PROPERTIES OUTPUT_NAME dragon)
if (APPLE)
  set_target_properties(${PROJECT_NAME}_python PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()

# ---[ Install
install(TARGETS ${PROJECT_NAME}_python DESTINATION ${PROJECT_SOURCE_DIR}/python/build)

# ---[ Protobuf
file(GLOB_RECURSE PYTHON_PROTO_FILES ${PROJECT_SOURCE_DIR}/python/dragon/*.proto)
foreach(_FILE ${PYTHON_PROTO_FILES})
    get_filename_component(_DIR ${_FILE} DIRECTORY)
    execute_process(COMMAND ${PROTOC_EXECUTABLE} -I=${_DIR} --python_out=${_DIR} ${_FILE})
endforeach()