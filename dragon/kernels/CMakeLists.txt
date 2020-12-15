# ---[ General sources
add_subdirectory(activation)
add_subdirectory(array)
add_subdirectory(control_flow)
add_subdirectory(math)
add_subdirectory(normalization)
add_subdirectory(recurrent)
add_subdirectory(vision)

# ---[ Extended sources
if (NOT BUILD_RUNTIME)
  add_subdirectory(loss)
  add_subdirectory(training)
endif()

# ---[ Merge CUDA kernels to speed up compiling
if (USE_CUDA)
  set(_gen_file ${CMAKE_CURRENT_BINARY_DIR}/../codegen/op_kernels.cu)
  file(WRITE ${_gen_file} "")
  foreach(_file ${KERNEL_CUDA_SOURCES})
    file(STRINGS ${_file} tmp NEWLINE_CONSUME)
    file(APPEND ${_gen_file} ${tmp} "\n")
  endforeach()
  set(MODULE_CUDA_SOURCES ${MODULE_CUDA_SOURCES} ${_gen_file})
endif()

# ---[ Submit to the parent scope
set(MODULE_SOURCES ${MODULE_SOURCES} PARENT_SCOPE)
set(MODULE_CUDA_SOURCES ${MODULE_CUDA_SOURCES} PARENT_SCOPE)
