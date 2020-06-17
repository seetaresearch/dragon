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
  add_subdirectory(framework)
  add_subdirectory(loss)
  add_subdirectory(training)
endif()

# ---[ Merge CUDA kernels to speed up compiling
if (USE_CUDA)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/op_kernels.cu "")
  foreach(_file ${KERNEL_CUDA_SOURCES})
    file(STRINGS ${_file} tmp NEWLINE_CONSUME)
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/op_kernels.cu ${tmp} "\n")
  endforeach()
  set(MODULE_CUDA_SOURCES
      ${MODULE_CUDA_SOURCES}
      ${CMAKE_CURRENT_BINARY_DIR}/op_kernels.cu)
endif()

# ---[ Submit to the parent scope
set(MODULE_SOURCES ${MODULE_SOURCES} PARENT_SCOPE)
set(MODULE_CUDA_SOURCES ${MODULE_CUDA_SOURCES} PARENT_SCOPE)
