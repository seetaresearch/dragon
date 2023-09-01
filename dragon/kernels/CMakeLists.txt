# ---[ Sources
add_subdirectory(activation)
add_subdirectory(array)
add_subdirectory(loss)
add_subdirectory(math)
add_subdirectory(normalization)
add_subdirectory(sequence)
add_subdirectory(training)
add_subdirectory(vision)

set(MODULE_CUDA_SOURCES ${MODULE_CUDA_SOURCES} ${KERNEL_CUDA_SOURCES})
set(MODULE_MLU_SOURCES ${MODULE_MLU_SOURCES} ${KERNEL_MLU_SOURCES})

# Submit to the parent scope.
set(MODULE_SOURCES ${MODULE_SOURCES} PARENT_SCOPE)
set(MODULE_CUDA_SOURCES ${MODULE_CUDA_SOURCES} PARENT_SCOPE)
set(MODULE_MLU_SOURCES ${MODULE_MLU_SOURCES} PARENT_SCOPE)
