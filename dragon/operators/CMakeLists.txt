# ---[ Subdirectories
add_subdirectory(activation)
add_subdirectory(array)
add_subdirectory(distributed)
add_subdirectory(generic)
add_subdirectory(loss)
add_subdirectory(math)
add_subdirectory(normalization)
add_subdirectory(sequence)
add_subdirectory(training)
add_subdirectory(vision)

# Submit to the parent scope.
set(MODULE_INCLUDES ${MODULE_INCLUDES} PARENT_SCOPE)
set(MODULE_SOURCES ${MODULE_SOURCES} PARENT_SCOPE)
