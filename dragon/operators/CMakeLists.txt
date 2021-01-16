# ---[ General sources
add_subdirectory(activation)
add_subdirectory(array)
add_subdirectory(generic)
add_subdirectory(math)
add_subdirectory(normalization)
add_subdirectory(recurrent)
add_subdirectory(vision)

# ---[ Extended sources
if (NOT BUILD_RUNTIME)
  add_subdirectory(distributed)
  add_subdirectory(loss)
  add_subdirectory(metric)
  add_subdirectory(training)
endif()

# ---[ Submit to the parent scope
set(MODULE_INCLUDES ${MODULE_INCLUDES} PARENT_SCOPE)
set(MODULE_SOURCES ${MODULE_SOURCES} PARENT_SCOPE)
