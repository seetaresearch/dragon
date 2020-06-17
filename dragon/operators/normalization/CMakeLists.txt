# ---[ General sources
file(GLOB INCLUDES *.h)
file(GLOB SOURCES *.cc)
set(MODULE_INCLUDES ${MODULE_INCLUDES} ${INCLUDES})
set(MODULE_SOURCES ${MODULE_SOURCES} ${SOURCES})

if (BUILD_RUNTIME)
  # Remove distributed batch normalization
  list(REMOVE_ITEM MODULE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/batch_norm_op_sync.cc)
endif()

# ---[ Submit to the parent scope
set(MODULE_INCLUDES ${MODULE_INCLUDES} PARENT_SCOPE)
set(MODULE_SOURCES ${MODULE_SOURCES} PARENT_SCOPE)
