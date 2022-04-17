# ---[ Protobuf
file(GLOB PROTO_FILES ${PROJECT_SOURCE_DIR}/proto/*.proto)
protobuf_generate_cpp(${PROTO_FILES})

# ---[ Runtime
if (PYTHON_EXECUTABLE AND BUILD_RUNTIME)
  set(HAS_RUNTIME_CODEGEN ON)
  execute_process(
    COMMAND
    ${PYTHON_EXECUTABLE}
    ${PROJECT_SOURCE_DIR}/../tools/codegen_runtime.py
    ${PROJECT_SOURCE_DIR} "REMOVE_GRADIENT")
else()
  set(HAS_RUNTIME_CODEGEN OFF)
endif()
