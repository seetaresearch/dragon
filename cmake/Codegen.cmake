# ---[ Protobuf
file(GLOB PROTO_FILES ${PROJECT_SOURCE_DIR}/dragon/proto/*.proto)
protobuf_generate_cpp(${PROTO_FILES})

if (BUILD_PYTHON)
  file(GLOB_RECURSE PROTO_FILES ${PROJECT_SOURCE_DIR}/dragon/python/*.proto)
  protobuf_generate_python(${PROTO_FILES})
endif()
