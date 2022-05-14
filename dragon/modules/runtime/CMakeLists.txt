# ---[ Settings
message(STATUS "Build module: ${CMAKE_CURRENT_LIST_DIR}")

# ---[ Defines
add_definitions(-DBUILD_RUNTIME)
if (USE_MPI)
  remove_definitions(-DUSE_MPI)
endif()

# ---[ Sources
set(MODULE_INCLUDES "")
file(GLOB MODULE_SOURCES *.cc)
add_subdirectory(${PROJECT_SOURCE_DIR}/core core)
add_subdirectory(${PROJECT_SOURCE_DIR}/onnx onnx)
add_subdirectory(${PROJECT_SOURCE_DIR}/utils utils)

file(REMOVE_RECURSE ${PROJECT_SOURCE_DIR}/proto_lite)
file(COPY ${PROJECT_SOURCE_DIR}/proto DESTINATION ${PROJECT_SOURCE_DIR}/core)
file(RENAME ${PROJECT_SOURCE_DIR}/core/proto ${PROJECT_SOURCE_DIR}/proto_lite)
file(GLOB PROTO_FILES ${PROJECT_SOURCE_DIR}/proto_lite/*.proto)
protobuf_generate_lite(${PROTO_FILES})
add_subdirectory(${PROJECT_SOURCE_DIR}/proto_lite proto)

if (HAS_RUNTIME_CODEGEN)
  add_subdirectory(${PROJECT_SOURCE_DIR}/runtime/kernels kernels)
  add_subdirectory(${PROJECT_SOURCE_DIR}/runtime/operators operators)
else()
  add_subdirectory(${PROJECT_SOURCE_DIR}/kernels kernels)
  add_subdirectory(${PROJECT_SOURCE_DIR}/operators operators)
endif()

# ---[ Target
if (USE_CUDA)
  cuda_add_library(dragonrt ${MODULE_SOURCES} ${MODULE_CUDA_SOURCES})
else ()
  add_library(dragonrt ${MODULE_SOURCES})
endif()

# ---[ Target defines
if (MSVC)
  if (BUILD_SHARED_LIBS)
    target_compile_definitions(dragonrt PRIVATE "DRAGON_API=__declspec(dllexport)")
  else()
    target_compile_definitions(dragonrt PRIVATE "DRAGON_API=")
  endif()
else()
  target_compile_definitions(dragonrt PRIVATE "DRAGON_API=")
endif()

# ---[ Target settings
set(_install_rpath ${RPATH_PORTABLE_ORIGIN})
if (USE_SHARED_LIBS)
  set(_install_rpath ${_install_rpath} ${THIRD_PARTY_LIBRARY_DIRS})
endif()
set_target_properties(dragonrt PROPERTIES INSTALL_RPATH "${_install_rpath}")

# ---[ Linker
target_link_libraries_v2(dragonrt protobuf-lite)
if (USE_OPENMP)
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_link_libraries_v2(dragonrt omp)
  endif()
endif()
if (USE_CUDA)
  if (USE_SHARED_LIBS)
    target_link_libraries_v2(dragonrt ${CUDA_LIBRARIES_SHARED})
  else()
    target_link_libraries_v2(dragonrt ${CUDA_LIBRARIES_STATIC})
  endif()
endif()
if (USE_CUDNN)
  if (USE_SHARED_LIBS)
    target_link_libraries_v2(dragonrt ${CUDNN_LIBRARIES_SHARED})
  else()
    target_link_libraries_v2(
      dragonrt -Wl,--whole-archive
      ${CUDNN_LIBRARIES_STATIC} -Wl,--no-whole-archive)
  endif()
endif()

# ---[ Command - Strip
strip_debug_symbol(dragonrt)

# ---[ Install - Headers
set(_install_dir ${CMAKE_INSTALL_PREFIX}/include)
foreach(_file ${MODULE_INCLUDES})
  get_filename_component(_dir ${_file} DIRECTORY)
  get_filename_component(_ext ${_file} EXT)
  file(RELATIVE_PATH _dir ${PROJECT_SOURCE_DIR} ${_dir})
  file(COPY ${_file} DESTINATION ${_install_dir}/dragon/${_dir})
  if (${_ext} STREQUAL ".pb.h" AND WIN32)
    get_filename_component(_name ${_file} NAME)
    protobuf_remove_constexpr(${_install_dir}/dragon/${_dir}/${_name})
  endif()
endforeach()
file(COPY dragon_runtime.h DESTINATION ${_install_dir}/dragon)
file(COPY ${THIRD_PARTY_DIR}/cub/cub DESTINATION ${_install_dir}/cub)
file(COPY ${THIRD_PARTY_DIR}/eigen/Eigen DESTINATION ${_install_dir})
file(COPY ${THIRD_PARTY_DIR}/eigen/unsupported/Eigen DESTINATION ${_install_dir}/unsupported)
file(COPY ${THIRD_PARTY_DIR}/pybind11/include/pybind11 DESTINATION ${_install_dir})
if (EXISTS ${PROTOBUF_SDK_ROOT_DIR}/include/google)
  file(COPY ${PROTOBUF_SDK_ROOT_DIR}/include/google DESTINATION ${_install_dir})
endif()

# ---[ Install - Library
set(_install_dir ${CMAKE_INSTALL_PREFIX}/lib)
install(TARGETS dragonrt DESTINATION ${_install_dir}/${LIBRARY_INSTALL_PREFIX})
if (MSVC)
  target_get_libraries(_files dragon "protobuf-lite.lib")
  file(COPY ${_files} DESTINATION ${_install_dir})
endif()
