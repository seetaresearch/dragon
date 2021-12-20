#!/usr/bin/env sh

set -e

# Android Settings
ANDROID_ABI="arm64-v8a"
ANDROID_NATIVE_API_LEVEL=21  # Android 5.0
TOOLCHAIN_FILE=${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake

# Build Settings
BUILD_SHARED_LIB=ON

# Environment
SCRIPT_PATH=$(cd "$(dirname "$0")";pwd)
PROJECT_SOURCE_DIR="${SCRIPT_PATH}/../dragon"
THRID_PARTY_DIR="${SCRIPT_PATH}/../third_party"
BUILD_DIR="${PROJECT_SOURCE_DIR}/build/${ANDROID_ABI}"
INSTALL_DIR="${PROJECT_SOURCE_DIR}/sdk/${ANDROID_ABI}"
PROTOBUF_DIR="${THRID_PARTY_DIR}/protobuf/${ANDROID_ABI}"
PROTOC_EXECUTABLE=${THRID_PARTY_DIR}/protobuf/x86_64/bin/protoc

# Build protoc at the host architecture
if [ ! -f ${PROTOC_EXECUTABLE} ]; then
  echo "Build protoc at the host arch."
  cd ${THRID_PARTY_DIR}/protobuf/protobuf-3.9.2 && rm -rf build && mkdir build && cd build
  cmake -Dprotobuf_BUILD_TESTS=OFF                                \
        -DCMAKE_BUILD_TYPE=Release                                \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON                      \
        -DCMAKE_INSTALL_PREFIX=${THRID_PARTY_DIR}/protobuf/x86_64 \
      ../cmake
  make install -j $(nproc)
fi

# Build protobuf at the target architecture
if [ ! -d ${PROTOBUF_DIR} ]; then
  echo "Build protobuf at the target arch."
  cd ${THRID_PARTY_DIR}/protobuf/protobuf-3.9.2 && rm -rf build && mkdir build && cd build
  cmake -DANDROID_ABI="arm64-v8a"                \
        -DANDROID_CPP_FEATURES="rtti exceptions" \
        -DANDROID_LINKER_FLAGS="-landroid -llog" \
        -DCMAKE_ANDROID_STL_TYPE=c++_shared      \
        -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
        -DCMAKE_BUILD_TYPE=Release               \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON     \
        -Dprotobuf_BUILD_TESTS=OFF               \
        -Dprotobuf_BUILD_SHARED_LIBS=OFF         \
        -Dprotobuf_BUILD_PROTOC_BINARIES=OFF     \
        -DCMAKE_INSTALL_PREFIX=${PROTOBUF_DIR}   \
        ../cmake
  make install -j $(nproc)
fi

# Build dragon
rm -rf ${BUILD_DIR} && mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}
cmake -DANDROID_ABI="arm64-v8a"                              \
      -DANDROID_NATIVE_API_LEVEL=${ANDROID_NATIVE_API_LEVEL} \
      -DANDROID_CPP_FEATURES="rtti exceptions"               \
      -DANDROID_LINKER_FLAGS="-landroid -llog"               \
      -DCMAKE_ANDROID_STL_TYPE=c++_shared                    \
      -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE}               \
      -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=BOTH               \
      -DLIBRARY_INSTALL_PREFIX=${ANDROID_ABI}                \
      -DBUILD_PYTHON=OFF                                     \
      -DBUILD_RUNTIME=ON                                     \
      -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIB}                \
      -DUSE_CUDA=OFF                                         \
      -DUSE_CUDNN=OFF                                        \
      -DUSE_MPI=OFF                                          \
      -DUSE_NCCL=OFF                                         \
      -DUSE_OPENMP=OFF                                       \
      -DPROTOBUF_SDK_ROOT_DIR=${PROTOBUF_DIR}                \
      -DPROTOBUF_PROTOC_EXECUTABLE=${PROTOC_EXECUTABLE}      \
      -DPYTHON_EXECUTABLE=$(which python)                    \
      -DWITH_SHARED_LIBS=OFF                                 \ 
      ../..
make install -j $(getconf _NPROCESSORS_ONLN)
cd ../../..
