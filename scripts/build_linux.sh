#!/bin/sh

# Build Variables
SOURCE_DIR=$(cd "$(dirname "$0")/..";pwd)
BUILD_DIR=$(cd "$(dirname "$0")/..";pwd)/build
THIRD_PARTY_DIR=${SOURCE_DIR}/third_party

# Build Options
BUILD_PYTHON=ON
BUILD_RUNTIME=OFF

# Library Options
USE_CUDA=ON
USE_CUDNN=ON
USE_OPENMP=ON
USE_AVX=ON
USE_AVX2=ON
USE_FMA=ON

# Project Variables
PROTOBUF_SDK_ROOT_DIR=${THIRD_PARTY_DIR}/protobuf
PROTOBUF_PROTOC_EXECUTABLE=${PROTOBUF_SDK_ROOT_DIR}/bin/protoc
PYTHON_EXECUTABLE=""
if [ ${BUILD_PYTHON} == ON ] && [ "${PYTHON_EXECUTABLE}" == "" ]; then
  PYTHON_EXECUTABLE=$(python -c "import sys;print(sys.executable)")
fi

if [ ! -d ${BUILD_DIR} ]; then
  mkdir ${BUILD_DIR}
fi
cd ${BUILD_DIR}

cmake ${SOURCE_DIR} \
  -DBUILD_PYTHON=${BUILD_PYTHON} \
  -DBUILD_RUNTIME=${BUILD_RUNTIME} \
  -DUSE_CUDA=${USE_CUDA} \
  -DUSE_CUDNN=${USE_CUDNN} \
  -DUSE_OPENMP=${USE_OPENMP} \
  -DUSE_AVX=${USE_AVX} \
  -DUSE_AVX2=${USE_AVX2} \
  -DUSE_FMA=${USE_FMA} \
  -DTHIRD_PARTY_DIR=${THIRD_PARTY_DIR} \
  -DPROTOBUF_SDK_ROOT_DIR=${PROTOBUF_SDK_ROOT_DIR} \
  -DPROTOBUF_PROTOC_EXECUTABLE=${PROTOBUF_PROTOC_EXECUTABLE} \
  -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}

make install -j ${nproc}
cd ${SOURCE_DIR} && pip install .
