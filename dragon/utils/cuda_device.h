/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_UTILS_CUDA_DEVICE_H_
#define DRAGON_UTILS_CUDA_DEVICE_H_

#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <device_launch_parameters.h>
#endif

#ifdef USE_NCCL
#include <nccl.h>
#endif

#include "dragon/core/common.h"

namespace dragon {

#ifdef USE_CUDA

/*! \brief The number of cuda threads to use */
const int CUDA_THREADS = 256;

/*! \brief The maximum number of blocks to use in the default kernel call */
const int CUDA_MAX_BLOCKS = 4096;

/*! \brief The maximum number of devices in a single machine */
const int CUDA_MAX_DEVICES = 16;

/*! \brief The maximum number of tensor dimsensions */
const int CUDA_TENSOR_MAX_DIMS = 8;

#define CUDA_VERSION_MIN(major, minor, patch) \
  (CUDA_VERSION >= (major * 1000 + minor * 100 + patch))

#define CUDA_VERSION_MAX(major, minor, patch) \
  (CUDA_VERSION < (major * 1000 + minor * 100 + patch))

#define CUDA_CHECK(condition)                                          \
  do {                                                                 \
    cudaError_t error = condition;                                     \
    CHECK_EQ(error, cudaSuccess) << "\n" << cudaGetErrorString(error); \
  } while (0)

#define CUBLAS_CHECK(condition)              \
  do {                                       \
    cublasStatus_t status = condition;       \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS); \
  } while (0)

#define CURAND_CHECK(condition)              \
  do {                                       \
    curandStatus_t status = condition;       \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS); \
  } while (0)

#ifdef USE_NCCL
#define NCCL_CHECK(condition)                                            \
  do {                                                                   \
    ncclResult_t status = condition;                                     \
    CHECK_EQ(status, ncclSuccess) << "\n" << ncclGetErrorString(status); \
  } while (0)
#endif // USE_NCCL

#define CUDA_TENSOR_DIMS_CHECK(num_dims)        \
  CHECK_LE(num_dims, CUDA_TENSOR_MAX_DIMS)      \
      << "Too many (> " << CUDA_TENSOR_MAX_DIMS \
      << ") dimensions to launch the cuda kernel."

#define CUDA_1D_KERNEL_LOOP(i, n)                               \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define CUDA_2D_KERNEL_LOOP1(i, n) \
  for (size_t i = blockIdx.x; i < n; i += gridDim.x)

#define CUDA_2D_KERNEL_LOOP2(j, m) \
  for (size_t j = threadIdx.x; j < m; j += blockDim.x)

inline int CUDA_BLOCKS(const int N) {
  return std::max(
      std::min((N + CUDA_THREADS - 1) / CUDA_THREADS, CUDA_MAX_BLOCKS), 1);
}

inline int CUDA_2D_BLOCKS(const int N) {
  return std::max(std::min(N, CUDA_MAX_BLOCKS), 1);
}

#if CUDA_VERSION_MAX(9, 0, 0)
#define __hdiv hdiv
#endif

inline int CUDA_NUM_DEVICES() {
  static int count = -1;
  if (count < 0) {
    auto err = cudaGetDeviceCount(&count);
    if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver) {
      count = 0;
    }
  }
  return count;
}

inline int CUDA_GET_DEVICE() {
  int device_id;
  cudaGetDevice(&device_id);
  return device_id;
}

struct CUDADeviceProps {
  CUDADeviceProps() : props(CUDA_NUM_DEVICES()) {
    for (int i = 0; i < CUDA_NUM_DEVICES(); ++i) {
      CUDA_CHECK(cudaGetDeviceProperties(&props[i], i));
    }
  }
  vector<cudaDeviceProp> props;
};

inline const cudaDeviceProp& GetCUDADeviceProp(int device_id) {
  static CUDADeviceProps props;
  CHECK_LT(device_id, (int)props.props.size())
      << "\nInvalid device id: " << device_id << "\nDetected "
      << props.props.size() << " devices.";
  return props.props[device_id];
}

inline bool CUDA_TRUE_FP16_AVAILABLE() {
  int device = CUDA_GET_DEVICE();
  auto& prop = GetCUDADeviceProp(device);
  return prop.major >= 6;
}

inline bool TENSOR_CORE_AVAILABLE() {
#if CUDA_VERSION < 9000
  return false;
#else
  int device = CUDA_GET_DEVICE();
  auto& prop = GetCUDADeviceProp(device);
  return prop.major >= 7;
#endif
}

class CUDADeviceGuard {
 public:
  explicit CUDADeviceGuard(int new_id) {
    cudaGetDevice(&prev_id_);
    if (prev_id_ != new_id) {
      CUDA_CHECK(cudaSetDevice(new_id));
    }
  }

  ~CUDADeviceGuard() {
    CUDA_CHECK(cudaSetDevice(prev_id_));
  }

 private:
  int prev_id_;
};

#else

#define CUDA_NOT_COMPILED LOG(FATAL) << "CUDA was not compiled."

class CUDADeviceGuard {
 public:
  explicit CUDADeviceGuard(int new_id) {
    CUDA_NOT_COMPILED;
  }
};

#endif // USE_CUDA

} // namespace dragon

#endif // DRAGON_UTILS_CUDA_DEVICE_H_
