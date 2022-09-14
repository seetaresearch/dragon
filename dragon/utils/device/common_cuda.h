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

#ifndef DRAGON_UTILS_DEVICE_COMMON_CUDA_H_
#define DRAGON_UTILS_DEVICE_COMMON_CUDA_H_

#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <device_launch_parameters.h>
#endif

#include "dragon/core/common.h"

namespace dragon {

#ifdef USE_CUDA

/*! \brief The number of cuda threads in a warp */
constexpr int CUDA_WARP_SIZE = 32;

/*! \brief The number of cuda threads in a block */
constexpr int CUDA_THREADS = 256;

/*! \brief The maximum number of devices in a single machine */
constexpr int CUDA_MAX_DEVICES = 16;

/*! \brief The maximum number of tensor dimsensions */
constexpr int CUDA_TENSOR_MAX_DIMS = 8;

#define CUDA_VERSION_MIN(major, minor) \
  (CUDA_VERSION >= (major * 1000 + minor * 10))

#define CUDA_VERSION_MAX(major, minor) \
  (CUDA_VERSION < (major * 1000 + minor * 10))

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
  int device, sm_count, threads_per_sm;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaDeviceGetAttribute(
      &sm_count, cudaDevAttrMultiProcessorCount, device));
  CUDA_CHECK(cudaDeviceGetAttribute(
      &threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device));
  const auto num_blocks = (N + CUDA_THREADS - 1) / CUDA_THREADS;
  const auto max_blocks = sm_count * threads_per_sm / CUDA_THREADS * 32;
  return std::max(1, std::min(num_blocks, max_blocks));
}

#if CUDA_VERSION_MAX(9, 0)
#define __hdiv hdiv
#endif

inline int CUDAGetDeviceCount() {
  static int count = -1;
  if (count < 0) {
    auto err = cudaGetDeviceCount(&count);
    if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver) {
      count = 0;
    }
  }
  return count;
}

inline int CUDAGetDevice() {
  int device_id;
  CUDA_CHECK(cudaGetDevice(&device_id));
  return device_id;
}

struct CUDADeviceProps {
  CUDADeviceProps() : props(CUDAGetDeviceCount()) {
    for (int i = 0; i < props.size(); ++i) {
      CUDA_CHECK(cudaGetDeviceProperties(&props[i], i));
    }
  }
  vector<cudaDeviceProp> props;
};

inline const cudaDeviceProp& CUDAGetDeviceProp(int device_id) {
  static CUDADeviceProps props;
  CHECK_LT(device_id, int(props.props.size()))
      << "\nInvalid device id: " << device_id << "\nFound "
      << props.props.size() << " devices.";
  return props.props[device_id];
}

inline bool CUDA_TRUE_FP16_AVAILABLE() {
  int device = CUDAGetDevice();
  auto& prop = CUDAGetDeviceProp(device);
  return prop.major >= 6;
}

inline bool TENSOR_CORE_AVAILABLE() {
#if CUDA_VERSION < 9000
  return false;
#else
  int device = CUDAGetDevice();
  auto& prop = CUDAGetDeviceProp(device);
  return prop.major >= 7;
#endif
}

class CUDADeviceGuard {
 public:
  explicit CUDADeviceGuard(int new_id) {
    CUDA_CHECK(cudaGetDevice(&prev_id_));
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

#define DISPATCH_FUNC_BY_VALUE_WITH_TYPE_1(Func, T, val, ...) \
  do {                                                        \
    switch (val) {                                            \
      case 1: {                                               \
        Func<T, 1>(__VA_ARGS__);                              \
        break;                                                \
      }                                                       \
      case 2: {                                               \
        Func<T, 2>(__VA_ARGS__);                              \
        break;                                                \
      }                                                       \
      case 3: {                                               \
        Func<T, 3>(__VA_ARGS__);                              \
        break;                                                \
      }                                                       \
      case 4: {                                               \
        Func<T, 4>(__VA_ARGS__);                              \
        break;                                                \
      }                                                       \
      case 5: {                                               \
        Func<T, 5>(__VA_ARGS__);                              \
        break;                                                \
      }                                                       \
      case 6: {                                               \
        Func<T, 6>(__VA_ARGS__);                              \
        break;                                                \
      }                                                       \
      case 7: {                                               \
        Func<T, 7>(__VA_ARGS__);                              \
        break;                                                \
      }                                                       \
      case 8: {                                               \
        Func<T, 8>(__VA_ARGS__);                              \
        break;                                                \
      }                                                       \
      default: {                                              \
        break;                                                \
      }                                                       \
    }                                                         \
  } while (false)

#define DISPATCH_FUNC_BY_VALUE_WITH_TYPE_2(Func, T1, T2, val, ...) \
  do {                                                             \
    switch (val) {                                                 \
      case 1: {                                                    \
        Func<T1, T2, 1>(__VA_ARGS__);                              \
        break;                                                     \
      }                                                            \
      case 2: {                                                    \
        Func<T1, T2, 2>(__VA_ARGS__);                              \
        break;                                                     \
      }                                                            \
      case 3: {                                                    \
        Func<T1, T2, 3>(__VA_ARGS__);                              \
        break;                                                     \
      }                                                            \
      case 4: {                                                    \
        Func<T1, T2, 4>(__VA_ARGS__);                              \
        break;                                                     \
      }                                                            \
      case 5: {                                                    \
        Func<T1, T2, 5>(__VA_ARGS__);                              \
        break;                                                     \
      }                                                            \
      case 6: {                                                    \
        Func<T1, T2, 6>(__VA_ARGS__);                              \
        break;                                                     \
      }                                                            \
      case 7: {                                                    \
        Func<T1, T2, 7>(__VA_ARGS__);                              \
        break;                                                     \
      }                                                            \
      case 8: {                                                    \
        Func<T1, T2, 8>(__VA_ARGS__);                              \
        break;                                                     \
      }                                                            \
      default: {                                                   \
        break;                                                     \
      }                                                            \
    }                                                              \
  } while (false)

#define DISPATCH_FUNC_BY_VALUE_WITH_TYPE_3(Func, T1, T2, T3, val, ...) \
  do {                                                                 \
    switch (val) {                                                     \
      case 1: {                                                        \
        Func<T1, T2, T3, 1>(__VA_ARGS__);                              \
        break;                                                         \
      }                                                                \
      case 2: {                                                        \
        Func<T1, T2, T3, 2>(__VA_ARGS__);                              \
        break;                                                         \
      }                                                                \
      case 3: {                                                        \
        Func<T1, T2, T3, 3>(__VA_ARGS__);                              \
        break;                                                         \
      }                                                                \
      case 4: {                                                        \
        Func<T1, T2, T3, 4>(__VA_ARGS__);                              \
        break;                                                         \
      }                                                                \
      case 5: {                                                        \
        Func<T1, T2, T3, 5>(__VA_ARGS__);                              \
        break;                                                         \
      }                                                                \
      case 6: {                                                        \
        Func<T1, T2, T3, 6>(__VA_ARGS__);                              \
        break;                                                         \
      }                                                                \
      case 7: {                                                        \
        Func<T1, T2, T3, 7>(__VA_ARGS__);                              \
        break;                                                         \
      }                                                                \
      case 8: {                                                        \
        Func<T1, T2, T3, 8>(__VA_ARGS__);                              \
        break;                                                         \
      }                                                                \
      default: {                                                       \
        break;                                                         \
      }                                                                \
    }                                                                  \
  } while (false)

#else

#define CUDA_NOT_COMPILED LOG(FATAL) << "CUDA library is not built with."

class CUDADeviceGuard {
 public:
  explicit CUDADeviceGuard(int new_id) {
    CUDA_NOT_COMPILED;
  }
};

#endif // USE_CUDA

} // namespace dragon

#endif // DRAGON_UTILS_DEVICE_COMMON_CUDA_H_
