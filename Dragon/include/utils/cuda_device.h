// ------------------------------------------------------------
// Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

#ifndef DRAGON_UTILS_CUDA_DEVICE_H_
#define DRAGON_UTILS_CUDA_DEVICE_H_

#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cublas.h>
#include <curand.h>
#include <cuda.h>

#ifdef WITH_MPI_NCCL
#include <nccl.h>
#endif  // WITH_MPI_NCCL

#include "core/common.h"

namespace dragon {

static const int CUDA_NUM_THREADS = 1024;
#define MAX_GPUS 8

#define CUDA_VERSION_MIN(major, minor, patch) \
    (CUDA_VERSION >= (major * 1000 + minor * 100 + patch))

#define CUDA_VERSION_MAX(major, minor, patch) \
    (CUDA_VERSION < (major * 1000 + minor * 100 + patch))

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << "\n" << cudaGetErrorString(error); \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS); \
  } while (0)

#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS); \
  } while (0)

#ifdef WITH_MPI_NCCL
#define NCCL_CHECK(condition) \
  do { \
    ncclResult_t status = condition; \
    CHECK_EQ(status, ncclSuccess) << "\n" << ncclGetErrorString(status); \
  } while (0)
#endif  // WITH_MPI_NCCL

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < n; i += blockDim.x * gridDim.x)

inline int GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

#if CUDA_VERSION_MAX(9, 0, 0)
#define __hdiv hdiv
#endif

inline int NUM_DEVICES() {
    static int count = -1;
    if (count < 0) {
        auto err = cudaGetDeviceCount(&count);
        if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver) count = 0;
    }
    return count;
}

inline int CURRENT_DEVICE() {
    int gpu_id;
    cudaGetDevice(&gpu_id);
    return gpu_id;
}

inline int POINTER_DEVICE(const void* ptr) {
    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));
    return attr.device;
}

class DeviceGuard {
 public:
    DeviceGuard(int newDevice) : previous_(CURRENT_DEVICE()) {
        if (previous_ != newDevice) 
            CUDA_CHECK(cudaSetDevice(newDevice));
    }
    ~DeviceGuard() { CUDA_CHECK(cudaSetDevice(previous_)); }

 private:
    int previous_;
};

}    // namespace dragon

#else

#define CUDA_NOT_COMPILED \
    LOG(FATAL) << "CUDA was not compiled."

#endif // WITH_CUDA

#endif    // DRAGON_UTILS_CUDA_DEVICE_H_