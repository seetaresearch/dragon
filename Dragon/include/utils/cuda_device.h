// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
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
#endif

#ifdef WITH_MPI_NCCL
#include <nccl.h>
#endif

#include "core/common.h"

namespace dragon {

#ifdef WITH_CUDA

static const int CUDA_NUM_THREADS = 1024;
//  We do have a server with 10 GPUs :-)
#define MAX_GPUS 10

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

inline int CUDA_NUM_DEVICES() {
    static int count = -1;
    if (count < 0) {
        auto err = cudaGetDeviceCount(&count);
        if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver) count = 0;
    }
    return count;
}

inline int CUDA_CURRENT_DEVICE() {
    int gpu_id;
    cudaGetDevice(&gpu_id);
    return gpu_id;
}

inline int CUDA_POINTER_DEVICE(const void* ptr) {
    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));
    return attr.device;
}

struct CUDADeviceProps {
    CUDADeviceProps() : props(CUDA_NUM_DEVICES()) {
        for (int i = 0; i < CUDA_NUM_DEVICES(); ++i)
            CUDA_CHECK(cudaGetDeviceProperties(&props[i], i));
}
    vector<cudaDeviceProp> props;
};

inline const cudaDeviceProp& GetDeviceProperty(const int device_id) {
    static CUDADeviceProps props;
    CHECK_LT(device_id, (int)props.props.size())
        << "Invalid device id: " << device_id
        << "\nDetected " << props.props.size() << " eligible cuda devices.";
    return props.props[device_id];
}

inline bool TENSOR_CORE_AVAILABLE() {
#if CUDA_VERSION < 9000
    return false;
#else
    int device = CUDA_CURRENT_DEVICE();
    auto& prop = GetDeviceProperty(device);
    return prop.major >= 7;
#endif
}

class DeviceGuard {
 public:
    DeviceGuard(int newDevice) : previous_(CUDA_CURRENT_DEVICE()) {
        if (previous_ != newDevice) 
            CUDA_CHECK(cudaSetDevice(newDevice));
    }
    ~DeviceGuard() { CUDA_CHECK(cudaSetDevice(previous_)); }

 private:
    int previous_;
};

#define CUDA_FP16_NOT_COMPILED \
    LOG(FATAL) << "CUDA-FP16 was not compiled."

#else

#define CUDA_NOT_COMPILED \
    LOG(FATAL) << "CUDA was not compiled."

#endif // WITH_CUDA

}    // namespace dragon

#endif    // DRAGON_UTILS_CUDA_DEVICE_H_