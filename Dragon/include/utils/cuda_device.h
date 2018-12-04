/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *      <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_UTILS_CUDA_DEVICE_H_
#define DRAGON_UTILS_CUDA_DEVICE_H_

#ifdef WITH_CUDA
#include <cuda.h>
#include <cublas.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

#ifdef WITH_MPI_NCCL
#include <nccl.h>
#endif

#include "core/common.h"

namespace dragon {

#ifdef WITH_CUDA

/*!
 * The number of cuda threads to use.
 * 
 * We set it to 1024 which would work for compute capability 2.x.
 *
 * Set it to 512 if using compute capability 1.x.
 */
const int CUDA_THREADS = 1024;

/*!
 * The maximum number of blocks to use in the default kernel call.
 *
 * We set it to 65535 which would work for compute capability 2.x,
 * where 65536 is the limit.
 */
const int CUDA_MAX_BLOCKS = 65535;

// You really need a NVIDIA DGX-2 !!! :-)
#define CUDA_MAX_DEVICES 16

#define CUDA_VERSION_MIN(major, minor, patch) \
    (CUDA_VERSION >= (major * 1000 + minor * 100 + patch))

#define CUDA_VERSION_MAX(major, minor, patch) \
    (CUDA_VERSION < (major * 1000 + minor * 100 + patch))

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) \
        << "\n" << cudaGetErrorString(error); \
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
    CHECK_EQ(status, ncclSuccess) \
        << "\n" << ncclGetErrorString(status); \
  } while (0)
#endif  // WITH_MPI_NCCL

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < n; i += blockDim.x * gridDim.x)

inline int CUDA_BLOCKS(const int N) {
    return std::max(
        std::min(
            (N + CUDA_THREADS - 1) / CUDA_THREADS,
            CUDA_MAX_BLOCKS
        ), 1);
}

#if CUDA_VERSION_MAX(9, 0, 0)
#define __hdiv hdiv
#endif

inline int CUDA_NUM_DEVICES() {
    static int count = -1;
    if (count < 0) {
        auto err = cudaGetDeviceCount(&count);
        if (err == cudaErrorNoDevice ||
            err == cudaErrorInsufficientDriver) count = 0;
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
        for (int i = 0; i < CUDA_NUM_DEVICES(); ++i)
            CUDA_CHECK(cudaGetDeviceProperties(&props[i], i));
}
    vector<cudaDeviceProp> props;
};

inline const cudaDeviceProp& GetDeviceProperty(
    const int               device_id) {
    static CUDADeviceProps props;
    CHECK_LT(device_id, (int)props.props.size())
        << "Invalid device id: " << device_id
        << "\nDetected " << props.props.size()
        << " eligible cuda devices.";
    return props.props[device_id];
}

inline bool CUDA_TRUE_FP16_AVAILABLE() {
    int device = CUDA_GET_DEVICE();
    auto& prop = GetDeviceProperty(device);
    return prop.major >= 6;
}

inline bool TENSOR_CORE_AVAILABLE() {
#if CUDA_VERSION < 9000
    return false;
#else
    int device = CUDA_GET_DEVICE();
    auto& prop = GetDeviceProperty(device);
    return prop.major >= 7;
#endif
}

class DeviceGuard {
 public:
    DeviceGuard(int new_id) : prev_id(CUDA_GET_DEVICE()) {
        if (prev_id != new_id) CUDA_CHECK(cudaSetDevice(new_id));
    }

    ~DeviceGuard() { CUDA_CHECK(cudaSetDevice(prev_id)); }

 private:
    int prev_id;
};

#else

#define CUDA_NOT_COMPILED \
    LOG(FATAL) << "CUDA was not compiled."

#endif  // WITH_CUDA

}  // namespace dragon

#endif  // DRAGON_UTILS_CUDA_DEVICE_H_