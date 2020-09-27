#ifndef DRAGON_UTILS_DEVICE_COMMON_CUB_H_
#define DRAGON_UTILS_DEVICE_COMMON_CUB_H_

#ifdef USE_CUDA

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cub/iterator/counting_input_iterator.cuh>

#include "dragon/utils/device/common_cuda.h"

namespace cub {

struct SumHalf {
  inline __device__ half operator()(const half& a, const half& b) const {
#if __CUDA_ARCH__ >= 530
    return __hadd(a, b);
#else
    return __float2half(__half2float(a) + __half2float(b));
#endif
  }
};

struct MinHalf {
  inline __device__ half operator()(const half& a, const half& b) const {
#if __CUDA_ARCH__ >= 530
    return __hlt(a, b) ? a : b;
#else
    return __half2float(a) < __half2float(b) ? a : b;
#endif
  }
};

struct MaxHalf {
  inline __device__ half operator()(const half& a, const half& b) const {
#if __CUDA_ARCH__ >= 530
    return __hgt(a, b) ? a : b;
#else
    return __half2float(a) > __half2float(b) ? a : b;
#endif
  }
};

} // namespace cub

namespace dragon {

template <typename T>
using BlockReduce = cub::BlockReduce<T, CUDA_THREADS>;

} // namespace dragon

#endif // USE_CUDA

#endif // DRAGON_UTILS_DEVICE_COMMON_CUB_H_
