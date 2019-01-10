#ifndef DRAGON_UTILS_CUB_DEVICE_H_
#define DRAGON_UTILS_CUB_DEVICE_H_

#ifdef WITH_CUDA

#include <cub/block/block_reduce.cuh>

#include "utils/cuda_device.h"

namespace dragon {

template <typename T>
using BlockReduce = cub::BlockReduce<T, CUDA_THREADS>;

}

#endif  // WITH_CUDA

#endif  // DRAGON_UTILS_CUB_DEVICE_H_