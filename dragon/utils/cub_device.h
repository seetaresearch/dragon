#ifndef DRAGON_UTILS_CUB_DEVICE_H_
#define DRAGON_UTILS_CUB_DEVICE_H_

#ifdef USE_CUDA

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cub/iterator/counting_input_iterator.cuh>

#include "dragon/utils/cuda_device.h"

namespace dragon {

template <typename T>
using BlockReduce = cub::BlockReduce<T, CUDA_THREADS>;

} // namespace dragon

#endif // USE_CUDA

#endif // DRAGON_UTILS_CUB_DEVICE_H_
