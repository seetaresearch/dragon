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

#ifndef DRAGON_UTILS_DEVICE_COMMON_CUB_H_
#define DRAGON_UTILS_DEVICE_COMMON_CUB_H_

#ifdef USE_CUDA

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cub/iterator/cache_modified_output_iterator.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>
#include <cub/warp/warp_reduce.cuh>

#include "dragon/utils/device/common_cuda.h"

namespace dragon {

#if defined(__CUDACC__)
template <typename T>
using BlockReduce = cub::BlockReduce<T, CUDA_THREADS>;

template <typename T, typename Reducer, int kThreadsPerWarp>
__inline__ __device__ T WarpReduce(T val) {
  for (int offset = kThreadsPerWarp / 2; offset > 0; offset /= 2) {
    val = Reducer()(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

template <typename T, typename Reducer, int kThreadsPerWarp>
__inline__ __device__ T WarpAllReduce(T val) {
  for (int offset = kThreadsPerWarp / 2; offset > 0; offset /= 2) {
    val = Reducer()(val, __shfl_xor_sync(0xffffffff, val, offset));
  }
  return val;
}

template <typename T, typename Reducer, int kThreadsPerBlock>
__inline__ __device__ T BlockAllReduce(T val) {
  typedef cub::BlockReduce<T, kThreadsPerBlock> BlockReduce;
  __shared__ T block_val;
  __shared__ typename BlockReduce::TempStorage storage;
  val = BlockReduce(storage).Reduce(val, Reducer());
  if (threadIdx.x == 0) block_val = val;
  __syncthreads();
  return block_val;
}
#endif // defined(__CUDACC__)

} // namespace dragon

#endif // USE_CUDA

#endif // DRAGON_UTILS_DEVICE_COMMON_CUB_H_
