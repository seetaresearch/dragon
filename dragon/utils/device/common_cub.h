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
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/warp/warp_reduce.cuh>

namespace dragon {

template <typename T>
using WarpReduce = cub::BlockReduce<T, CUDA_WARP_SIZE>;

template <typename T>
using BlockReduce = cub::BlockReduce<T, CUDA_THREADS>;

template <typename T, typename Reducer>
__inline__ __device__ T WarpAllReduce(T val) {
  for (int mask = CUDA_WARP_SIZE / 2; mask > 0; mask /= 2) {
    val = Reducer()(val, __shfl_xor_sync(0xffffffff, val, mask));
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

} // namespace dragon

#endif // USE_CUDA

#endif // DRAGON_UTILS_DEVICE_COMMON_CUB_H_
