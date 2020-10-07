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

namespace dragon {

template <typename T>
using BlockReduce = cub::BlockReduce<T, CUDA_THREADS>;

} // namespace dragon

#endif // USE_CUDA

#endif // DRAGON_UTILS_DEVICE_COMMON_CUB_H_
