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

#ifndef DRAGON_KERNELS_ARRAY_OP_KERNELS_H_
#define DRAGON_KERNELS_ARRAY_OP_KERNELS_H_

#include "dragon/core/context.h"
#include "dragon/core/context_cuda.h"

namespace dragon {

namespace kernels {

template <typename T, class Context>
void Assign(
    const int num_dims,
    const int64_t* x_dims,
    const int64_t* y_strides,
    const int64_t* starts,
    const T* x,
    T* y,
    Context* ctx);

template <typename IndexT, typename ValueT, class Context>
void BooleanMask(
    const int N,
    const IndexT* index,
    const ValueT* x,
    ValueT* y,
    Context* ctx);

template <typename IndexT, typename ValueT, class Context>
void BooleanMaskGrad(
    const int N,
    const IndexT* index,
    const ValueT* dy,
    ValueT* dx,
    Context* ctx);

template <typename T, class Context>
void ConstPad(
    const int num_dims,
    const int64_t* x_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const int64_t* pads,
    const float value,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void EdgePad(
    const int num_dims,
    const int64_t* x_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const int64_t* pads,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void Gather(
    const int N,
    const int S,
    const int C,
    const int K,
    const int64_t* index,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void GatherGrad(
    const int N,
    const int S,
    const int C,
    const int K,
    const int64_t* index,
    const T* dy,
    float* dx,
    Context* ctx);

template <typename T, class Context>
void GatherElements(
    const int axis,
    const int num_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const int64_t* index,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void LinSpace(
    const int N,
    const int C,
    const int axis,
    const double* starts,
    const double* stops,
    T* y,
    Context* ctx);

template <typename T, class Context>
void SetEye(
    const int batch_size,
    const int M,
    const int N,
    const int k,
    T* y,
    Context* ctx);

template <typename T, class Context>
void SetOneHot(
    const int N,
    const int depth,
    const float value,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void SetTrilu(
    const int batch_size,
    const int M,
    const int N,
    const int k,
    const int upper,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void Permutation(const int N, const uint32_t* r, T* y, Context* ctx);

template <typename T, class Context>
void Range(
    const int N,
    const double start,
    const double delta,
    T* y,
    Context* ctx);

template <typename T, class Context>
void ReflectPad(
    const int num_dims,
    const int64_t* x_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const int64_t* pads,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void Repeat(
    const int N,
    const int S,
    const int C,
    const int repeats,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void RepeatGrad(
    const int N,
    const int S,
    const int C,
    const int repeats,
    const T* dy,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void Reverse(
    const int num_dims,
    const uint8_t* x_flips,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void Roll(
    const int num_dims,
    const int64_t* x_shifts,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void ScatterElements(
    const int axis,
    const int num_dims,
    const T value,
    const int64_t* dims,
    const int64_t* y_strides,
    const int64_t* index,
    T* y,
    Context* ctx);

template <typename T, class Context>
void ScatterElements(
    const int axis,
    const int num_dims,
    const int64_t* dims,
    const int64_t* x_strides,
    const int64_t* y_strides,
    const int64_t* index,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, typename AccT, class Context>
void ScatterAdd(
    const int axis,
    const int num_dims,
    const int64_t* dims,
    const int64_t* x_strides,
    const int64_t* y_strides,
    const int64_t* index,
    const T* x,
    AccT* y,
    Context* ctx);

template <typename T, class Context>
void Slice(
    const int num_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const int64_t* starts,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void SliceGrad(
    const int num_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const int64_t* starts,
    const T* dy,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void Tile(
    const int num_dims,
    const int64_t* x_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void Unique(
    const int dim,
    const T* x,
    T* y,
    int64_t* inverse_index,
    int64_t* counts,
    int* num,
    Context* ctx);

} // namespace kernels

} // namespace dragon

#endif // DRAGON_KERNELS_ARRAY_OP_KERNELS_H_
