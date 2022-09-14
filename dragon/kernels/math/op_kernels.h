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

#ifndef DRAGON_KERNELS_MATH_OP_KERNELS_H_
#define DRAGON_KERNELS_MATH_OP_KERNELS_H_

#include "dragon/core/context.h"
#include "dragon/core/context_cuda.h"
#include "dragon/core/context_mps.h"

namespace dragon {

namespace kernels {

template <typename T, class Context>
void ArgMax(
    const int N,
    const int S,
    const int C,
    const T* x,
    int64_t* y,
    Context* ctx);

template <typename T, class Context>
void ArgMin(
    const int N,
    const int S,
    const int C,
    const T* x,
    int64_t* y,
    Context* ctx);

template <typename T, class Context>
void Clip(
    const int N,
    const float low,
    const float high,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void ClipGrad(
    const int N,
    const float low,
    const float high,
    const T* dy,
    const T* x,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void CosGrad(const int N, const T* dy, const T* x, T* dx, Context* ctx);

template <typename T, class Context>
void CumSum(
    const int N,
    const int S,
    const int C,
    const bool exclusive,
    const bool reverse,
    const T* x,
    T* y,
    Context* ctx);

template <typename IndexT, class Context>
void Flagged(
    const int N,
    const uint8_t* mask,
    IndexT* index,
    int* num_selected,
    Context* ctx);

template <typename T, typename AccT, class Context>
void Moments(
    const int num_dims,
    const int64_t* dims,
    const int num_axes,
    const int64_t* axes,
    const T* x,
    AccT* mean,
    AccT* var,
    Context* ctx);

template <typename T, class Context>
void ReciprocalGrad(const int N, const T* dy, const T* y, T* dx, Context* ctx);

template <typename T, class Context>
void ReduceSumGrad(
    const int num_dims,
    const int64_t* x_dims,
    const int64_t* y_dims,
    const int64_t* y_strides,
    const float scale,
    const T* dy,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void RsqrtGrad(const int N, const T* dy, const T* y, T* dx, Context* ctx);

template <typename T, class Context>
void SinGrad(const int N, const T* dy, const T* x, T* dx, Context* ctx);

template <typename T, class Context>
void TopK(
    const int N,
    const int S,
    const int C,
    const int K,
    const int largest,
    const T* x,
    T* value,
    int64_t* index,
    Context* ctx);

template <typename IndexT, typename CoordT, class Context>
void UnravelIndex(
    const int N,
    const int num_dims,
    const int64_t* dims,
    const IndexT* index,
    CoordT* coord,
    Context* ctx);

} // namespace kernels

} // namespace dragon

#endif // DRAGON_KERNELS_MATH_OP_KERNELS_H_
