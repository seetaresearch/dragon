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

#ifndef DRAGON_KERNELS_NORMALIZATION_OP_KERNELS_H_
#define DRAGON_KERNELS_NORMALIZATION_OP_KERNELS_H_

#include "dragon/core/context.h"
#include "dragon/core/context_cuda.h"
#include "dragon/core/context_mps.h"

namespace dragon {

namespace kernels {

template <typename InputT, typename OutputT, class Context>
void ChannelNorm(
    const int axis,
    const int num_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const InputT* x,
    const float* mean,
    const float* std,
    OutputT* y,
    Context* ctx);

template <typename T, typename AccT, class Context>
void BatchNormExpectation(
    const int N,
    const int C,
    const int S,
    const float normalizer,
    const string& data_format,
    const T* x,
    AccT* ex,
    AccT* ex2,
    Context* ctx);

template <typename T, typename AccT, class Context>
void BatchNorm(
    const int N,
    const int C,
    const int S,
    const string& data_format,
    const T* x,
    const AccT* mu,
    const AccT* rsig,
    const AccT* gamma,
    const AccT* beta,
    AccT* scale,
    AccT* bias,
    T* y,
    Context* ctx);

template <typename T, typename AccT, class Context>
void BatchNormWGrad(
    const int N,
    const int C,
    const int S,
    const string& data_format,
    const T* x,
    const AccT* mu,
    const AccT* rsig,
    const T* dy,
    AccT* dgamma,
    AccT* dbeta,
    Context* ctx);

template <typename T, typename AccT, class Context>
void BatchNormTrainingGrad(
    const int N,
    const int C,
    const int S,
    const float normalizer,
    const string& data_format,
    const T* x,
    const AccT* mu,
    const AccT* rsig,
    const AccT* gamma,
    const AccT* dgamma,
    const AccT* dbeta,
    const T* dy,
    T* dx,
    Context* ctx);

template <typename T, typename AccT, class Context>
void BatchNormInferenceGrad(
    const int N,
    const int C,
    const int S,
    const string& data_format,
    const T* x,
    const AccT* mu,
    const AccT* rsig,
    const AccT* gamma,
    const T* dy,
    AccT* dgamma,
    AccT* dbeta,
    T* dx,
    Context* ctx);

template <typename T, typename AccT, class Context>
void GroupNorm(
    const int N,
    const int G,
    const int D,
    const int S,
    const string& data_format,
    const T* x,
    const AccT* mu,
    const AccT* rsig,
    const AccT* gamma,
    const AccT* beta,
    T* y,
    Context* ctx);

template <typename T, typename AccT, class Context>
void GroupNormGrad(
    const int N,
    const int G,
    const int D,
    const int S,
    const string& data_format,
    const T* x,
    const AccT* mu,
    const AccT* rsig,
    const AccT* gamma,
    const T* dy,
    AccT* ds,
    AccT* db,
    AccT* dgamma,
    AccT* dbeta,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void L1Norm(
    const int N,
    const int S,
    const int C,
    const float normalizer,
    const float epsilon,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void L1NormGrad(
    const int N,
    const int S,
    const int C,
    const float normalizer,
    const float epsilon,
    const T* dy,
    const T* x,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void L2Norm(
    const int N,
    const int S,
    const int C,
    const float normalizer,
    const float epsilon,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void L2NormGrad(
    const int N,
    const int S,
    const int C,
    const float normalizer,
    const float epsilon,
    const T* dy,
    const T* x,
    T* dx,
    Context* ctx);

template <typename T, typename AccT, class Context>
void LayerNorm(
    const int N,
    const int C,
    const float epsilon,
    const T* x,
    const AccT* gamma,
    const AccT* beta,
    AccT* mu,
    AccT* rsig,
    T* y,
    Context* ctx);

} // namespace kernels

} // namespace dragon

#endif // DRAGON_KERNELS_NORMALIZATION_OP_KERNELS_H_
