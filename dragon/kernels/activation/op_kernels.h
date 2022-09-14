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

#ifndef DRAGON_KERNELS_ACTIVATION_OP_KERNELS_H_
#define DRAGON_KERNELS_ACTIVATION_OP_KERNELS_H_

#include "dragon/core/context.h"
#include "dragon/core/context_cuda.h"
#include "dragon/core/context_mps.h"

namespace dragon {

namespace kernels {

template <typename T, class Context>
void Dropout(
    const int N,
    const float ratio,
    const float scale,
    const float* r,
    const T* x,
    T* y,
    uint8_t* mask,
    Context* ctx);

template <typename T, class Context>
void DropPath(
    const int N,
    const int C,
    const float ratio,
    const float scale,
    const float* r,
    const T* x,
    T* y,
    uint8_t* mask,
    Context* ctx);

template <typename T, class Context>
void DropPathGrad(
    const int N,
    const int C,
    const float scale,
    const uint8_t* mask,
    const T* dy,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void DropBlock2d(
    const int N,
    const int C,
    const int H,
    const int W,
    const int block_size,
    const float ratio,
    const float scale,
    const string& data_format,
    const float* r,
    const T* x,
    T* y,
    uint8_t* mask,
    Context* ctx);

template <typename T, class Context>
void Elu(const int N, const float alpha, const T* x, T* y, Context* ctx);

template <typename T, class Context>
void EluGrad(
    const int N,
    const float alpha,
    const T* dy,
    const T* y,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void Gelu(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
void GeluGrad(const int N, const T* dy, const T* x, T* dx, Context* ctx);

template <typename T, class Context>
void ApproxGelu(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
void ApproxGeluGrad(const int N, const T* dy, const T* x, T* dx, Context* ctx);

template <typename T, class Context>
void HardSigmoid(
    const int N,
    const float alpha,
    const float beta,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void HardSigmoidGrad(
    const int N,
    const float alpha,
    const T* dy,
    const T* y,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void HardSwish(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
void HardSwishGrad(const int N, const T* dy, const T* x, T* dx, Context* ctx);

template <typename T, class Context>
void PRelu(
    const int N,
    const int S,
    const int C,
    const string& data_format,
    const T* x,
    const T* w,
    T* y,
    Context* ctx);

template <typename T, class Context>
void PReluGrad(
    const int N,
    const int S,
    const int C,
    const string& data_format,
    const T* dy,
    const T* x,
    const T* w,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void PReluWGrad(
    const int N,
    const int S,
    const int C,
    const string& data_format,
    const T* dy,
    const T* x,
    T* dw,
    Context* ctx);

template <typename T, class Context>
void Relu(const int N, const float alpha, const T* x, T* y, Context* ctx);

template <typename T, class Context>
void ReluGrad(
    const int N,
    const float alpha,
    const T* dy,
    const T* y,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void ReluN(const int N, const float max_value, const T* x, T* y, Context* ctx);

template <typename T, class Context>
void ReluNGrad(
    const int N,
    const float max_value,
    const T* dy,
    const T* y,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void Selu(
    const int N,
    const float alpha,
    const float gamma,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void SeluGrad(
    const int N,
    const float alpha,
    const float gamma,
    const T* dy,
    const T* y,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void Sigmoid(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
void SigmoidGrad(const int N, const T* dy, const T* y, T* dx, Context* ctx);

template <typename T, class Context>
void Silu(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
void SiluGrad(const int N, const T* dy, const T* x, T* dx, Context* ctx);

template <typename T, class Context>
void Softmax(
    const int N,
    const int S,
    const int C,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void SoftmaxGrad(
    const int N,
    const int S,
    const int C,
    const T* dy,
    const T* y,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void LogSoftmax(
    const int N,
    const int S,
    const int C,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void LogSoftmaxGrad(
    const int N,
    const int S,
    const int C,
    const T* dy,
    const T* y,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void Tanh(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
void TanhGrad(const int N, const T* dy, const T* y, T* dx, Context* ctx);

} // namespace kernels

} // namespace dragon

#endif // DRAGON_KERNELS_ACTIVATION_OP_KERNELS_H_
