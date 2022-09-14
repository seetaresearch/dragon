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

#ifndef DRAGON_UTILS_MATH_BLAS_H_
#define DRAGON_UTILS_MATH_BLAS_H_

#include "dragon/core/context.h"
#include "dragon/core/context_cuda.h"
#include "dragon/core/context_mps.h"

namespace dragon {

typedef enum CBLAS_TRANSPOSE {
  CblasNoTrans,
  CblasTrans,
} CBLAS_TRANSPOSE;

namespace math {

template <typename T, class Context>
DRAGON_API void Copy(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Copy(
    const int N,
    const int x_offset,
    const int y_offset,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void CopyMatrix(
    const int M,
    const int N,
    const int ldx,
    const int ldy,
    const int x_offset,
    const int y_offset,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void
Scale(const int N, const float alpha, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
Axpy(const int N, const float alpha, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Axpby(
    const int N,
    const float alpha,
    const T* x,
    const float beta,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void Dot(const int N, const T* a, const T* b, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API T Dot(const int N, const T* a, const T* b, Context* ctx);

template <typename T, class Context>
DRAGON_API void ASum(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API T ASum(const int N, const T* x, Context* ctx);

template <typename T, class Context>
DRAGON_API void Gemv(
    const CBLAS_TRANSPOSE TransA,
    const int M,
    const int N,
    const float alpha,
    const T* A,
    const T* x,
    const float beta,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void Gemm(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const T* A,
    const T* B,
    const float beta,
    T* C,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void GemmBatched(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const T** A,
    const T** B,
    const float beta,
    T** C,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void GemmStridedBatched(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const int A_stride,
    const int B_stride,
    const int C_stride,
    const float alpha,
    const T* A,
    const T* B,
    const float beta,
    T* C,
    Context* ctx);

} // namespace math

} // namespace dragon

#endif // DRAGON_UTILS_MATH_BLAS_H_
