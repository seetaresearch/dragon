/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *    <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_UTILS_MATH_BROADCAST_H_
#define DRAGON_UTILS_MATH_BROADCAST_H_

#include "dragon/core/context.h"

namespace dragon {

namespace math {

template <typename T, class Context>
DRAGON_API void Set(
    const int x_ndim,
    const int64_t* x_dims,
    const int y_ndim,
    const int64_t* y_dims,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void Add(
    const int a_ndim,
    const int64_t* a_dims,
    const int b_ndim,
    const int64_t* b_dims,
    const T* a,
    const T* b,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void Sub(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void Mul(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void Div(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void Pow(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void Minimum(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void Maximum(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void Equal(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    bool* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void NotEqual(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    bool* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void Less(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    bool* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void LessEqual(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    bool* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void Greater(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    bool* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void GreaterEqual(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const T* a,
    const T* b,
    bool* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void Where(
    const int A_ndim,
    const int64_t* A_dims,
    const int B_ndim,
    const int64_t* B_dims,
    const int C_ndim,
    const int64_t* C_dims,
    const T* a,
    const T* b,
    const bool* c,
    T* y,
    Context* ctx);

} // namespace math

} // namespace dragon

#endif // DRAGON_UTILS_MATH_BROADCAST_H_
