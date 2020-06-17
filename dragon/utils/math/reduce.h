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

#ifndef DRAGON_UTILS_MATH_REDUCE_H_
#define DRAGON_UTILS_MATH_REDUCE_H_

#include "dragon/core/context.h"

namespace dragon {

namespace math {

template <typename T, class Context>
DRAGON_API void ReduceMax(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void ReduceMin(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void ReduceSum(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const float scale,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void
Sum(const int n, const float alpha, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API T Sum(const int n, const float alpha, const T* x, Context* ctx);

} // namespace math

} // namespace dragon

#endif // DRAGON_UTILS_MATH_REDUCE_H_
