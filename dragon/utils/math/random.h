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

#ifndef DRAGON_UTILS_MATH_RANDOM_H_
#define DRAGON_UTILS_MATH_RANDOM_H_

#include "dragon/core/context.h"

namespace dragon {

namespace math {

template <typename T, class Context>
DRAGON_API void RandomUniform(
    const int n,
    const float low,
    const float high,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void RandomNormal(
    const int n,
    const float mu,
    const float sigma,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void RandomBernoulli(const int n, const float p, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void TruncatedNormal(
    const int n,
    const float mu,
    const float sigma,
    const float low,
    const float high,
    T* y,
    Context* ctx);

} // namespace math

} // namespace dragon

#endif // DRAGON_UTILS_MATH_RANDOM_H_
