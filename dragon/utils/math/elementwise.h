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

#ifndef DRAGON_UTILS_MATH_ELEMENTWISE_H_
#define DRAGON_UTILS_MATH_ELEMENTWISE_H_

#include "dragon/core/context.h"

namespace dragon {

namespace math {

template <typename T, class Context>
DRAGON_API void Abs(const int n, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Ceil(const int n, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Cos(const int n, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Exp(const int n, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Floor(const int n, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Inv(const int n, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
InvStd(const int n, const float eps, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Invert(const int n, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Log(const int n, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
Powx(const int n, const float exponent, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Round(const int n, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Rsqrt(const int n, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Set(const int n, const T value, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Sign(const int n, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Sin(const int n, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Sqrt(const int n, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Square(const int n, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void NotZero(const int n, const T* x, bool* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void IsInf(const int n, const T* x, bool* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void IsNaN(const int n, const T* x, bool* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
ReplaceNaN(const int n, const T value, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
Bias(const int n, const float beta, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Add(const int n, const T* a, const T* b, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Sub(const int n, const T* a, const T* b, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Mul(const int n, const T* a, const T* b, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Div(const int n, const T* a, const T* b, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Pow(const int n, const T* a, const T* b, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
Minimum(const int n, const T* a, const T* b, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
Maximum(const int n, const T* a, const T* b, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
Equal(const int n, const T* a, const T* b, bool* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
NotEqual(const int n, const T* a, const T* b, bool* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
Less(const int n, const T* a, const T* b, bool* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
LessEqual(const int n, const T* a, const T* b, bool* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
Greater(const int n, const T* a, const T* b, bool* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
GreaterEqual(const int n, const T* a, const T* b, bool* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
Where(const int n, const T* a, const T* b, const bool* c, T* y, Context* ctx);

} // namespace math

} // namespace dragon

#endif // DRAGON_UTILS_MATH_ELEMENTWISE_H_
