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
#include "dragon/core/context_cuda.h"
#include "dragon/core/context_mps.h"

namespace dragon {

namespace math {

template <typename T, class Context>
DRAGON_API void Abs(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Neg(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Ceil(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Cos(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Exp(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Floor(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Inv(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
InvStd(const int N, const float eps, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Log(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
Powx(const int N, const float exponent, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Round(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Rsqrt(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Set(const int N, const T value, T* y, Context* ctx);

template <typename Tx, typename Ty, class Context>
DRAGON_API void Cast(const int N, const Tx* x, Ty* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Sign(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Sin(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Sqrt(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Square(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void BitwiseNot(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Not(const int N, const T* x, bool* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void NotZero(const int N, const T* x, bool* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void IsInf(const int N, const T* x, bool* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void IsNaN(const int N, const T* x, bool* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void IsFinite(const int N, const T* x, bool* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
ReplaceNaN(const int N, const float value, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void ApplyMask(
    const int N,
    const float alpha,
    const uint8_t* mask,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
DRAGON_API void
Bias(const int N, const float beta, const T* x, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Add(const int N, const T* a, const T* b, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Sub(const int N, const T* a, const T* b, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Mul(const int N, const T* a, const T* b, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Div(const int N, const T* a, const T* b, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Pow(const int N, const T* a, const T* b, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Atan2(const int N, const T* a, const T* b, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
Minimum(const int N, const T* a, const T* b, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
Maximum(const int N, const T* a, const T* b, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
BitwiseAnd(const int N, const T* a, const T* b, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
BitwiseOr(const int N, const T* a, const T* b, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
BitwiseXor(const int N, const T* a, const T* b, T* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void And(const int N, const T* a, const T* b, bool* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Or(const int N, const T* a, const T* b, bool* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void Xor(const int N, const T* a, const T* b, bool* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
Equal(const int N, const T* a, const T* b, bool* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
NotEqual(const int N, const T* a, const T* b, bool* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
Less(const int N, const T* a, const T* b, bool* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
LessEqual(const int N, const T* a, const T* b, bool* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
Greater(const int N, const T* a, const T* b, bool* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
GreaterEqual(const int N, const T* a, const T* b, bool* y, Context* ctx);

template <typename T, class Context>
DRAGON_API void
Where(const int N, const T* a, const T* b, const bool* c, T* y, Context* ctx);

} // namespace math

} // namespace dragon

#endif // DRAGON_UTILS_MATH_ELEMENTWISE_H_
