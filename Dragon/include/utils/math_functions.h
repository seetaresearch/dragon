/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *      <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_UTILS_MATH_FUNCTIONS_H_
#define DRAGON_UTILS_MATH_FUNCTIONS_H_

#include <cstdint>
#include <climits>

#include "proto/dragon.pb.h"

namespace dragon {

// We still follow the CBLAS Transpose custom
typedef enum CBLAS_TRANSPOSE {
    CblasNoTrans,
    CblasTrans,
} CBLAS_TRANSPOSE;

namespace math {

/*!
 * ----------------------------------------------
 *
 *
 *            Simple Unary Functions
 *
 *
 * ----------------------------------------------
 */

template <typename T, class Context>
void Exp(
    const int               n,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void Log(
    const int               n,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void Inv(
    const int               n,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void Sqrt(
    const int               n,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void RSqrt(
    const int               n,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void Square(
    const int               n,
    const T*                x,
    T*                      y,
    Context*                ctx);

/*!
 * ----------------------------------------------
 *
 *
 *             Scale Unary Functions
 *
 *
 * ----------------------------------------------
 */

template <typename T, class Context>
void Set(
    const int               n,
    const T                 alpha,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void BroadcastSet(
    const int               rows,
    const int               cols,
    const int               type,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void Pow(
    const int               n,
    const float             exp,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void Scale(
    const int               n,
    const float             alpha,
    const T*                x,
    T*                      y,
    Context*                ctx);

template<typename T, class Context>
void Axpy(
    const int               n,
    const float             alpha,
    const T*                x,
    T*                      y,
    Context*                ctx);

template<typename T, class Context>
void Axpby(
    const int               n,
    const float             alpha,
    const T*                x,
    const float             beta,
    T*                      y,
    Context*                ctx);

template<typename T, class Context>
void AddScalar(
    const int               n,
    const float             alpha,
    T*                      y,
    Context*                ctx);

/*!
 * ----------------------------------------------
 *
 *
 *             Extended Unary Functions
 *
 *
 * ----------------------------------------------
 */

template<typename T, class Context>
void InvStd(
    const int               n,
    const float             eps,
    const T*                x,
    T*                      y,
    Context*                ctx);

template<typename T, class Context>
void Sum(
    const int               n,
    const float             alpha,
    const T*                x,
    T*                      y,
    Context*                ctx);

template<typename T, class Context>
T Sum(
    const int               n,
    const float             alpha,
    const T*                x,
    Context*                ctx);

template<typename T, class Context>
T ASum(
    const int               n,
    const T*                x,
    Context*                ctx);

/*!
 * ----------------------------------------------
 *
 *
 *            Simply Binary Functions
 *
 *
 * ----------------------------------------------
 */

template <typename T, class Context>
void Add(
    const int               n,
    const T*                a,
    const T*                b,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void Sub(
    const int               n,
    const T*                a,
    const T*                b,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void Mul(
    const int               n,
    const T*                a,
    const T*                b,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void Div(
    const int               n,
    const T*                a,
    const T*                b,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void Dot(
    const int               n,
    const T*                a,
    const T*                b,
    T*                      y,
    Context*                ctx);

/*!
 * ----------------------------------------------
 *
 *
 *          Broadcast Binary Functions
 *
 *
 * ----------------------------------------------
 */

template <typename T, class Context>
void BroadcastAdd(
    const int               rows,
    const int               cols,
    const int               type,
    const T*                a,
    const T*                b,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void BroadcastSub(
    const int               rows,
    const int               cols,
    const int               type,
    const T*                a,
    const T*                b,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void BroadcastMul(
    const int               rows,
    const int               cols,
    const int               type,
    const T*                a,
    const T*                b,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void BroadcastDiv(
    const int               rows,
    const int               cols,
    const int               type,
    const T*                a,
    const T*                b,
    T*                      y,
    Context*                ctx);

/*!
 * ----------------------------------------------
 *
 *
 *        Linear Algebra Binary Functions
 *
 *
 * ----------------------------------------------
 */

template <typename T, class Context>
void Gemm(
    const CBLAS_TRANSPOSE   TransA,
    const CBLAS_TRANSPOSE   TransB,
    const int               M,
    const int               N,
    const int               K,
    const float             alpha,
    const T*                A,
    const T*                B,
    const float             beta,
    T*                      C,
    Context*                ctx,
    TensorProto_DataType    math_type = TensorProto_DataType_FLOAT);

template<typename T, class Context>
void Gemv(
    const CBLAS_TRANSPOSE   TransA,
    const int               M,
    const int               N,
    const float             alpha,
    const T*                A,
    const T*                x,
    const float             beta,
    T*                      y,
    Context*                ctx,
    TensorProto_DataType    math_type = TensorProto_DataType_FLOAT);

/*!
 * ----------------------------------------------
 *
 *
 *               Random Functions
 *
 *
 * ----------------------------------------------
 */

template <typename T, class Context>
void RandomUniform(
    const int               n,
    const float             low,
    const float             high,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void RandomNormal(
    const int               n,
    const float             mu,
    const float             sigma,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void RandomTruncatedNormal(
    const int               n,
    const float             mu,
    const float             sigma,
    const float             low,
    const float             high,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void RandomBernoulli(
    const int               n,
    const float             p,
    T*                      y,
    Context*                ctx);

}  // namespace math

}  // namespace dragon

#endif  // DRAGON_UTILS_MATH_FUNCTIONS_H_