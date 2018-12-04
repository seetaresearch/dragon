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

#ifndef DRAGON_UTILS_SSE_ALTERNATIVE_H_
#define DRAGON_UTILS_SSE_ALTERNATIVE_H_

#ifdef WITH_SSE

#include "utils/sse_device.h"

namespace dragon {

namespace sse {

/******************** Level-0 ********************/

template <typename T>
void Set(
    const int               n,
    const T                 alpha,
    T*                      x);

/******************** Level-1 ********************/

template <typename T>
void Add(
    const int               n,
    const T*                a,
    const T*                b,
    T*                      y);

template <typename T>
void Sub(
    const int               n,
    const T*                a,
    const T*                b,
    T*                      y);

template <typename T>
void Mul(
    const int               n,
    const T*                a,
    const T*                b,
    T*                      y);

template <typename T>
void Div(
    const int               n,
    const T*                a,
    const T*                b,
    T*                      y);

/******************** Level-2 ********************/

template <typename T>
void Scal(
    const int               n,
    const T                 alpha,
    T*                      y);

template <typename T>
void Scale(
    const int               n,
    const T                 alpha,
    const T*                x,
    T*                      y);

template <typename T>
T Dot(
    const int               n,
    const T*                a,
    const T*                b);

template<typename T>
T Sum(
    const int               n,
    const T*                x);

template<typename T>
void AddScalar(
    const int               n,
    const T                 alpha,
    T*                      y);

template<typename T>
void MulScalar(
    const int               n,
    const T                 alpha,
    T*                      y);

template<typename T>
void Axpy(
    const int               n,
    const T                 alpha,
    const T*                x,
    T*                      y);

template<typename T>
void Axpby(
    const int               n,
    const T                 alpha,
    const T*                x,
    const T                 beta,
    T*                      y);

}  // namespace sse

}  // namespace dragon

#endif  // WITH_SSE

#endif  // DRAGON_UTILS_SSE_ALTERNATIVE_H_