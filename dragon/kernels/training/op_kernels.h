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

#ifndef DRAGON_KERNELS_TRAINING_OP_KERNELS_H_
#define DRAGON_KERNELS_TRAINING_OP_KERNELS_H_

#include "dragon/core/context.h"
#include "dragon/core/context_cuda.h"
#include "dragon/core/context_mps.h"

namespace dragon {

namespace kernels {

template <typename T, typename CopyT, class Context>
void Adam(
    const int N,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float wd,
    const T* x,
    const T* g,
    T* m,
    T* v,
    T* y,
    CopyT* y_copy,
    Context* ctx);

template <typename T, typename CopyT, class Context>
void AdamW(
    const int N,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float wd,
    const T* x,
    const T* g,
    T* m,
    T* v,
    T* y,
    CopyT* y_copy,
    Context* ctx);

template <typename T, typename CopyT, class Context>
void MomentumSGD(
    const int N,
    const float lr,
    const float momentum,
    const float wd,
    const T* x,
    const T* g,
    T* m,
    T* y,
    CopyT* y_copy,
    Context* ctx);

template <typename T, typename CopyT, class Context>
void NesterovSGD(
    const int N,
    const float lr,
    const float momentum,
    const float wd,
    const T* x,
    const T* g,
    T* m,
    T* y,
    CopyT* y_copy,
    Context* ctx);

template <typename T, typename CopyT, class Context>
void RMSprop(
    const int N,
    const float lr,
    const float momentum,
    const float alpha,
    const float eps,
    const float wd,
    const T* x,
    const T* g,
    T* m,
    T* v,
    T* y,
    CopyT* y_copy,
    Context* ctx);

} // namespace kernels

} // namespace dragon

#endif // DRAGON_KERNELS_TRAINING_OP_KERNELS_H_
