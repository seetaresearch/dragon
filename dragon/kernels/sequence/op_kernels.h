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

#ifndef DRAGON_KERNELS_RECURRENT_OP_KERNELS_H_
#define DRAGON_KERNELS_RECURRENT_OP_KERNELS_H_

#include "dragon/core/context.h"
#include "dragon/core/context_cuda.h"

namespace dragon {

namespace kernels {

template <typename T, class Context>
void LSTMCell(
    const int N,
    const int C,
    const T* c_prev,
    T* x,
    T* c,
    T* h,
    Context* ctx);

template <typename T, class Context>
void LSTMCellGrad(
    const int N,
    const int C,
    const T* cx,
    const T* actx,
    const T* c,
    const T* dc,
    const T* dh,
    T* dcx,
    T* dx,
    Context* ctx);

} // namespace kernels

} // namespace dragon

#endif // DRAGON_KERNELS_RECURRENT_OP_KERNELS_H_
