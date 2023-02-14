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

#ifndef DRAGON_KERNELS_LOSS_OP_KERNELS_H_
#define DRAGON_KERNELS_LOSS_OP_KERNELS_H_

#include "dragon/core/context.h"
#include "dragon/core/context_cuda.h"
#include "dragon/core/context_mlu.h"
#include "dragon/core/context_mps.h"

namespace dragon {

namespace kernels {

template <typename T, class Context>
void BroadcastLossGrad(
    const int N,
    const int S,
    const int C,
    const T* dl,
    T* dx,
    Context* ctx);

template <typename InputT, typename TargetT, class Context>
void MaskLoss(
    const int N,
    const int ignore_index,
    const TargetT* target,
    InputT* loss,
    Context* ctx);

template <typename InputT, typename TargetT, class Context>
void MaskLossGrad(
    const int N,
    const int C,
    const int ignore_index,
    const TargetT* target,
    InputT* dx,
    Context* ctx);

template <typename T, class Context>
void CrossEntropy(
    const int N,
    const T* input,
    const T* target,
    T* loss,
    Context* ctx);

template <typename InputT, typename TargetT, class Context>
void CrossEntropy(
    const int N,
    const int S,
    const int C,
    const int ignore_index,
    const InputT* input,
    const TargetT* target,
    InputT* loss,
    InputT* mask,
    Context* ctx);

template <typename InputT, typename TargetT, class Context>
void NLLLoss(
    const int N,
    const int S,
    const int C,
    const int ignore_index,
    const InputT* input,
    const TargetT* target,
    InputT* loss,
    InputT* mask,
    Context* ctx);

template <typename InputT, typename TargetT, class Context>
void NLLLossGrad(
    const int N,
    const int S,
    const int C,
    const int ignore_index,
    const InputT* input,
    const TargetT* target,
    InputT* dx,
    InputT* mask,
    Context* ctx);

template <typename T, class Context>
void ReduceLoss(
    const int N,
    const int num_masks,
    const float normalizer,
    const T* x,
    const T* mask,
    T* y,
    Context* ctx);

template <typename T, class Context>
void ReduceLossGrad(
    const int N,
    const int num_masks,
    const float normalizer,
    const T* dl,
    const T* mask,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void SigmoidCrossEntropy(
    const int N,
    const T* input,
    const T* target,
    T* loss,
    T* mask,
    Context* ctx);

template <typename T, class Context>
void SigmoidCrossEntropyGrad(
    const int N,
    const T* input,
    const T* target,
    T* dx,
    T* mask,
    Context* ctx);

template <typename InputT, typename TargetT, class Context>
void SigmoidFocalLoss(
    const int N,
    const int S,
    const int C,
    const int start_index,
    const float alpha,
    const float gamma,
    const InputT* input,
    const TargetT* target,
    InputT* loss,
    InputT* mask,
    Context* ctx);

template <typename InputT, typename TargetT, class Context>
void SigmoidFocalLossGrad(
    const int N,
    const int S,
    const int C,
    const int start_index,
    const float alpha,
    const float gamma,
    const InputT* input,
    const TargetT* target,
    InputT* dx,
    InputT* mask,
    Context* ctx);

template <typename T, class Context>
void SmoothL1Loss(
    const int N,
    const float beta,
    const T* input,
    const T* target,
    T* loss,
    Context* ctx);

template <typename T, class Context>
void SmoothL1LossGrad(
    const int N,
    const float beta,
    const T* input,
    const T* target,
    T* dx,
    Context* ctx);

template <typename InputT, typename TargetT, class Context>
void SoftmaxCrossEntropyGrad(
    const int N,
    const int S,
    const int C,
    const int ignore_index,
    const InputT* input,
    const TargetT* target,
    InputT* dx,
    InputT* mask,
    Context* ctx);

} // namespace kernels

} // namespace dragon

#endif // DRAGON_KERNELS_LOSS_OP_KERNELS_H_
