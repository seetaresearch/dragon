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

#ifndef DRAGON_UTILS_OP_KERNELS_H_
#define DRAGON_UTILS_OP_KERNELS_H_

#include "dragon/core/context.h"

namespace dragon {

namespace kernels {

/*
 * ActivationOp Kernels
 */

template <typename T, class Context>
void Dropout(
    const int N,
    const float ratio,
    const float scale,
    const T* x,
    T* y,
    uint8_t* mask,
    uint32_t* r,
    Context* ctx);

template <typename T, class Context>
void DropBlock2d(
    const int N,
    const int C,
    const int H,
    const int W,
    const int block_size,
    const float ratio,
    const float scale,
    const string& data_format,
    const T* x,
    T* y,
    uint8_t* mask,
    uint32_t* r,
    Context* ctx);

template <typename T, class Context>
void DropPath(
    const int N,
    const int C,
    const float ratio,
    const float scale,
    const T* x,
    T* y,
    uint8_t* mask,
    uint32_t* r,
    Context* ctx);

template <typename T, class Context>
void DropPathGrad(
    const int N,
    const int C,
    const float scale,
    const uint8_t* mask,
    const T* dy,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void Elu(const int N, const float alpha, const T* x, T* y, Context* ctx);

template <typename T, class Context>
void EluGrad(
    const int N,
    const float alpha,
    const T* dy,
    const T* y,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void Gelu(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
void GeluGrad(const int N, const T* dy, const T* x, T* dx, Context* ctx);

template <typename T, class Context>
void ApproxGelu(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
void ApproxGeluGrad(const int N, const T* dy, const T* x, T* dx, Context* ctx);

template <typename T, class Context>
void HardSigmoid(
    const int N,
    const float alpha,
    const float beta,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void HardSigmoidGrad(
    const int N,
    const float alpha,
    const T* dy,
    const T* y,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void HardSwish(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
void HardSwishGrad(const int N, const T* dy, const T* x, T* dx, Context* ctx);

template <typename T, class Context>
void PRelu(
    const int N,
    const int S,
    const int C,
    const string& data_format,
    const T* x,
    const T* w,
    T* y,
    Context* ctx);

template <typename T, class Context>
void PReluGrad(
    const int N,
    const int S,
    const int C,
    const string& data_format,
    const T* dy,
    const T* x,
    const T* w,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void PReluWGrad(
    const int N,
    const int S,
    const int C,
    const string& data_format,
    const T* dy,
    const T* x,
    T* dw,
    Context* ctx);

template <typename T, class Context>
void Relu(const int N, const float alpha, const T* x, T* y, Context* ctx);

template <typename T, class Context>
void ReluGrad(
    const int N,
    const float alpha,
    const T* dy,
    const T* y,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void ReluN(const int N, const float max_value, const T* x, T* y, Context* ctx);

template <typename T, class Context>
void ReluNGrad(
    const int N,
    const float max_value,
    const T* dy,
    const T* y,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void Selu(
    const int N,
    const float alpha,
    const float gamma,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void SeluGrad(
    const int N,
    const float alpha,
    const float gamma,
    const T* dy,
    const T* y,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void Sigmoid(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
void SigmoidGrad(const int N, const T* dy, const T* y, T* dx, Context* ctx);

template <typename T, class Context>
void Silu(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
void SiluGrad(const int N, const T* dy, const T* x, T* dx, Context* ctx);

template <typename T, class Context>
void Softmax(
    const int N,
    const int S,
    const int C,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void SoftmaxGrad(
    const int N,
    const int S,
    const int C,
    const T* dy,
    const T* y,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void LogSoftmax(
    const int N,
    const int S,
    const int C,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void LogSoftmaxGrad(
    const int N,
    const int S,
    const int C,
    const T* dy,
    const T* y,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void Tanh(const int N, const T* x, T* y, Context* ctx);

template <typename T, class Context>
void TanhGrad(const int N, const T* dy, const T* y, T* dx, Context* ctx);

/*
 * ArrayOp Kernels
 */

template <typename T, class Context>
void ArgMax(
    const int N,
    const int S,
    const int C,
    const T* x,
    int64_t* y,
    Context* ctx);

template <typename T, class Context>
void ArgMin(
    const int N,
    const int S,
    const int C,
    const T* x,
    int64_t* y,
    Context* ctx);

template <typename T, class Context>
void Assign(
    const int num_dims,
    const int64_t* x_dims,
    const int64_t* y_strides,
    const int64_t* starts,
    const T* x,
    T* y,
    Context* ctx);

template <typename IndexT, typename ValueT, class Context>
void BooleanMask(
    const int N,
    const IndexT* index,
    const ValueT* x,
    ValueT* y,
    Context* ctx);

template <typename IndexT, typename ValueT, class Context>
void BooleanMaskGrad(
    const int N,
    const IndexT* index,
    const ValueT* dy,
    ValueT* dx,
    Context* ctx);

template <typename T, class Context>
void ChannelAffine(
    const int N,
    const int S,
    const int C,
    const T* x,
    const T* scale,
    const T* bias,
    T* y,
    Context* ctx);

template <typename InputT, typename OutputT, class Context>
void ChannelNormalize(
    const int axis,
    const int num_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const InputT* x,
    const float* mean,
    const float* std,
    OutputT* y,
    Context* ctx);

template <typename T, class Context>
void ChannelShuffle(
    const int N,
    const int S,
    const int C,
    const int G,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void ConstPad(
    const int num_dims,
    const int64_t* x_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const int64_t* pads,
    const float value,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void CumSum(
    const int N,
    const int S,
    const int C,
    const bool exclusive,
    const bool reverse,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void EdgePad(
    const int num_dims,
    const int64_t* x_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const int64_t* pads,
    const T* x,
    T* y,
    Context* ctx);

template <typename IndexT, class Context>
void Flagged(
    const int N,
    const uint8_t* mask,
    IndexT* index,
    int* num_selected,
    Context* ctx);

template <typename T, class Context>
void Gather(
    const int N,
    const int S,
    const int C,
    const int K,
    const int64_t* index,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void GatherGrad(
    const int N,
    const int S,
    const int C,
    const int K,
    const int64_t* index,
    const T* dy,
    float* dx,
    Context* ctx);

template <typename T, class Context>
void GatherElements(
    const int axis,
    const int num_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const int64_t* index,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void LinSpace(
    const int N,
    const int C,
    const int axis,
    const double* starts,
    const double* stops,
    T* y,
    Context* ctx);

template <typename T, class Context>
void SetEye(
    const int batch_size,
    const int M,
    const int N,
    const int k,
    T* y,
    Context* ctx);

template <typename T, class Context>
void SetOneHot(
    const int N,
    const int depth,
    const float value,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void SetTrilu(
    const int batch_size,
    const int M,
    const int N,
    const int k,
    const int upper,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void Permutation(const int N, T* y, uint32_t* r, Context* ctx);

template <typename T, class Context>
void Range(
    const int N,
    const double start,
    const double delta,
    T* y,
    Context* ctx);

template <typename T, class Context>
void ReduceSumGrad(
    const int num_dims,
    const int64_t* x_dims,
    const int64_t* y_dims,
    const int64_t* y_strides,
    const float scale,
    const T* dy,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void ReflectPad(
    const int num_dims,
    const int64_t* x_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const int64_t* pads,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void Repeat(
    const int N,
    const int S,
    const int C,
    const int repeats,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void RepeatGrad(
    const int N,
    const int S,
    const int C,
    const int repeats,
    const T* dy,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void Reverse(
    const int num_dims,
    const uint8_t* x_flips,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void Roll(
    const int num_dims,
    const int64_t* x_shifts,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void ScatterElements(
    const int axis,
    const int num_dims,
    const T value,
    const int64_t* dims,
    const int64_t* y_strides,
    const int64_t* index,
    T* y,
    Context* ctx);

template <typename T, class Context>
void ScatterElements(
    const int axis,
    const int num_dims,
    const int64_t* dims,
    const int64_t* x_strides,
    const int64_t* y_strides,
    const int64_t* index,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, typename AccT, class Context>
void ScatterAdd(
    const int axis,
    const int num_dims,
    const int64_t* dims,
    const int64_t* x_strides,
    const int64_t* y_strides,
    const int64_t* index,
    const T* x,
    AccT* y,
    Context* ctx);

template <typename T, class Context>
void Slice(
    const int num_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const int64_t* starts,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void SliceGrad(
    const int num_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const int64_t* starts,
    const T* dy,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void Tile(
    const int num_dims,
    const int64_t* x_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void TileGrad(
    const int N,
    const int CxS,
    const int repeats,
    const T* dy,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void Transpose(
    const int num_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void TopK(
    const int N,
    const int S,
    const int C,
    const int K,
    const int largest,
    const T* x,
    T* value,
    int64_t* index,
    Context* ctx);

template <typename T, class Context>
void Unique(
    const int dim,
    const T* x,
    T* y,
    int64_t* inverse_index,
    int64_t* counts,
    int* num,
    Context* ctx);

template <typename IndexT, typename CoordT, class Context>
void UnravelIndex(
    const int N,
    const int num_dims,
    const int64_t* dims,
    const IndexT* index,
    CoordT* coord,
    Context* ctx);

/*
 * LossOp Kernels
 */

template <typename T, class Context>
void BroadcastLossGrad(
    const int N,
    const int S,
    const int C,
    const T* dl,
    T* dx,
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
void SmoothL1(const int N, const float beta, const T* x, T* y, Context* ctx);

template <typename T, class Context>
void SmoothL1Grad(
    const int N,
    const float beta,
    const T* x,
    T* y,
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

/*
 * MathOp Kernels
 */

template <typename T, class Context>
void Clip(
    const int N,
    const float low,
    const float high,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void ClipGrad(
    const int N,
    const float low,
    const float high,
    const T* dy,
    const T* x,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void CosGrad(const int N, const T* dy, const T* x, T* dx, Context* ctx);

template <typename T, typename AccT, class Context>
void Moments(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const T* x,
    AccT* mean,
    AccT* var,
    Context* ctx);

template <typename T, class Context>
void ReciprocalGrad(const int N, const T* dy, const T* y, T* dx, Context* ctx);

template <typename T, class Context>
void RsqrtGrad(const int N, const T* dy, const T* y, T* dx, Context* ctx);

template <typename T, class Context>
void SinGrad(const int N, const T* dy, const T* x, T* dx, Context* ctx);

/*
 * NormalizationOp Kernels
 */

template <typename T, typename AccT, class Context>
void BatchNormExpectation(
    const int N,
    const int C,
    const int S,
    const float normalizer,
    const string& data_format,
    const T* x,
    AccT* ex,
    AccT* ex2,
    Context* ctx);

template <typename T, typename AccT, class Context>
void BatchNorm(
    const int N,
    const int C,
    const int S,
    const string& data_format,
    const T* x,
    const AccT* mu,
    const AccT* rsig,
    const AccT* gamma,
    const AccT* beta,
    AccT* scale,
    AccT* bias,
    T* y,
    Context* ctx);

template <typename T, typename AccT, class Context>
void BatchNormWGrad(
    const int N,
    const int C,
    const int S,
    const string& data_format,
    const T* x,
    const AccT* mu,
    const AccT* rsig,
    const T* dy,
    AccT* dgamma,
    AccT* dbeta,
    Context* ctx);

template <typename T, typename AccT, class Context>
void BatchNormTrainingGrad(
    const int N,
    const int C,
    const int S,
    const float normalizer,
    const string& data_format,
    const T* x,
    const AccT* mu,
    const AccT* rsig,
    const AccT* gamma,
    const AccT* dgamma,
    const AccT* dbeta,
    const T* dy,
    T* dx,
    Context* ctx);

template <typename T, typename AccT, class Context>
void BatchNormInferenceGrad(
    const int N,
    const int C,
    const int S,
    const string& data_format,
    const T* x,
    const AccT* mu,
    const AccT* rsig,
    const AccT* gamma,
    const T* dy,
    AccT* dgamma,
    AccT* dbeta,
    T* dx,
    Context* ctx);

template <typename T, typename AccT, class Context>
void GroupNorm(
    const int N,
    const int G,
    const int D,
    const int S,
    const string& data_format,
    const T* x,
    const AccT* mu,
    const AccT* rsig,
    const AccT* gamma,
    const AccT* beta,
    AccT* scale,
    AccT* bias,
    T* y,
    Context* ctx);

template <typename T, typename AccT, class Context>
void GroupNormGrad(
    const int N,
    const int G,
    const int D,
    const int S,
    const string& data_format,
    const T* x,
    const AccT* mu,
    const AccT* rsig,
    const AccT* gamma,
    const T* dy,
    AccT* ds,
    AccT* db,
    AccT* dgamma,
    AccT* dbeta,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void L1Normalize(
    const int N,
    const int S,
    const int C,
    const float normalizer,
    const float epsilon,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void L1NormalizeGrad(
    const int N,
    const int S,
    const int C,
    const float normalizer,
    const float epsilon,
    const T* dy,
    const T* x,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void L2Normalize(
    const int N,
    const int S,
    const int C,
    const float normalizer,
    const float epsilon,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void L2NormalizeGrad(
    const int N,
    const int S,
    const int C,
    const float normalizer,
    const float epsilon,
    const T* dy,
    const T* x,
    T* dx,
    Context* ctx);

template <typename T, typename AccT, class Context>
void LayerNorm(
    const int N,
    const int C,
    const float epsilon,
    const T* x,
    const AccT* gamma,
    const AccT* beta,
    AccT* mu,
    AccT* rsig,
    T* y,
    Context* ctx);

/*
 * RecurrentOp Kernels
 */

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

/*
 * TrainingOp Kernels
 */

template <typename T, class Context>
void AdamUpdate(
    const int N,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    T* g,
    T* m,
    T* v,
    Context* ctx);

template <typename T, class Context>
void NesterovUpdate(
    const int N,
    const float lr,
    const float momentum,
    T* g,
    T* m,
    Context* ctx);

template <typename T, class Context>
void RMSPropUpdate(
    const int N,
    const float lr,
    const float momentum,
    const float decay,
    const float eps,
    T* g,
    T* m,
    T* v,
    Context* ctx);

template <typename T, class Context>
void SGDUpdate(
    const int N,
    const float lr,
    const float momentum,
    T* g,
    T* m,
    Context* ctx);

/*
 * VisionOp Kernels
 */

template <typename T, class Context>
void AvgPool2d(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const string& data_format,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void AvgPool2dGrad(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const string& data_format,
    const T* dy,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void AvgPool3d(
    const int N,
    const int C,
    const int D,
    const int H,
    const int W,
    const int out_d,
    const int out_h,
    const int out_w,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const string& data_format,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void AvgPool3dGrad(
    const int N,
    const int C,
    const int D,
    const int H,
    const int W,
    const int out_d,
    const int out_h,
    const int out_w,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const string& data_format,
    const T* dy,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void BiasAdd(
    const int N,
    const int S,
    const int C,
    const T* x,
    const T* bias,
    T* y,
    Context* ctx);

template <typename T, class Context>
void Col2Im2d(
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const string& data_format,
    const T* col,
    T* im,
    Context* ctx);

template <typename T, class Context>
void Col2ImNd(
    const int num_dims,
    const int channels,
    const int* in_shape,
    const int* out_shape,
    const int* kshape,
    const int* strides,
    const int* pads,
    const int* dilations,
    const string& data_format,
    const T* col,
    T* im,
    Context* ctx);

template <typename T, class Context>
void Im2Col2d(
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const string& data_format,
    const T* im,
    T* col,
    Context* ctx);

template <typename T, class Context>
void Im2ColNd(
    const int num_dims,
    const int channels,
    const int* in_shape,
    const int* out_shape,
    const int* kshape,
    const int* strides,
    const int* pads,
    const int* dilations,
    const string& data_format,
    const T* im,
    T* col,
    Context* ctx);

template <typename T, class Context>
void DepthwiseConv2d(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const string& data_format,
    const T* x,
    const T* filter,
    T* y,
    Context* ctx);

template <typename T, class Context>
void DepthwiseConv2dGrad(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const string& data_format,
    const T* dy,
    const T* filter,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void DepthwiseConv2dWGrad(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const string& data_format,
    const T* dy,
    const T* x,
    T* dfilter,
    Context* ctx);

template <typename T, class Context>
void MaxPool2d(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const string& data_format,
    const T* x,
    int* mask,
    T* y,
    Context* ctx);

template <typename T, class Context>
void MaxPool2dGrad(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const string& data_format,
    const T* dy,
    int* mask,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void MaxPool3d(
    const int N,
    const int C,
    const int D,
    const int H,
    const int W,
    const int out_d,
    const int out_h,
    const int out_w,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const string& data_format,
    const T* x,
    int* mask,
    T* y,
    Context* ctx);

template <typename T, class Context>
void MaxPool3dGrad(
    const int N,
    const int C,
    const int D,
    const int H,
    const int W,
    const int out_d,
    const int out_h,
    const int out_w,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const string& data_format,
    const T* dy,
    int* mask,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void ResizeLinear2d(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const bool align_corners,
    const string& data_format,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void ResizeLinear2dGrad(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const bool align_corners,
    const string& data_format,
    const T* dy,
    float* dx,
    Context* ctx);

template <typename T, class Context>
void ResizeNearest2d(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const string& data_format,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void ResizeNearest2dGrad(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const string& data_format,
    const T* dy,
    float* dx,
    Context* ctx);

template <typename T, class Context>
void ResizeNearest3d(
    const int N,
    const int C,
    const int D,
    const int H,
    const int W,
    const int out_d,
    const int out_h,
    const int out_w,
    const string& data_format,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void ResizeNearest3dGrad(
    const int N,
    const int C,
    const int D,
    const int H,
    const int W,
    const int out_d,
    const int out_h,
    const int out_w,
    const string& data_format,
    const T* dy,
    float* dx,
    Context* ctx);

template <typename T, class Context>
void RoiAlign(
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int num_rois,
    const float spatial_scale,
    const int sampling_ratio,
    const T* x,
    const float* rois,
    T* y,
    Context* ctx);

template <typename T, class Context>
void RoiAlignGrad(
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int num_rois,
    const float spatial_scale,
    const int sampling_ratio,
    const T* dy,
    const float* rois,
    float* dx,
    Context* ctx);

template <typename T, class Context>
void RoiPool(
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int num_rois,
    const float spatial_scale,
    const T* x,
    const float* rois,
    int* mask,
    T* y,
    Context* ctx);

template <typename T, class Context>
void RoiPoolGrad(
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int num_rois,
    const float spatial_scale,
    const T* dy,
    const float* rois,
    int* mask,
    float* dx,
    Context* ctx);

} // namespace kernels

} // namespace dragon

#endif // DRAGON_UTILS_OP_KERNELS_H_
