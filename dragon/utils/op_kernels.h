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

namespace kernel {

/* activation.dropout */

template <typename T, class Context>
void ApplyMask(
    const int count,
    const float scale,
    const T* x,
    const uint8_t* mask,
    T* y,
    Context* ctx);

template <typename T, class Context>
void Dropout(
    const int count,
    const float prob,
    const float scale,
    const T* x,
    uint8_t* mask,
    T* y,
    uint32_t* r,
    Context* ctx);

/* activation.drop_block */

template <class Context>
void DropBlock2d(
    const int N,
    const int C,
    const int H,
    const int W,
    const int seed_h,
    const int seed_w,
    const int block_size,
    const float gamma,
    const string& data_format,
    uint32_t* r,
    int* mask,
    Context* ctx);

/* activation.drop_path */

template <typename T, class Context>
void DropPath(
    const int rows,
    const int cols,
    const float scale,
    const T* x,
    const float* mask,
    T* y,
    Context* ctx);

/* activation.elu */

template <typename T, class Context>
void Elu(const int count, const float alpha, const T* x, T* y, Context* ctx);

template <typename T, class Context>
void EluGrad(
    const int count,
    const float alpha,
    const T* dy,
    const T* y,
    T* dx,
    Context* ctx);

/* activation.prelu */

template <typename T, class Context>
void PRelu(
    const int N,
    const int C,
    const int S,
    const string& data_format,
    const T* x,
    const T* w,
    T* y,
    Context* ctx);

template <typename T, class Context>
void PReluGrad(
    const int N,
    const int C,
    const int S,
    const string& data_format,
    const T* dy,
    const T* x,
    const T* w,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void PReluWGrad(
    const int N,
    const int C,
    const int S,
    const string& data_format,
    const T* dy,
    const T* x,
    T* dw,
    Context* ctx);

/* activation.relu */

template <typename T, class Context>
void Relu(const int count, const float alpha, const T* x, T* y, Context* ctx);

template <typename T, class Context>
void ReluN(
    const int count,
    const float max_value,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void ReluGrad(
    const int count,
    const float alpha,
    const T* dy,
    const T* y,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void ReluNGrad(
    const int count,
    const float max_value,
    const T* dy,
    const T* y,
    T* dx,
    Context* ctx);

/* activation.selu */

template <typename T, class Context>
void Selu(
    const int count,
    const float alpha,
    const float gamma,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void SeluGrad(
    const int count,
    const float alpha,
    const float gamma,
    const T* dy,
    const T* y,
    T* dx,
    Context* ctx);

/* activation.sigmoid */

template <typename T, class Context>
void Sigmoid(const int count, const T* x, T* y, Context* ctx);

template <typename T, class Context>
void SigmoidGrad(const int count, const T* dy, const T* y, T* dx, Context* ctx);

/* activation.softmax */

template <typename T, class Context>
void Softmax(
    const int outer_dim,
    const int axis_dim,
    const int inner_dim,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void SoftmaxGrad(
    const int outer_dim,
    const int axis_dim,
    const int inner_dim,
    const T* dy,
    const T* y,
    T* dx,
    Context* ctx);

/* activation.tanh */

template <typename T, class Context>
void Tanh(const int count, const T* x, T* y, Context* ctx);

template <typename T, class Context>
void TanhGrad(const int count, const T* dy, const T* y, T* dx, Context* ctx);

/* array.argmax */

template <typename T, class Context>
void ArgMax(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const T* x,
    int64_t* y,
    Context* ctx);

/* array.argmin */

template <typename T, class Context>
void ArgMin(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const T* x,
    int64_t* y,
    Context* ctx);

/* array.channel_affine */

template <typename T, class Context>
void ChannelAffine(
    const int outer_dim,
    const int axis_dim,
    const int inner_dim,
    const T* x,
    const T* w,
    const T* b,
    T* y,
    Context* ctx);

/* array.channel_normalize */

template <typename Tx, typename Ty, class Context>
void ChannelNormalize(
    const int axis,
    const int num_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const Tx* x,
    const float* mean,
    const float* std,
    Ty* y,
    Context* ctx);

/* array.channel_shuffle */

template <typename T, class Context>
void ChannelShuffle(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const int group,
    const T* x,
    T* y,
    Context* ctx);

/* array.cumsum */

template <typename T, class Context>
void CumSum(
    const int outer_dim,
    const int axis_dim,
    const int inner_dim,
    const bool exclusive,
    const bool reverse,
    const T* x,
    T* y,
    Context* ctx);

/* array.eye */

template <typename T, class Context>
void Eye(const int n, const int m, const int k, T* y, Context* ctx);

/* array.index_select */

template <typename T, class Context>
void IndexSelect(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const int select_dim,
    const int64_t* indices,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void IndexSelectGrad(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const int select_dim,
    const int64_t* index,
    const T* dy,
    T* dx,
    Context* ctx);

/* array.linspace */

template <typename T, class Context>
void LinSpace(
    const int rows,
    const int cols,
    const int axis,
    const double* start,
    const double* end,
    T* y,
    Context* ctx);

/* array.masked_select */

template <typename IndexType, typename ValueType, class Context>
void MaskedSelect(
    const int num_selected,
    const IndexType* index,
    const ValueType* x,
    ValueType* y,
    Context* ctx);

template <typename IndexType, typename ValueType, class Context>
void MaskedSelectGrad(
    const int count,
    const int num_selected,
    const IndexType* index,
    const ValueType* dy,
    ValueType* dx,
    Context* ctx);

/* array.non_zero */

template <typename IndexType, class Context>
void Flagged(
    const int count,
    const uint8_t* mask,
    IndexType* index,
    int* num_selected,
    Context* ctx);

template <typename IndexType, typename CoordType, class Context>
void UnravelIndex(
    const int count,
    const int num_dims,
    const int64_t* dims,
    const IndexType* index,
    CoordType* coord,
    Context* ctx);

/* array.pad */

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
void EdgePad(
    const int num_dims,
    const int64_t* x_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const int64_t* pads,
    const T* x,
    T* y,
    Context* ctx);

/* array.one_hot */

template <typename T, class Context>
void OneHot(
    const int count,
    const int depth,
    const int on_value,
    const T* x,
    T* y,
    Context* ctx);

/* array.permutation */

template <typename T, class Context>
void Permutation(const int count, T* y, uint32_t* r, Context* ctx);

/* array.range */

template <typename T, class Context>
void Range(
    const int count,
    const double start,
    const double delta,
    T* y,
    Context* ctx);

/* array.reduce */

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

/* array.repeat */

template <typename T, class Context>
void Repeat(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const int repeats,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void RepeatGrad(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const int repeats,
    const T* dy,
    T* dx,
    Context* ctx);

/* array.slice */

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

/* array.tile */

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
    const int rows,
    const int cols,
    const int multiple,
    const T* dy,
    T* dx,
    Context* ctx);

/* array.transpose */

template <typename T, class Context>
void Transpose(
    const int num_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void TransposeGrad(
    const int num_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const T* dy,
    T* dx,
    Context* ctx);

/* array.top_k */

template <typename T, class Context>
void TopSelect(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const int topk,
    const int largest,
    const T* x,
    T* value,
    int64_t* index,
    Context* ctx);

/* array.unique */

template <typename T, class Context>
void Unique(
    const int dim,
    const T* x,
    T* y,
    int64_t* inverse_index,
    int64_t* counts,
    int* num,
    Context* ctx);

/* control_flow.assgin */

template <typename T, class Context>
void Assign(
    const int num_dims,
    const int64_t* x_dims,
    const int64_t* y_strides,
    const int64_t* starts,
    const T* x,
    T* y,
    Context* ctx);

/* loss.generic_loss */

template <typename T, class Context>
void ReduceLoss(
    const int count,
    const int num_masks,
    const float normalizer,
    const T* x,
    const T* mask,
    T* y,
    Context* ctx);

template <typename T, class Context>
void ReduceLossGrad(
    const int count,
    const int num_masks,
    const float normalizer,
    const T* dy,
    const T* mask,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void BroadcastLossGrad(
    const int outer_dim,
    const int axis_dim,
    const int inner_dim,
    const T* dy,
    T* dx,
    Context* ctx);

/* loss.nll_loss */

template <typename LogitType, typename TargetType, class Context>
void NLLLoss(
    const int outer_dim,
    const int axis_dim,
    const int inner_dim,
    const int ignore_index,
    const LogitType* log_prob,
    const TargetType* target,
    LogitType* loss,
    LogitType* mask,
    Context* ctx);

template <typename LogitType, typename TargetType, class Context>
void NLLLossGrad(
    const int outer_dim,
    const int axis_dim,
    const int inner_dim,
    const int ignore_index,
    const LogitType* log_prob,
    const TargetType* target,
    LogitType* dx,
    LogitType* mask,
    Context* ctx);

/* loss.sigmoid_ce_loss */

template <typename T, class Context>
void SigmoidCrossEntropy(
    const int count,
    const T* logit,
    const T* target,
    T* loss,
    T* mask,
    Context* ctx);

template <typename T, class Context>
void SigmoidCrossEntropyGrad(
    const int count,
    const T* logit,
    const T* target,
    T* dlogit,
    T* mask,
    Context* ctx);

/* loss.sigmoid_focal_loss */

template <typename LogitType, typename TargetType, class Context>
void SigmoidFocalLoss(
    const int outer_dim,
    const int axis_dim,
    const int inner_dim,
    const float pos_alpha,
    const float neg_alpha,
    const float gamma,
    const int neg_id,
    const LogitType* logit,
    const TargetType* target,
    LogitType* loss,
    LogitType* mask,
    Context* ctx);

template <typename LogitType, typename TargetType, class Context>
void SigmoidFocalLossGrad(
    const int outer_dim,
    const int axis_dim,
    const int inner_dim,
    const float pos_alpha,
    const float neg_alpha,
    const float gamma,
    const int negative_index,
    const LogitType* logit,
    const TargetType* target,
    LogitType* dlogit,
    LogitType* mask,
    Context* ctx);

/* loss.smooth_l1_loss */

template <typename T, class Context>
void SmoothL1(
    const int count,
    const float beta,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void SmoothL1Grad(
    const int count,
    const float beta,
    const T* x,
    T* y,
    Context* ctx);

/* loss.softmax_ce_loss */

template <typename T, class Context>
void SoftmaxCrossEntropy(
    const int count,
    const T* prob,
    const T* targets,
    T* losses,
    Context* ctx);

/* loss.sparse_softmax_cross_entropy */

template <typename LogitType, typename TargetType, class Context>
void SparseSoftmaxCrossEntropy(
    const int outer_dim,
    const int axis_dim,
    const int inner_dim,
    const int ignore_index,
    const LogitType* prob,
    const TargetType* target,
    LogitType* loss,
    LogitType* mask,
    Context* ctx);

template <typename LogitType, typename TargetType, class Context>
void SparseSoftmaxCrossEntropyGrad(
    const int outer_dim,
    const int axis_dim,
    const int inner_dim,
    const int ignore_index,
    const LogitType* prob,
    const TargetType* target,
    LogitType* dx,
    LogitType* mask,
    Context* ctx);

/* math.abs */

template <typename T, class Context>
void AbsGrad(const int count, const T* x, const T* dy, T* dx, Context* ctx);

/* math.clip */

template <typename T, class Context>
void Clip(
    const int count,
    const float low,
    const float high,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void ClipGrad(
    const int count,
    const float low,
    const float high,
    const T* dy,
    const T* x,
    T* dx,
    Context* ctx);

/* math.cos */

template <typename T, class Context>
void CosGrad(const int count, const T* dy, const T* x, T* dx, Context* ctx);

/* math.moments */

template <typename Tx, typename Ty, class Context>
void Moments(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const Tx* x,
    Ty* mean,
    Ty* var,
    Context* ctx);

/* math.reciprocal */

template <typename T, class Context>
void ReciprocalGrad(
    const int count,
    const T* dy,
    const T* y,
    T* dx,
    Context* ctx);

/* math.rsqrt */

template <typename T, class Context>
void RsqrtGrad(const int count, const T* dy, const T* y, T* dx, Context* ctx);

/* math.sin */

template <typename T, class Context>
void SinGrad(const int count, const T* dy, const T* x, T* dx, Context* ctx);

/* normalization.batch_norm */

template <typename Tx, typename Tp, class Context>
void BatchNormExpectation(
    const int N,
    const int C,
    const int S,
    const Tp denorm,
    const string& data_format,
    const Tx* x,
    Tp* ex,
    Tp* ex2,
    Context* ctx);

template <typename Tx, typename Tp, class Context>
void BatchNormInternalGrad(
    const int N,
    const int C,
    const int S,
    const string& data_format,
    const Tx* x,
    const Tp* mu,
    const Tp* rsig,
    const Tp* gamma,
    const Tx* dy,
    Tp* dgamma,
    Tp* dbeta,
    Context* ctx);

template <typename Tx, typename Tp, class Context>
void BatchNormTrainingGrad(
    const int N,
    const int C,
    const int S,
    const string& data_format,
    const Tx* x,
    const Tp* mu,
    const Tp* rsig,
    const Tp* gamma,
    const Tp* dgamma,
    const Tp* dbeta,
    const Tx* dy,
    Tx* dx,
    Context* ctx);

template <typename Tx, typename Tp, class Context>
void BatchNormBackwardTraining(
    const int N,
    const int C,
    const int S,
    const string& data_format,
    const Tx* x,
    const Tp* mu,
    const Tp* rsig,
    const Tp* gamma,
    const Tx* dy,
    Tx* dx,
    Tp* dgamma,
    Tp* dbeta,
    Context* ctx);

template <typename Tx, typename Tp, class Context>
void BatchNormBackwardInference(
    const int N,
    const int C,
    const int S,
    const string& data_format,
    const Tx* x,
    const Tp* mu,
    const Tp* rsig,
    const Tp* gamma,
    const Tx* dy,
    Tx* dx,
    Tp* dgamma,
    Tp* dbeta,
    Context* ctx);

/* normalization.group_norm */

template <typename Tx, typename Tp, class Context>
void GroupNormForward(
    const int N,
    const int G,
    const int D,
    const int S,
    const string& data_format,
    const Tx* x,
    const Tp* mu,
    const Tp* rsig,
    const Tp* gamma,
    const Tp* beta,
    Tp* scale,
    Tp* bias,
    Tx* y,
    Context* ctx);

template <typename Tx, typename Tp, class Context>
void GroupNormBackward(
    const int N,
    const int G,
    const int D,
    const int S,
    const string& data_format,
    const Tx* x,
    const Tp* mu,
    const Tp* rsig,
    const Tp* gamma,
    const Tx* dy,
    Tp* ds,
    Tp* db,
    Tx* dx,
    Tp* dgamma,
    Tp* dbeta,
    Context* ctx);

/* normalization.lp_norm */

template <typename T, class Context>
void L1Normalize(
    const int outer_dim,
    const int reduce_dim,
    const int inner_dim,
    const float scale,
    const float eps,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void L1NormalizeGrad(
    const int outer_dim,
    const int reduce_dim,
    const int inner_dim,
    const float scale,
    const float eps,
    const T* dy,
    const T* x,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void L2Normalize(
    const int outer_dim,
    const int reduce_dim,
    const int inner_dim,
    const float scale,
    const float eps,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void L2NormalizeGrad(
    const int outer_dim,
    const int reduce_dim,
    const int inner_dim,
    const float scale,
    const float eps,
    const T* dy,
    const T* x,
    T* dx,
    Context* ctx);

/* recurrent.lstm_cell */

template <typename T, class Context>
void LSTMCell(
    const int N,
    const int C,
    const T* cx,
    T* actx,
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

/* training.adam_update */

template <typename T, class Context>
void AdamUpdate(
    const int count,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    T* g,
    T* m,
    T* v,
    Context* ctx);

/* training.nesterov_update */

template <typename T, class Context>
void NesterovUpdate(
    const int count,
    const float lr,
    const float momentum,
    T* g,
    T* m,
    Context* ctx);

/* training.rmsprop_update */

template <typename T, class Context>
void RMSPropUpdate(
    const int count,
    const float lr,
    const float momentum,
    const float decay,
    const float eps,
    T* g,
    T* m,
    T* v,
    Context* ctx);

/* training.sgd_update */

template <typename T, class Context>
void SGDUpdate(
    const int count,
    const float lr,
    const float momentum,
    T* g,
    T* m,
    Context* ctx);

/* vision.bias_add */

template <typename T, class Context>
void BiasAdd(
    const int outer_dim,
    const int axis_dim,
    const int inner_dim,
    const T* x,
    const T* b,
    T* y,
    Context* ctx);

/* vision.conv */

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

/* vision.depthwise_conv */

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
    const T* w,
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
    const T* d,
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
    T* dw,
    Context* ctx);

/* vision.resize */

template <typename T, class Context>
void ResizeNearest(
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
void ResizeNearestGrad(
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
void ResizeLinear(
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
void ResizeLinearGrad(
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

/* vision.pooling */

template <typename T, class Context>
void MaxPool2d(
    const int N,
    const int C,
    const int H,
    const int W,
    const int pool_h,
    const int pool_w,
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
void AvgPool2d(
    const int N,
    const int C,
    const int H,
    const int W,
    const int pool_h,
    const int pool_w,
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
void MaxPool2dGrad(
    const int N,
    const int C,
    const int H,
    const int W,
    const int pool_h,
    const int pool_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const string& data_format,
    const T* dy,
    const int* mask,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void AvgPool2dGrad(
    const int N,
    const int C,
    const int H,
    const int W,
    const int pool_h,
    const int pool_w,
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

/* vision.roi_align */

template <typename T, class Context>
void RoiAlign(
    const int C,
    const int H,
    const int W,
    const int pool_h,
    const int pool_w,
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
    const int pool_h,
    const int pool_w,
    const int num_rois,
    const float spatial_scale,
    const int sampling_ratio,
    const T* dy,
    const float* rois,
    float* dx,
    Context* ctx);

/* vision.roi_pool */

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
    const int* mask,
    float* dx,
    Context* ctx);

} // namespace kernel

} // namespace dragon

#endif // DRAGON_UTILS_OP_KERNELS_H_
