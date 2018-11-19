// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

#ifndef DRAGON_UTILS_OP_KERNEL_H_
#define DRAGON_UTILS_OP_KERNEL_H_

#include "core/context.h"

namespace dragon {

namespace kernel {

typedef int64_t TIndex;

/******************** activation.dropout ********************/

template <typename T, class Context>
void Dropout(
    const int               count,
    float                   prob,
    float                   scale,
    const T*                x,
    uint32_t*               mask32,
    uint8_t*                mask8,
    T*                      y,
    Context*                ctx);

template <typename Tx, typename Tm, class Context>
void ApplyMask(
    const int               count,
    const float             scale,
    const Tx*               x,
    const Tm*               mask,
    Tx*                     y,
    Context*                ctx);

/******************** activation.elu ********************/

template <typename T, class Context>
void Elu(
    const int               count,
    const float             alpha,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void EluGrad(
    const int               count,
    const float             alpha,
    const T*                dy,
    const T*                y,
    T*                      dx,
    Context*                ctx);

/******************** activation.prelu ********************/

template <typename T, class Context>
void PRelu(
    const int               count,
    const int               channels,
    const int               dim,
    const bool              channel_shared,
    const string&           data_format,
    const T*                x,
    const T*                w,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void PReluGrad(
    const int               count,
    const int               channels,
    const int               dim,
    const bool              channel_shared,
    const string&           data_format,
    const T*                dy,
    const T*                x,
    const T*                w,
    T*                      dx,
    Context*                ctx);

template <typename T, class Context>
void PReluWGrad(
    const int               rows,
    const int               row_offset,
    const int               channels,
    const int               dim,
    const bool              channel_shared,
    const string&           data_format,
    const T*                dy,
    const T*                x,
    const T*                multiplier,
    T*                      bcast_dw,
    T*                      dw,
    Context*                ctx);

/******************** activation.relu ********************/

template <typename T, class Context>
void Relu(
    const int               count,
    const float             slope,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void ReluGrad(
    const int               count,
    const float             slope,
    const T*                dy,
    const T*                y,
    T*                      dx,
    Context*                ctx);

/******************** activation.selu ********************/

template <typename T, class Context>
void SElu(
    const int               count,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void SEluGrad(
    const int               count,
    const T*                dy,
    const T*                y,
    T*                      dx,
    Context*                ctx);

/******************** activation.sigmoid ********************/

template <typename T, class Context>
void Sigmoid(
    const int               count,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void SigmoidGrad(
    const int               count,
    const T*                dy,
    const T*                y,
    T*                      dx,
    Context*                ctx);

/******************** activation.softmax ********************/

template <typename T, class Context>
void Softmax(
    const int               count,
    const int               classes,
    const int               outer_dim,
    const int               inner_dim,
    const T*                sum_multiplier,
    const T*                x,
    T*                      scale,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void SoftmaxGrad(
    const int               count,
    const int               classes,
    const int               outer_dim,
    const int               inner_dim,
    const T*                sum_multiplier,
    const T*                dy,
    const T*                y,
    T*                      scale,
    T*                      dx,
    Context*                ctx);

/******************** activation.tanh ********************/

template <typename T, class Context>
void Tanh(
    const int               count,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void TanhGrad(
    const int               count,
    const T*                dy,
    const T*                y,
    T*                      dx,
    Context*                ctx);

/******************** arithmetic.affine ********************/

template <typename T, class Context>
void Affine(
    const int               count,
    const int               outer_dim,
    const int               scale_dim,
    const int               inner_dim,
    const T*                x,
    const T*                alpha,
    const T*                beta,
    const T*                beta_multiplier,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void AffineGrad(
    const int               count,
    const int               outer_dim,
    const int               scale_dim,
    const int               inner_dim,
    const T*                dy,
    const T*                alpha,
    T*                      dx,
    Context*                ctx);

/******************** arithmetic.clip ********************/

template <typename T, class Context>
void Clip(
    const int               count,
    const float             low,
    const float             high,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void ClipGrad(
    const int               count,
    const float             low,
    const float             high,
    const T*                x,
    const T*                dy,
    T*                      dx,
    Context*                ctx);

/******************** arithmetic.maximum ********************/

template <typename T, class Context>
void MaximumE(
    const int               count,
    const T*                x1,
    const T*                x2,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void MaximumB(
    const int               count,
    const T*                x1,
    const T                 x2,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void MaximumEGrad(
    const int               count,
    const T*                x1,
    const T*                x2,
    const T*                dy,
    T*                      dx1,
    T*                      dx2,
    Context*                ctx);

template <typename T, class Context>
void MaximumBGrad(
    const int               count,
    const T*                x1,
    const T                 x2,
    const T*                dy,
    T*                      dx1,
 /* T*                      dx2, */
    Context*                ctx);

/******************** arithmetic.minimum ********************/

template <typename T, class Context>
void MinimumE(
    const int               count,
    const T*                x1,
    const T*                x2,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void MinimumB(
    const int               count,
    const T*                x1,
    const T                 x2,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void MinimumEGrad(
    const int               count,
    const T*                x1,
    const T*                x2,
    const T*                dy,
    T*                      dx1,
    T*                      dx2,
    Context*                ctx);

template <typename T, class Context>
void MinimumBGrad(
    const int               count,
    const T*                x1,
    const T                 x2,
    const T*                dy,
    T*                      dx1,
 /* T*                      dx2, */
    Context*                ctx);

/******************** control_flow.compare ********************/

template <typename T, class Context>
void Equal(
    const int               count,
    const T*                a,
    const T*                b,
    T*                      y,
    Context*                ctx);

/******************** loss.l1_loss ********************/

template <typename T, class Context>
void AbsGrad(
    const int               count,
    const T*                dy,
    T*                      dx,
    Context*                ctx);

/******************** loss.nll_loss ********************/

template <typename Tx, typename Ty, class Context>
void NLLLoss(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const Tx*               log_prob,
    const Ty*               labels,
    const int*              ignores,
    const int               num_ignores,
    float*                  losses,
    float*                  flags,
    Context*                ctx);

template <typename Tx, typename Ty, class Context>
void NLLLossGrad(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const Tx*               prob,
    const Ty*               labels,
    const int*              ignores,
    const int               num_ignores,
    Tx*                     dx,
    float*                  flags,
    Context*                ctx);

/******************** loss.sigmoid_cross_entropy ********************/

template <typename T, class Context>
void SigmoidCrossEntropy(
    const int               count,
    const T*                logits,
    const T*                targets,
    T*                      losses,
    T*                      flags,
    Context*                ctx);

template <typename T, class Context>
void SigmoidCrossEntropyGrad(
    const int               count,
    const T*                logits,
    const T*                targets,
    T*                      dlogits,
    T*                      flags,
    Context*                ctx);

/******************** loss.sigmoid_focal_loss ********************/

template <typename T, class Context>
void SigmoidFocalLoss(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const float*            logits,
    const float*            targets,
    float*                  losses,
    float*                  flags,
    Context*                ctx);

template <typename T, class Context>
void SigmoidFocalLossGradient(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const float*            logits,
    const float*            targets,
    float*                  dlogits,
    float*                  flags,
    Context*                ctx);

/******************** loss.smooth_l1_loss ********************/

template <typename T, class Context>
void SmoothL1(
    const int               count,
    const float             beta,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void SmoothL1Grad(
    const int               count,
    const float             beta,
    const T*                dy,
    T*                      dx,
    Context*                ctx);

/******************** loss.softmax_cross_entropy ********************/

template <typename T, class Context>
void SoftmaxCrossEntropy(
    const int               count,
    const T*                prob,
    const T*                target,
    T*                      loss,
    Context*                ctx);

/******************** loss.softmax_focal_loss ********************/

template <typename T, class Context>
void SoftmaxFocalLoss(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const T*                prob,
    const T*                labels,
    const int*              ignores,
    const int               num_ignores,
    T*                      losses,
    T*                      flags,
    Context*                ctx);

template <typename T, class Context>
void SoftmaxFocalLossGrad(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const T*                prob,
    const T*                labels,
    const int*              ignores,
    const int               num_ignores,
    T*                      dx,
    T*                      flags,
    Context*                ctx);

/******************** loss.sparse_softmax_cross_entropy ********************/

template <typename Tx, typename Ty, class Context>
void SparseSoftmaxCrossEntropy(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const Tx*               prob,
    const Ty*               labels,
    const int*              ignores,
    const int               num_ignores,
    float*                  losses,
    float*                  flags,
    Context*                ctx);

template <typename Tx, typename Ty, class Context>
void SparseSoftmaxCrossEntropyGrad(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const Tx*               prob,
    const Ty*               labels,
    const int*              ignores,
    const int               num_ignores,
    Tx*                     dx,
    float*                  flags,
    Context*                ctx);

/******************** misc.astype ********************/

template <typename Ta, typename Tb, class Context>
void TypeA2B(
    const int               count,
    const Ta*               a,
    Tb*                     b,
    Context*                ctx);

/******************** misc.image_data ********************/

template <typename Tx, typename Ty, class Context>
void ImageData(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean_values,
    const float*            std_values,
    const string&           data_format,
    const Tx*               x,
    Ty*                     y,
    Context*                ctx);

/******************** ndarray.arange ********************/

template <typename T, class Context>
void Arange(
    const int               count,
    const int               start,
    const int               step,
    T*                      y,
    Context*                ctx);

/******************** ndarray.argreduce ********************/

template <typename T, class Context>
void Argmax(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const int               top_k,
    const T*                x,
    int64_t*                indices,
    T*                      values,
    Context*                ctx);

template <typename T, class Context>
void Argmin(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const int               top_k,
    const T*                x,
    int64_t*                indices,
    T*                      values,
    Context*                ctx);

/******************** ndarray.gather ********************/

template <typename T, class Context>
void CanonicalAxis(
    const int               count,
    const int               dim,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void Gather(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int*              indices,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void GatherGrad(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int*              indices,
    const T*                dy,
    T*                      dx,
    Context*                ctx);

/******************** ndarray.concat ********************/

template <typename T, class Context>
void Concat(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_concat_dim,
    const int               y_concat_dim,
    const int               concat_offset,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void ConcatGrad(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_concat_dim,
    const int               y_concat_dim,
    const int               concat_offset,
    const T*                dy,
    T*                      dx,
    Context*                ctx);

/******************** ndarray.crop ********************/

template <typename T, class Context>
void Crop1D(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               start,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void Crop1DGrad(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               start,
    const int               end,
    const T*                dy,
    T*                      dx,
    Context*                ctx);

/******************** ndarray.pad ********************/

template <typename T, class Context>
void ConstPad1D(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const float             value,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void ReflectPad1D(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void EdgePad1D(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void ConstPad1DGrad(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const T*                dy,
    T*                      dx,
    Context*                ctx);

template <typename T, class Context>
void ReflectPad1DGrad(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const T*                dy,
    T*                      dx,
    Context*                ctx);

template <typename T, class Context>
void EdgePad1DGrad(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const T*                dy,
    T*                      dx,
    Context*                ctx);

/******************** ndarray.one_hot ********************/

template <typename T, class Context>
void OneHot(
    const int               count,
    const int               depth,
    const int               on_value,
    const T*                x,
    T*                      y,
    Context*                ctx);

/******************** ndarray.reduce ********************/

template <typename T, class Context>
void Sum(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void SumGrad(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const T                 coeff,
    const T*                dy,
    T*                      dx,
    Context*                ctx);

/******************** ndarray.repeat ********************/

template <typename T, class Context>
void Repeat(
    const int               count,
    const int               outer_dim,
    const int               dim,
    const int               inner_dim,
    const int               repeats,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void RepeatGrad(
    const int               count,
    const int               outer_dim,
    const int               dim,
    const int               inner_dim,
    const int               repeats,
    const T*                dy,
    T*                      dx,
    Context*                ctx);

/******************** ndarray.slice ********************/

template <typename T, class Context>
void Slice(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int               slice_offset,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void SliceGrad(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int               slice_offset,
    const T*                dy,
    T*                      x,
    Context*                ctx);

/******************** ndarray.tile ********************/

template <typename T, class Context>
void Tile(
    const int               count,
    const int               outer_dim,
    const int               ex_inner_dim,
    const int               multiple,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void TileGrad(
    const int               count,
    const int               outer_dim,
    const int               ex_inner_dim,
    const int               multiple,
    const T*                dy,
    T*                      dx,
    Context*                ctx);

/******************** ndarray.transpose ********************/

template <typename T, class Context>
void Transpose(
    const int               count,
    const int               ndim,
    const int*              order,
    const int*              old_steps,
    const int*              new_steps,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void TransposeGrad(
    const int               count,
    const int               ndim,
    const int*              order,
    const int*              old_steps,
    const int*              new_steps,
    const T*                dy,
    T*                      dx,
    Context*                ctx);

/******************** recurrent.lstm_cell ********************/

template <typename T, class Context>
void LSTMCell(
    const int               count,
    const int               N,
    const int               C,
    const T*                cx,
    T*                      xact,
    T*                      c,
    T*                      h,
    Context*                ctx);

template <typename T, class Context>
void LSTMCellGrad(
    const int               count,
    const int               N,
    const int               C,
    const T*                cx,
    const T*                xact,
    const T*                c,
    const T*                dc,
    const T*                dh,
    T*                      dcx,
    T*                      dx,
    Context*                ctx);

/******************** update.adam_update ********************/

template <typename T, class Context>
void AdamUpdate(
    const int               count,
    const float             lr,
    const float             beta1,
    const float             beta2,
    const float             eps,
    T*                      g,
    T*                      m,
    T*                      v,
    Context*                ctx);

/******************** update.nesterov_update ********************/

template <typename T, class Context>
void NesterovUpdate(
    const int               count,
    const float             lr,
    const float             momentum,
    T*                      g,
    T*                      h,
    Context*                ctx);

/******************** update.rmsprop_update ********************/

template <typename T, class Context>
void RMSPropUpdate(
    const int               count,
    const float             lr,
    const float             decay,
    const float             eps,
    T*                      g,
    T*                      h,
    Context*                ctx);

/******************** update.sgd_update ********************/

template <typename T, class Context>
void SGDUpdate(
    const int               count,
    const float             lr,
    const float             momentum,
    T*                      g,
    T*                      h,
    Context*                ctx);

/******************** vision.bias_add ********************/

template <typename T, class Context>
void BiasAdd(
    const int               count,
    const int               outer_dim,
    const int               dim,
    const int               inner_dim,
    const string&           data_format,
    const T*                bias,
    const T*                bias_multiplier,
    T*                      y,
    Context*                ctx);

/******************** vision.bilinear_resize ********************/

template <typename T, class Context>
void BilinearResize(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const string&           data_format,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void BilinearResizeGrad(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const string&           data_format,
    const T*                dy,
    T*                      dx,
    Context*                ctx);

/******************** vision.conv ********************/

template <typename T, class Context>
void Im2Col2d(
    const int               C,
    const int               H,
    const int               W,
    const int               col_h,
    const int               col_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const string&           data_format,
    const T*                im,
    T*                      col,
    Context*                ctx);

template <typename T, class Context>
void Col2Im2d(
    const int               C,
    const int               H,
    const int               W,
    const int               col_h,
    const int               col_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const string&           data_format,
    const T*                col,
    T*                      im,
    Context*                ctx);

/******************** vision.drop_block ********************/

template <class Context>
void DropBlock2d(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               seed_h,
    const int               seed_w,
    const int               block_size,
    const float             gamma,
    const string&           data_format,
    uint32_t*               seed,
    int*                    mask,
    Context*                ctx);

/******************** vision.nn_resize ********************/

template <typename T, class Context>
void NNResize(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const string&           data_format,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void NNResizeGrad(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const string&           data_format,
    const T*                dy,
    T*                      dx,
    Context*                ctx);

/******************** vision.pooling ********************/

template <typename T, class Context>
void MAXPooling2d(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const string&           data_format,
    const T*                x,
    int*                    mask,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void AVGPooling2d(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const string&           data_format,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void MAXPooling2dGrad(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const string&           data_format,
    const T*                dy,
    const int*              mask,
    T*                      dx,
    Context*                ctx);

template <typename T, class Context>
void AVGPooling2dGrad(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const string&           data_format,
    const T*                dy,
    T*                      dx,
    Context*                ctx);

/******************** vision.roi_pooling ********************/

template <typename T, class Context>
void ROIPooling(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const T*                x,
    const T*                rois,
    int*                    mask,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void ROIPoolingGrad(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const T*                dy,
    const T*                rois,
    const int*              mask,
    T*                      dx,
    Context*                ctx);

/******************** vision.roi_align ********************/

template <typename T, class Context>
void ROIAlign(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const int               sampling_ratio,
    const T*                x,
    const T*                rois,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void ROIAlignGrad(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const int               sampling_ratio,
    const float*            dy,
    const float*            rois,
    float*                  dx,
    Context*                ctx);

}    // namespace kernel

}    // namepsace dragon

#endif    // DRAGON_UTILS_OP_KERNEL_H_