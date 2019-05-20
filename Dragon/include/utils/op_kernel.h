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

#ifndef DRAGON_UTILS_OP_KERNEL_H_
#define DRAGON_UTILS_OP_KERNEL_H_

#include "core/context.h"

namespace dragon {

class Tensor;

namespace kernel {

/*! activation.dropout */

template <typename T, class Context>
void Dropout(
    const int               count,
    const float             prob,
    const float             scale,
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

/*! activation.droppath */

template <typename T, class Context>
void DropPath(
    const int               rows,
    const int               cols,
    const float             scale,
    const T*                x,
    const float*            mask,
    T*                      y,
    Context*                ctx);

/*! activation.elu */

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

/*! activation.prelu */

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

/*! activation.relu */

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

/*! activation.selu */

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

/*! activation.sigmoid */

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

/*! activation.softmax */

template <typename T, class Context>
void Softmax(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const T*                multiplier,
    const T*                x,
    T*                      scale,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void SoftmaxGrad(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const T*                multiplier,
    const T*                dy,
    const T*                y,
    T*                      scale,
    T*                      dx,
    Context*                ctx);

/*! activation.tanh */

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

/*! arithmetic.affine */

template <typename T, class Context>
void Affine(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const T*                x,
    const T*                alpha,
    const T*                beta,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void AffineGrad(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const T*                dy,
    const T*                alpha,
    T*                      dx,
    Context*                ctx);

/*! arithmetic.clip */

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

/*! arithmetic.maximum */

template <typename T, class Context>
void Maximum(
    const int               count,
    const T*                a,
    const T*                b,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void BroadcastMaximum(
    const int               count,
    const T*                a,
    const T                 b,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void MaximumGrad(
    const int               count,
    const T*                a,
    const T*                b,
    const T*                dy,
    T*                      da,
    T*                      db,
    Context*                ctx);

template <typename T, class Context>
void BroadcastMaximumGrad(
    const int               count,
    const T*                a,
    const T                 b,
    const T*                dy,
    T*                      da,
    T*                      db,
    Context*                ctx);

/*! arithmetic.minimum */

template <typename T, class Context>
void Minimum(
    const int               count,
    const T*                a,
    const T*                b,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void BroadcastMinimum(
    const int               count,
    const T*                a,
    const T                 b,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void MinimumGrad(
    const int               count,
    const T*                a,
    const T*                b,
    const T*                dy,
    T*                      da,
    T*                      db,
    Context*                ctx);

template <typename T, class Context>
void BroadcastMinimumGrad(
    const int               count,
    const T*                a,
    const T                 b,
    const T*                dy,
    T*                      da,
    T*                      db,
    Context*                ctx);

/*! arithmetic.moments */

template <typename Tx, typename Ty, class Context>
void Moments(
    const int               ndims,
    const int*              dims,
    const int               naxes,
    const int*              axes,
    const Tx*               x,
    Ty*                     mean,
    Ty*                     var,
    Context*                ctx);

/*! array.arange */

template <typename T, class Context>
void Arange(
    const int               count,
    const int               start,
    const int               step,
    T*                      y,
    Context*                ctx);

/*! array.argreduce */

template <typename T, class Context>
void ArgMax(
    const int               outer_dim,
    const int               inner_dim,
    const int               axis_dim,
    const int               top_k,
    const T*                x,
    int64_t*                indices,
    T*                      values,
    Context*                ctx);

template <typename T, class Context>
void ArgMin(
    const int               outer_dim,
    const int               inner_dim,
    const int               axis_dim,
    const int               top_k,
    const T*                x,
    int64_t*                indices,
    T*                      values,
    Context*                ctx);

/*! array.concat */

template <typename T, class Context>
void Concat(
    const int               outer_dim,
    const int               inner_dim,
    const int               axis_dim,
    const int               cat_dim,
    const int               cat_ofs,
    const T*                x,
    T*                      y,
    Context*                ctx);

/*! array.crop */

template <typename T, class Context>
void Crop(
    const int               count,
    const int               ndims,
    const int*              x_strides,
    const int*              y_dims,
    const int*              starts,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void CropGrad(
    const int               count,
    const int               ndims,
    const int*              x_strides,
    const int*              y_dims,
    const int*              starts,
    const T*                dy,
    T*                      dx,
    Context*                ctx);

/*! array.index_select */

template <typename T, class Context>
void IndexSelect(
    const int               outer_dim,
    const int               inner_dim,
    const int               axis_dim,
    const int               num_indices,
    const int64_t*          indices,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void IndexSelectGrad(
    const int               outer_dim,
    const int               inner_dim,
    const int               axis_dim,
    const int               num_indices,
    const int64_t*          indices,
    const T*                dy,
    T*                      dx,
    Context*                ctx);

/*! array.masked_select */

template <typename T, class Context>
void MaskedSelect(
    const int               count,
    const uint8_t*          mask,
    const T*                x,
    Tensor*                 indices,
    Tensor*                 scratch,
    Tensor*                 y,
    Context*                ctx);

template <typename T, class Context>
void MaskedSelectGrad(
    const int               count,
    const int               num_indices,
    const int64_t*          indices,
    const T*                dy,
    T*                      dx,
    Context*                ctx);

/*! array.non_zero */

template <class Context>
void UnravelIndex(
    const int               count,
    const int               ndims,
    const int*              dims,
    const int64_t*          x,
    int64_t*                y,
    Context*                ctx);

/*! array.pad */

template <typename T, class Context>
void ConstPad(
    const int               count,
    const int               ndims,
    const int*              x_dims,
    const int*              x_strides,
    const int*              y_dims,
    const int*              l_pads,
    const float             value,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void ReflectPad(
    const int               count,
    const int               ndims,
    const int*              x_dims,
    const int*              x_strides,
    const int*              y_dims,
    const int*              l_pads,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void EdgePad(
    const int               count,
    const int               ndims,
    const int*              x_dims,
    const int*              x_strides,
    const int*              y_dims,
    const int*              l_pads,
    const T*                x,
    T*                      y,
    Context*                ctx);

/*! array.one_hot */

template <typename T, class Context>
void OneHot(
    const int               count,
    const int               depth,
    const int               on_value,
    const T*                x,
    T*                      y,
    Context*                ctx);

/*! array.reduce */

template <typename T, class Context>
void ReduceSum(
    const int               ndims,
    const int*              dims,
    const int               naxes,
    const int*              axes,
    const float             scale,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void ReduceSumGrad(
    const int               count,
    const int               ndims,
    const int*              x_dims,
    const int*              y_dims,
    const int*              y_strides,
    const float             scale,
    const T*                dy,
    T*                      dx,
    Context*                ctx);

/*! array.repeat */

template <typename T, class Context>
void Repeat(
    const int               outer_dim,
    const int               inner_dim,
    const int               axis_dim,
    const int               repeats,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void RepeatGrad(
    const int               outer_dim,
    const int               inner_dim,
    const int               axis_dim,
    const int               repeats,
    const T*                dy,
    T*                      dx,
    Context*                ctx);

/*! array.slice */

template <typename T, class Context>
void Slice(
    const int               outer_dim,
    const int               inner_dim,
    const int               axis_dim,
    const int               slice_dim,
    const int               slice_ofs,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void SliceGrad(
    const int               outer_dim,
    const int               inner_dim,
    const int               axis_dim,
    const int               slice_dim,
    const int               slice_ofs,
    const T*                dy,
    T*                      x,
    Context*                ctx);

/*! array.tile */

template <typename T, class Context>
void Tile(
    const int               count,
    const int               ndims,
    const int*              x_dims,
    const int*              x_strides,
    const int*              y_dims,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void TileGrad(
    const int               rows,
    const int               cols,
    const int               multiple,
    const T*                dy,
    T*                      dx,
    Context*                ctx);

/*! array.transpose */

template <typename T, class Context>
void Transpose(
    const int               count,
    const int               ndims,
    const int*              x_strides,
    const int*              y_dims,
    const T*                x,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void TransposeGrad(
    const int               count,
    const int               ndims,
    const int*              x_strides,
    const int*              y_dims,
    const T*                dy,
    T*                      dx,
    Context*                ctx);

/*! array.where */

template <typename T, class Context>
void Where(
    const int               count,
    const uint8_t*          mask,
    const T*                a,
    const T*                b,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void WhereGrad(
    const int               count,
    const uint8_t*          mask,
    const T*                dy,
    T*                      da,
    T*                      db,
    Context*                ctx);

/*! control_flow.assgin */

template <typename T, class Context>
void Assign(
    const int               count,
    const int               ndims,
    const int*              x_dims,
    const int*              y_strides,
    const int*              starts,
    const T*                x,
    T*                      y,
    Context*                ctx);

/*! control_flow.compare */

template <typename T, class Context>
void NotZero(
    const int               count,
    const T*                x,
    bool*                   y,
    Context*                ctx);

template <typename T, class Context>
void Equal(
    const int               count,
    const T*                a,
    const T*                b,
    bool*                   y,
    Context*                ctx);

template <typename T, class Context>
void NotEqual(
    const int               count,
    const T*                a,
    const T*                b,
    bool*                   y,
    Context*                ctx);

template <typename T, class Context>
void Less(
    const int               count,
    const T*                a,
    const T*                b,
    bool*                   y,
    Context*                ctx);

template <typename T, class Context>
void LessEqual(
    const int               count,
    const T*                a,
    const T*                b,
    bool*                   y,
    Context*                ctx);

template <typename T, class Context>
void Greater(
    const int               count,
    const T*                a,
    const T*                b,
    bool*                   y,
    Context*                ctx);

template <typename T, class Context>
void GreaterEqual(
    const int               count,
    const T*                a,
    const T*                b,
    bool*                   y,
    Context*                ctx);

/*! loss.l1_loss */

template <typename T, class Context>
void AbsGrad(
    const int               count,
    const T*                dy,
    T*                      dx,
    Context*                ctx);

/*! loss.nll_loss */

template <typename Tx, typename Ty, class Context>
void NLLLoss(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               nignores,
    const int*              ignore,
    const Tx*               log_prob,
    const Ty*               target,
    Tx*                     loss,
    int*                    flag,
    Context*                ctx);

template <typename Tx, typename Ty, class Context>
void NLLLossGrad(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               nignores,
    const int*              ignore,
    const Tx*               log_prob,
    const Ty*               target,
    Tx*                     dx,
    int*                    flag,
    Context*                ctx);

/*! loss.sigmoid_ce_loss */

template <typename T, class Context>
void SigmoidCrossEntropy(
    const int               count,
    const T*                logit,
    const T*                target,
    T*                      loss,
    int*                    flag,
    Context*                ctx);

template <typename T, class Context>
void SigmoidCrossEntropyGrad(
    const int               count,
    const T*                logit,
    const T*                target,
    T*                      dlogit,
    int*                    flag,
    Context*                ctx);

/*! loss.sigmoid_focal_loss */

template <typename Tx, typename Ty, class Context>
void SigmoidFocalLoss(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const Tx*               logit,
    const Ty*               target,
    Tx*                     loss,
    int*                    flag,
    Context*                ctx);

template <typename Tx, typename Ty, class Context>
void SigmoidFocalLossGrad(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const Tx*               logit,
    const Ty*               target,
    Tx*                     dlogit,
    int*                    flag,
    Context*                ctx);

/*! loss.smooth_l1_loss */

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

/*! loss.softmax_ce_loss */

template <typename T, class Context>
void SoftmaxCrossEntropy(
    const int               count,
    const T*                prob,
    const T*                targets,
    T*                      losses,
    Context*                ctx);

/*! loss.softmax_focal_loss */

template <typename Tx, typename Ty, class Context>
void SoftmaxFocalLoss(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const int               nignores,
    const int*              ignores,
    const Tx*               prob,
    const Ty*               labels,
    Tx*                     losses,
    int*                    flags,
    Context*                ctx);

template <typename Tx, typename Ty, class Context>
void SoftmaxFocalLossGrad(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const int               nignores,
    const int*              ignores,
    const Tx*               prob,
    const Ty*               labels,
    Tx*                     dx,
    int*                    flags,
    Context*                ctx);

/*! loss.sparse_softmax_cross_entropy */

template <typename Tx, typename Ty, class Context>
void SparseSoftmaxCrossEntropy(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               nignores,
    const int*              ignore,
    const Tx*               prob,
    const Ty*               target,
    Tx*                     loss,
    int*                    flag,
    Context*                ctx);

template <typename Tx, typename Ty, class Context>
void SparseSoftmaxCrossEntropyGrad(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const int               nignores,
    const int*              ignore,
    const Tx*               prob,
    const Ty*               target,
    Tx*                     dx,
    int*                    flag,
    Context*                ctx);

/*! misc.astype */

template <typename Ta, typename Tb, class Context>
void TypeA2B(
    const int               count,
    const Ta*               a,
    Tb*                     b,
    Context*                ctx);

/*! misc.gradient */

template <typename T, class Context>
void GradientTwoSum(
    const int               count,
    const T*                dy1,
    const T*                dy2,
    T*                      dx,
    Context*                ctx);

/*! misc.image_data */

template <typename Tx, typename Ty, class Context>
void ImageData(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const string&           data_format,
    const float*            mean,
    const float*            std,
    const Tx*               x,
    Ty*                     y,
    Context*                ctx);

/*! norm.batch_norm */

template <typename Tx, typename Tp, class Context>
void BatchNormBackwardTraining(
    const int               N,
    const int               C,
    const int               S,
    const string&           data_format,
    const Tx*               x,
    const Tp*               mu,
    const Tp*               rsig,
    const Tp*               gamma,
    const Tx*               dy,
    Tp*                     ds,
    Tp*                     db,
    Tx*                     dx,
    Tp*                     dgamma,
    Tp*                     dbeta,
    Context*                ctx);

template <typename Tx, typename Tp, class Context>
void BatchNormBackwardInference(
    const int               N,
    const int               C,
    const int               S,
    const string&           data_format,
    const Tx*               x,
    const Tp*               mu,
    const Tp*               rsig,
    const Tp*               gamma,
    const Tx*               dy,
    Tx*                     dx,
    Tp*                     dgamma,
    Tp*                     dbeta,
    Context*                ctx);

/*! norm.group_norm */

template <typename Tx, typename Tp, class Context>
void GroupNormForward(
    const int               N,
    const int               G,
    const int               D,
    const int               S,
    const string&           data_format,
    const Tx*               x,
    const Tp*               mu,
    const Tp*               rsig,
    const Tp*               gamma,
    const Tp*               beta,
    Tp*                     scale,
    Tp*                     bias,
    Tx*                     y,
    Context*                ctx);

template <typename Tx, typename Tp, class Context>
void GroupNormBackward(
    const int               N,
    const int               G,
    const int               D,
    const int               S,
    const string&           data_format,
    const Tx*               x,
    const Tp*               mu,
    const Tp*               rsig,
    const Tp*               gamma,
    const Tx*               dy,
    Tp*                     ds,
    Tp*                     db,
    Tx*                     dx,
    Tp*                     dgamma,
    Tp*                     dbeta,
    Context*                ctx);

/*! recurrent.lstm_cell */

template <typename T, class Context>
void LSTMCell(
    const int               N,
    const int               C,
    const T*                cx,
    T*                      actx,
    T*                      c,
    T*                      h,
    Context*                ctx);

template <typename T, class Context>
void LSTMCellGrad(
    const int               N,
    const int               C,
    const T*                cx,
    const T*                actx,
    const T*                c,
    const T*                dc,
    const T*                dh,
    T*                      dcx,
    T*                      dx,
    Context*                ctx);

/*! update.adam_update */

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

/*! update.nesterov_update */

template <typename T, class Context>
void NesterovUpdate(
    const int               count,
    const float             lr,
    const float             momentum,
    T*                      g,
    T*                      h,
    Context*                ctx);

/*! update.rmsprop_update */

template <typename T, class Context>
void RMSPropUpdate(
    const int               count,
    const float             lr,
    const float             decay,
    const float             eps,
    T*                      g,
    T*                      h,
    Context*                ctx);

/*! update.sgd_update */

template <typename T, class Context>
void SGDUpdate(
    const int               count,
    const float             lr,
    const float             momentum,
    T*                      g,
    T*                      h,
    Context*                ctx);

/*! update.op_base */

template <typename T, class Context>
void MixedPrecL2Decay(
    const int               count,
    const float             alpha,
    const T*                w,
    float*                  dx,
    Context*                ctx);

template <typename T, class Context>
void MixedPrecUpdate(
    const int               count,
    const float*            updates,
    T*                      w,
    Context*                ctx);

/*! vision.bias_add */

template <typename T, class Context>
void BiasAdd(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const string&           data_format,
    const T*                bias,
    const T*                multiplier,
    T*                      y,
    Context*                ctx);

/*! vision.bilinear_resize */

template <typename T, class Context>
void BilinearResize(
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

/*! vision.conv */

template <typename T, class Context>
void Im2Col2d(
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
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
    const int               out_h,
    const int               out_w,
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

/*! vision.depthwise_conv */

template <typename T, class Context>
void DepthwiseConv2d(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const string&           data_format,
    const T*                x,
    const T*                w,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void DepthwiseConv2dGrad(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const string&           data_format,
    const T*                dy,
    const T*                d,
    T*                      dx,
    Context*                ctx);

template <typename T, class Context>
void DepthwiseConv2dWGrad(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const string&           data_format,
    const T*                dy,
    const T*                x,
    T*                      dw,
    Context*                ctx);

/*! vision.drop_block */

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

/*! vision.nn_resize */

template <typename T, class Context>
void NNResize(
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

/*! vision.pooling */

template <typename T, class Context>
void MaxPool2d(
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
void AvgPool2d(
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
void MaxPool2dGrad(
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
void AvgPool2dGrad(
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

/*! vision.roi_pooling */

template <typename T, class Context>
void ROIPool(
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const T*                x,
    const float*            rois,
    int*                    mask,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void ROIPoolGrad(
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

/*! vision.roi_align */

template <typename T, class Context>
void ROIAlign(
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const int               sampling_ratio,
    const T*                x,
    const float*            rois,
    T*                      y,
    Context*                ctx);

template <typename T, class Context>
void ROIAlignGrad(
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const int               sampling_ratio,
    const T*                dy,
    const float*            rois,
    T*                      dx,
    Context*                ctx);

}  // namespace kernel

}    // namepsace dragon

#endif  // DRAGON_UTILS_OP_KERNEL_H_