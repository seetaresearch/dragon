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

#ifndef DRAGON_OPERATORS_VISION_CONV_OP_IMPL_H_
#define DRAGON_OPERATORS_VISION_CONV_OP_IMPL_H_

#include "dragon/kernels/op_kernels.h"
#include "dragon/operators/vision/conv_op_base.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void ConvOpBase<Context>::Im2Col(const T* im, T* col) {
  if (num_axes_ == 1 || num_axes_ == 2) {
    kernels::Im2Col2d(
        conv_in_channels_,
        in_shape_[0],
        num_axes_ == 1 ? 1 : in_shape_[1],
        out_shape_[0],
        num_axes_ == 1 ? 1 : out_shape_[1],
        kshape_[0],
        num_axes_ == 1 ? 1 : kshape_[1],
        strides_[0],
        num_axes_ == 1 ? 1 : strides_[1],
        pads_begin_[0],
        num_axes_ == 1 ? 0 : pads_begin_[1],
        dilations_[0],
        num_axes_ == 1 ? 1 : dilations_[1],
        data_format(),
        im,
        col,
        ctx());
  } else {
    kernels::Im2ColNd(
        num_axes_,
        conv_in_channels_,
        vec32_t{in_shape_.begin(), in_shape_.end()}.data(),
        vec32_t{out_shape_.begin(), out_shape_.end()}.data(),
        vec32_t{kshape_.begin(), kshape_.end()}.data(),
        vec32_t{strides_.begin(), strides_.end()}.data(),
        vec32_t{pads_begin_.begin(), pads_begin_.end()}.data(),
        vec32_t{dilations_.begin(), dilations_.end()}.data(),
        data_format(),
        im,
        col,
        ctx());
  }
}

template <class Context>
template <typename T>
void ConvOpBase<Context>::Col2Im(const T* col, T* im) {
  if (num_axes_ == 1 || num_axes_ == 2) {
    kernels::Col2Im2d(
        conv_in_channels_,
        in_shape_[0],
        num_axes_ == 1 ? 1 : in_shape_[1],
        out_shape_[0],
        num_axes_ == 1 ? 1 : out_shape_[1],
        kshape_[0],
        num_axes_ == 1 ? 1 : kshape_[1],
        strides_[0],
        num_axes_ == 1 ? 1 : strides_[1],
        pads_begin_[0],
        num_axes_ == 1 ? 0 : pads_begin_[1],
        dilations_[0],
        num_axes_ == 1 ? 1 : dilations_[1],
        data_format(),
        col,
        im,
        ctx());
  } else {
    kernels::Col2ImNd(
        num_axes_,
        conv_in_channels_,
        vec32_t{in_shape_.begin(), in_shape_.end()}.data(),
        vec32_t{out_shape_.begin(), out_shape_.end()}.data(),
        vec32_t{kshape_.begin(), kshape_.end()}.data(),
        vec32_t{strides_.begin(), strides_.end()}.data(),
        vec32_t{pads_begin_.begin(), pads_begin_.end()}.data(),
        vec32_t{dilations_.begin(), dilations_.end()}.data(),
        data_format(),
        col,
        im,
        ctx());
  }
}

template <class Context>
template <typename T>
void ConvOpBase<Context>::FwdData(const T* X, const T* W, T* Y) {
  auto* X_col = const_cast<T*>(X);
  if (skip_im2col_ == 0) {
    X_col = ctx()->workspace()->template data<T, Context>(col_dim_);
    Im2Col(X, X_col);
  }
  for (int g = 0; g < group_; g++) {
    if (data_format() == "NCHW") {
      math::Gemm(
          CblasNoTrans,
          CblasNoTrans,
          conv_out_channels_ / group_,
          conv_out_dim_,
          kernel_dim_,
          1.f,
          W + W_stride_ * g,
          X_col + col_stride_ * g,
          0.f,
          Y + Y_stride1_ * g,
          ctx());
    } else if (data_format() == "NHWC") {
      math::Gemm(
          CblasNoTrans,
          CblasTrans,
          conv_out_dim_,
          conv_out_channels_,
          kernel_dim_,
          1.f,
          X_col,
          W,
          0.f,
          Y,
          ctx());
    }
  }
}

template <class Context>
template <typename T>
void ConvOpBase<Context>::FwdBias(const T* B, T* Y) {
  const auto batch_size = Input(0).dim(0);
  if (data_format() == "NCHW") {
    kernels::BiasAdd(batch_size, out_dim_, out_channels_, Y, B, Y, ctx());
  } else if (data_format() == "NHWC") {
    kernels::BiasAdd(batch_size * out_dim_, 1, out_channels_, Y, B, Y, ctx());
  }
}

template <class Context>
template <typename T>
void ConvOpBase<Context>::BwdData(const T* dY, const T* W, T* dX) {
  auto* dX_col = (skip_im2col_ == 0)
      ? ctx()->workspace()->template data<T, Context>(col_dim_)
      : dX;
  for (int g = 0; g < group_; g++) {
    if (data_format() == "NCHW") {
      math::Gemm(
          CblasTrans,
          CblasNoTrans,
          kernel_dim_,
          conv_out_dim_,
          conv_out_channels_ / group_,
          1.f,
          W + W_stride_ * g,
          dY + Y_stride1_ * g,
          0.f,
          dX_col + col_stride_ * g,
          ctx());
    } else if (data_format() == "NHWC") {
      math::Gemm(
          CblasNoTrans,
          CblasNoTrans,
          conv_out_dim_,
          kernel_dim_,
          conv_out_channels_,
          1.f,
          dY,
          W,
          0.f,
          dX_col,
          ctx());
    }
  }
  if (skip_im2col_ == 0) {
    Col2Im(dX_col, dX);
  }
}

template <class Context>
template <typename T>
void ConvOpBase<Context>::BwdFilter(
    const T* dY,
    const T* X,
    T* dW,
    bool accum) {
  auto* X_col = const_cast<T*>(X);
  if (skip_im2col_ == 0) {
    X_col = ctx()->workspace()->template data<T, Context>(col_dim_);
    Im2Col(X, X_col);
  }
  for (int g = 0; g < group_; g++) {
    if (data_format() == "NCHW") {
      math::Gemm(
          CblasNoTrans,
          CblasTrans,
          conv_out_channels_ / group_,
          kernel_dim_,
          conv_out_dim_,
          1.f,
          dY + Y_stride1_ * g,
          X_col + col_stride_ * g,
          accum ? 1.f : 0.f,
          dW + W_stride_ * g,
          ctx());
    } else if (data_format() == "NHWC") {
      math::Gemm(
          CblasTrans,
          CblasNoTrans,
          conv_out_channels_,
          kernel_dim_,
          conv_out_dim_,
          1.f,
          dY,
          X_col,
          accum ? 1.f : 0.f,
          dW,
          ctx());
    }
  }
}

template <class Context>
template <typename T>
void ConvOpBase<Context>::BwdBias(const T* dY, T* dB) {
  vec64_t dims, axes;
  const auto batch_size = Input(0).dim(0);
  if (data_format() == "NCHW") {
    dims = {batch_size, out_channels_, out_dim_}, axes = {0, 2};
  } else if (data_format() == "NHWC") {
    dims = {batch_size, out_dim_, out_channels_}, axes = {0, 1};
  }
  math::ReduceSum(3, dims.data(), 2, axes.data(), 1.f, dY, dB, ctx());
}

} // namespace dragon

#endif // DRAGON_OPERATORS_VISION_CONV_OP_IMPL_H_
