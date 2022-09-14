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
void ConvOpBase<Context>::WeightedX(const T* x, const T* w, T* y) {
  auto* col = const_cast<T*>(x);
  if (skip_im2col_ == 0) {
    col = ctx()->workspace()->template data<T, Context>(col_dim_);
    Im2Col(x, col);
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
          w + W_stride_ * g,
          col + col_stride_ * g,
          0.f,
          y + Y_stride1_ * g,
          ctx());
    } else if (data_format() == "NHWC") {
      math::Gemm(
          CblasNoTrans,
          CblasTrans,
          conv_out_dim_,
          conv_out_channels_,
          kernel_dim_,
          1.f,
          col,
          w,
          0.f,
          y,
          ctx());
    }
  }
}

template <class Context>
template <typename T>
void ConvOpBase<Context>::AddBias(const T* bias, T* y) {
  if (data_format() == "NCHW") {
    kernels::BiasAdd(
        Input(0).dim(0), out_dim_, out_channels_, y, bias, y, ctx());
  } else if (data_format() == "NHWC") {
    kernels::BiasAdd(
        Input(0).dim(0) * out_dim_, 1, out_channels_, y, bias, y, ctx());
  }
}

template <class Context>
template <typename T>
void ConvOpBase<Context>::GradX(const T* dy, const T* w, T* dx) {
  auto* col = (skip_im2col_ == 0)
      ? ctx()->workspace()->template data<T, Context>(col_dim_)
      : dx;
  for (int g = 0; g < group_; g++) {
    if (data_format() == "NCHW") {
      math::Gemm(
          CblasTrans,
          CblasNoTrans,
          kernel_dim_,
          conv_out_dim_,
          conv_out_channels_ / group_,
          1.f,
          w + W_stride_ * g,
          dy + Y_stride1_ * g,
          0.f,
          col + col_stride_ * g,
          ctx());
    } else if (data_format() == "NHWC") {
      math::Gemm(
          CblasNoTrans,
          CblasNoTrans,
          conv_out_dim_,
          kernel_dim_,
          conv_out_channels_,
          1.f,
          dy,
          w,
          0.f,
          col,
          ctx());
    }
  }
  if (skip_im2col_ == 0) {
    Col2Im(col, dx);
  }
}

template <class Context>
template <typename T>
void ConvOpBase<Context>::GradW(const T* dy, const T* x, T* dw, bool accum) {
  auto* col = const_cast<T*>(x);
  if (skip_im2col_ == 0) {
    col = ctx()->workspace()->template data<T, Context>(col_dim_);
    Im2Col(x, col);
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
          dy + Y_stride1_ * g,
          col + col_stride_ * g,
          accum ? 1.f : 0.f,
          dw + W_stride_ * g,
          ctx());
    } else if (data_format() == "NHWC") {
      math::Gemm(
          CblasTrans,
          CblasNoTrans,
          conv_out_channels_,
          kernel_dim_,
          conv_out_dim_,
          1.f,
          dy,
          col,
          accum ? 1.f : 0.f,
          dw,
          ctx());
    }
  }
}

template <class Context>
template <typename T>
void ConvOpBase<Context>::GradBias(const T* dy, T* db) {
  vec64_t dims, axes;
  if (data_format() == "NCHW") {
    dims = {Input(0).dim(0), out_channels_, out_dim_};
    axes = {0, 2};
  } else if (data_format() == "NHWC") {
    dims = {Input(0).dim(0), out_dim_, out_channels_};
    axes = {0, 1};
  }
  math::ReduceSum(3, dims.data(), 2, axes.data(), 1.f, dy, db, ctx());
}

} // namespace dragon

#endif // DRAGON_OPERATORS_VISION_CONV_OP_IMPL_H_
