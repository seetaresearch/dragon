/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *    <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_VISION_CONV_OP_BASE_H_
#define DRAGON_OPERATORS_VISION_CONV_OP_BASE_H_

#include "dragon/core/operator.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
class ConvOpBase : public Operator<Context> {
 public:
  ConvOpBase(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        padding_(OpArg<string>("padding", "VALID")),
        out_channels_(OpArg<int64_t>("out_channels", 0)),
        group_(OpArg<int64_t>("group", 1)) {
    if (data_format() == "NCHW") {
      axis_ = 2;
    } else if (data_format() == "NHWC") {
      axis_ = 1;
    } else {
      LOG(FATAL) << "Unknown DataFormat: " << data_format();
    }
    num_axes_ = -1; // Unknown
    GET_ARGS_WITH_DESC(int64_t, output_shape);
    GET_ARGS_WITH_DESC(int64_t, output_padding);
  }
  USE_OPERATOR_FUNCTIONS;

  vec64_t kshape_, stride_;
  vec64_t pad_l_, pad_r_, dilation_;
  vec64_t in_shape_, w_shape_, b_shape_, out_shape_;

  string padding_;
  int64_t group_;
  int64_t axis_, num_axes_;
  int64_t in_channels_, out_channels_, out_dim_;
  int64_t x_offset_, w_offset_, y_offset_;

  DECLARE_ARGS_WITH_DESC(int64_t, output_shape);
  DECLARE_ARGS_WITH_DESC(int64_t, output_padding);

  void Setup(int num_axes);
  void Reshape(bool backward = false);

  virtual bool HasBias() = 0;
  virtual bool Transposed() = 0;

  template <typename T>
  void Wx(const T*, const T*, T*, bool = false);

  template <typename T>
  void Pb(const T*, T*);

  template <typename T>
  void Dx(const T*, const T*, T*);

  template <typename T>
  void Dw(const T*, const T*, T*, bool = false);

  template <typename T>
  void Db(const T*, T*);

 private:
  void ComputeOutShape();

  template <typename T>
  void Im2Col(const T* im, T* col) {
    if (Input(0).ndim() == 4) {
      kernel::Im2Col2d(
          conv_in_channels_,
          in_shape_[0],
          in_shape_[1],
          out_shape_[0],
          out_shape_[1],
          kshape_[0],
          kshape_[1],
          stride_[0],
          stride_[1],
          pad_l_[0],
          pad_l_[1],
          dilation_[0],
          dilation_[1],
          data_format(),
          im,
          col,
          ctx());
    } else {
      LOG(FATAL) << "ConvNd has not been implemented.";
    }
  }

  template <typename T>
  void Col2Im(const T* col, T* im) {
    if (Input(0).ndim() == 4) {
      kernel::Col2Im2d(
          conv_in_channels_,
          in_shape_[0],
          in_shape_[1],
          out_shape_[0],
          out_shape_[1],
          kshape_[0],
          kshape_[1],
          stride_[0],
          stride_[1],
          pad_l_[0],
          pad_l_[1],
          dilation_[0],
          dilation_[1],
          data_format(),
          col,
          im,
          ctx());
    } else {
      LOG(FATAL) << "ConvNd has not been implemented.";
    }
  }

  int64_t is_1x1_;
  int64_t kernel_dim_, col_dim_;
  int64_t col_offset_, out_offset_;
  int64_t conv_in_channels_, conv_out_channels_, conv_out_dim_;
};

DEFINE_ARGS_WITH_DESC(int64_t, ConvOpBase, output_shape);
DEFINE_ARGS_WITH_DESC(int64_t, ConvOpBase, output_padding);

#define USE_CONVOLUTION_FUNCTIONS           \
  using ConvOpBase<Context>::Setup;         \
  using ConvOpBase<Context>::Reshape;       \
  using ConvOpBase<Context>::Transposed;    \
  using ConvOpBase<Context>::HasBias;       \
  using ConvOpBase<Context>::Wx;            \
  using ConvOpBase<Context>::Pb;            \
  using ConvOpBase<Context>::Dx;            \
  using ConvOpBase<Context>::Dw;            \
  using ConvOpBase<Context>::Db;            \
  using ConvOpBase<Context>::kshape_;       \
  using ConvOpBase<Context>::stride_;       \
  using ConvOpBase<Context>::pad_l_;        \
  using ConvOpBase<Context>::pad_r_;        \
  using ConvOpBase<Context>::dilation_;     \
  using ConvOpBase<Context>::group_;        \
  using ConvOpBase<Context>::in_channels_;  \
  using ConvOpBase<Context>::out_channels_; \
  using ConvOpBase<Context>::axis_;         \
  using ConvOpBase<Context>::num_axes_;     \
  using ConvOpBase<Context>::x_offset_;     \
  using ConvOpBase<Context>::w_offset_;     \
  using ConvOpBase<Context>::y_offset_;     \
  using ConvOpBase<Context>::in_shape_;     \
  using ConvOpBase<Context>::w_shape_;      \
  using ConvOpBase<Context>::b_shape_;      \
  using ConvOpBase<Context>::out_shape_

} // namespace dragon

#endif // DRAGON_OPERATORS_VISION_CONV_OP_BASE_H_
