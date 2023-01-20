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

#ifndef DRAGON_OPERATORS_VISION_CONV_OP_BASE_H_
#define DRAGON_OPERATORS_VISION_CONV_OP_BASE_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class ConvOpBase : public Operator<Context> {
 public:
  ConvOpBase(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        padding_(OP_SINGLE_ARG(string, "padding", "VALID")),
        out_channels_(OP_SINGLE_ARG(int64_t, "out_channels", 0)),
        group_(OP_SINGLE_ARG(int64_t, "group", 1)) {
    if (data_format() == "NCHW") {
      axis_ = 2;
    } else if (data_format() == "NHWC") {
      axis_ = 1;
    } else {
      LOG(FATAL) << "Unknown DataFormat: " << data_format();
    }
    num_axes_ = -1; // Unknown
    INITIALIZE_OP_REPEATED_ARG(int64_t, output_shape);
    INITIALIZE_OP_REPEATED_ARG(int64_t, output_padding);
  }
  USE_OPERATOR_FUNCTIONS;

 protected:
  virtual bool HasBias() {
    return false;
  }

  virtual bool Transposed() {
    return false;
  }

  void GetBaseArguments();

  void ComputeOutShape();

  void Reshape(bool backward = false);

  template <typename T>
  void Im2Col(const T* im, T* col);

  template <typename T>
  void Col2Im(const T* col, T* im);

  template <typename T>
  void FwdData(const T* x, const T* w, T* y);

  template <typename T>
  void FwdBias(const T* b, T* y);

  template <typename T>
  void BwdData(const T* dy, const T* w, T* dx);

  template <typename T>
  void BwdFilter(const T* dy, const T* x, T* dw, bool = false);

  template <typename T>
  void BwdBias(const T* dy, T* db);

  vec64_t kshape_, dilations_, strides_;
  vec64_t pads_begin_, pads_end_;
  vec64_t in_shape_, out_shape_;
  vec64_t w_shape_, b_shape_;

  string padding_;
  int64_t group_;
  int64_t axis_, num_axes_;
  int64_t in_channels_, out_channels_;
  int64_t conv_in_channels_, conv_out_channels_;
  int64_t kernel_dim_;
  int64_t X_stride_, W_stride_, Y_stride_;

  DECLARE_OP_REPEATED_ARG(int64_t, output_shape);
  DECLARE_OP_REPEATED_ARG(int64_t, output_padding);

 private:
  int64_t skip_im2col_;
  int64_t col_dim_, col_stride_;
  int64_t out_dim_, conv_out_dim_;
  int64_t Y_stride1_;
};

#define USE_CONV_FUNCTIONS                       \
  using ConvOpBase<Context>::GetBaseArguments;   \
  using ConvOpBase<Context>::Reshape;            \
  using ConvOpBase<Context>::Transposed;         \
  using ConvOpBase<Context>::HasBias;            \
  using ConvOpBase<Context>::FwdData;            \
  using ConvOpBase<Context>::FwdBias;            \
  using ConvOpBase<Context>::BwdData;            \
  using ConvOpBase<Context>::BwdFilter;          \
  using ConvOpBase<Context>::BwdBias;            \
  using ConvOpBase<Context>::kshape_;            \
  using ConvOpBase<Context>::dilations_;         \
  using ConvOpBase<Context>::strides_;           \
  using ConvOpBase<Context>::pads_begin_;        \
  using ConvOpBase<Context>::pads_end_;          \
  using ConvOpBase<Context>::group_;             \
  using ConvOpBase<Context>::in_channels_;       \
  using ConvOpBase<Context>::out_channels_;      \
  using ConvOpBase<Context>::conv_in_channels_;  \
  using ConvOpBase<Context>::conv_out_channels_; \
  using ConvOpBase<Context>::axis_;              \
  using ConvOpBase<Context>::num_axes_;          \
  using ConvOpBase<Context>::X_stride_;          \
  using ConvOpBase<Context>::W_stride_;          \
  using ConvOpBase<Context>::Y_stride_;          \
  using ConvOpBase<Context>::in_shape_;          \
  using ConvOpBase<Context>::w_shape_;           \
  using ConvOpBase<Context>::b_shape_;           \
  using ConvOpBase<Context>::out_shape_

#ifdef USE_MPS
#ifdef __OBJC__
typedef MPSGraphConvolution2DOpDescriptor* MPSGraphConvolution2DOpDescriptor_t;
#else
struct MPSGraphConvolution2DOpDescriptor;
typedef MPSGraphConvolution2DOpDescriptor* MPSGraphConvolution2DOpDescriptor_t;
#endif

template <class Context>
class MPSConvOpBase : public ConvOpBase<Context> {
 public:
  MPSConvOpBase(const OperatorDef& def, Workspace* ws);
  USE_OPERATOR_FUNCTIONS;
  USE_CONV_FUNCTIONS;

 protected:
  void SetConvDesc();

  MPSGraphConvolution2DOpDescriptor_t conv2d_desc_;
};

#define USE_MPS_CONV_FUNCTIONS               \
  using MPSConvOpBase<Context>::SetConvDesc; \
  using MPSConvOpBase<Context>::conv2d_desc_;
#endif // USE_MPS

} // namespace dragon

#endif // DRAGON_OPERATORS_VISION_CONV_OP_BASE_H_
