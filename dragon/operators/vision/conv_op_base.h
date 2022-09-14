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
  void WeightedX(const T* x, const T* w, T* y);

  template <typename T>
  void AddBias(const T* b, T* y);

  template <typename T>
  void GradX(const T* dy, const T* w, T* dx);

  template <typename T>
  void GradW(const T* dy, const T* x, T* dw, bool = false);

  template <typename T>
  void GradBias(const T* dy, T* db);

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

DEFINE_OP_REPEATED_ARG(int64_t, ConvOpBase, output_shape);
DEFINE_OP_REPEATED_ARG(int64_t, ConvOpBase, output_padding);

#define USE_CONV_FUNCTIONS                       \
  using ConvOpBase<Context>::GetBaseArguments;   \
  using ConvOpBase<Context>::Reshape;            \
  using ConvOpBase<Context>::Transposed;         \
  using ConvOpBase<Context>::HasBias;            \
  using ConvOpBase<Context>::WeightedX;          \
  using ConvOpBase<Context>::AddBias;            \
  using ConvOpBase<Context>::GradX;              \
  using ConvOpBase<Context>::GradW;              \
  using ConvOpBase<Context>::GradBias;           \
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

#ifdef USE_CUDNN

template <class Context>
class CuDNNConvOpBase : public ConvOpBase<Context> {
 public:
  CuDNNConvOpBase(const OperatorDef& def, Workspace* ws)
      : ConvOpBase<Context>(def, ws) {
    GetBaseArguments();
    if (data_format() == "NCHW") {
      tensor_format_ = CUDNN_TENSOR_NCHW;
    } else if (data_format() == "NHWC") {
      tensor_format_ = CUDNN_TENSOR_NHWC;
    } else {
      LOG(FATAL) << "Unknown DataFormat: " << data_format();
    }
  }
  USE_OPERATOR_FUNCTIONS;
  USE_CONV_FUNCTIONS;

 protected:
  template <typename T>
  void SetConvDesc() {
    auto input_type = TypeMeta::Id<T>();
    if (input_type == TypeMeta::Id<float16>()) {
      compute_type_ = CUDNN_DATA_FLOAT;
    } else if (input_type == TypeMeta::Id<float>()) {
      compute_type_ = CUDNN_DATA_FLOAT;
    } else if (input_type == TypeMeta::Id<double>()) {
      compute_type_ = CUDNN_DATA_DOUBLE;
    }
    if (num_axes_ == 1 || num_axes_ == 2) {
      CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
          conv_desc_,
          pads_begin_[0],
          num_axes_ == 1 ? 0 : pads_begin_[1],
          strides_[0],
          num_axes_ == 1 ? 1 : strides_[1],
          dilations_[0],
          num_axes_ == 1 ? 1 : dilations_[1],
          CUDNN_CROSS_CORRELATION,
          compute_type_));
    } else {
      CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(
          conv_desc_,
          num_axes_,
          vec32_t{pads_begin_.begin(), pads_begin_.end()}.data(),
          vec32_t{strides_.begin(), strides_.end()}.data(),
          vec32_t{dilations_.begin(), dilations_.end()}.data(),
          CUDNN_CROSS_CORRELATION,
          compute_type_));
    }
    CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc_, group_));
    if (TENSOR_CORE_AVAILABLE()) {
      cudnnMathType_t math_type;
      if (input_type == TypeMeta::Id<float16>()) {
        math_type = CUDNN_TENSOR_OP_MATH;
      } else {
        math_type = CUDNN_DEFAULT_MATH;
#if CUDNN_VERSION_MIN(8, 0, 0)
        if (!CUDAContext::objects().cudnn_allow_tf32_) {
          math_type = CUDNN_FMA_MATH;
        }
#endif
      }
      CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc_, math_type));
    }
  }

  template <typename T>
  void SetFilterDesc() {
    if (num_axes_ == 1 || num_axes_ == 2) {
      CUDNN_CHECK(cudnnSetFilter4dDescriptor(
          filter_desc_,
          CuDNNType<T>::type,
          tensor_format_,
          conv_out_channels_,
          conv_in_channels_ / group_,
          kshape_[0],
          num_axes_ == 1 ? 1 : kshape_[1]));
    } else {
      vec64_t dims = {conv_out_channels_, conv_in_channels_ / group_};
      dims.insert(dims.end(), kshape_.begin(), kshape_.end());
      CUDNN_CHECK(cudnnSetFilterNdDescriptor(
          filter_desc_,
          CuDNNType<T>::type,
          tensor_format_,
          dims.size(),
          vec32_t{dims.begin(), dims.end()}.data()));
    }
  }

  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnDataType_t compute_type_;
  cudnnTensorFormat_t tensor_format_;
  size_t scratch_size_, scratch_max_size_;
};

#define USE_CUDNN_CONV_FUNCTIONS                  \
  using CuDNNConvOpBase<Context>::SetConvDesc;    \
  using CuDNNConvOpBase<Context>::SetFilterDesc;  \
  using CuDNNConvOpBase<Context>::conv_desc_;     \
  using CuDNNConvOpBase<Context>::filter_desc_;   \
  using CuDNNConvOpBase<Context>::compute_type_;  \
  using CuDNNConvOpBase<Context>::tensor_format_; \
  using CuDNNConvOpBase<Context>::scratch_size_;  \
  using CuDNNConvOpBase<Context>::scratch_max_size_

#endif // USE_CUDNN

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

#endif

} // namespace dragon

#endif // DRAGON_OPERATORS_VISION_CONV_OP_BASE_H_
