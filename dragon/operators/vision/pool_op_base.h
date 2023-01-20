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

#ifndef DRAGON_OPERATORS_VISION_POOL_OP_BASE_H_
#define DRAGON_OPERATORS_VISION_POOL_OP_BASE_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class PoolOpBase : public Operator<Context> {
 public:
  PoolOpBase(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        mode_(OP_SINGLE_ARG(string, "mode", "MAX")),
        padding_(OP_SINGLE_ARG(string, "padding", "VALID")),
        ceil_mode_(OP_SINGLE_ARG(int64_t, "ceil_mode", 0)),
        global_pool_(OP_SINGLE_ARG(int64_t, "global_pool", 0)) {
    if (data_format() == "NCHW") {
      axis_ = 2;
    } else if (data_format() == "NHWC") {
      axis_ = 1;
    } else {
      LOG(FATAL) << "Unknown DataFormat: " << data_format();
    }
    num_axes_ = -1; // Unknown
  }
  USE_OPERATOR_FUNCTIONS;

 protected:
  void GetBaseArguments();

  void ComputeOutShape();

  string mode_, padding_;
  int64_t axis_, num_axes_;
  int64_t global_pool_, ceil_mode_;
  vec64_t kshape_, strides_;
  vec64_t pads_begin_, pads_end_;
  vec64_t in_dims_, out_dims_, out_shape_;
};

#define USE_POOL_FUNCTIONS                     \
  using PoolOpBase<Context>::GetBaseArguments; \
  using PoolOpBase<Context>::ComputeOutShape;  \
  using PoolOpBase<Context>::mode_;            \
  using PoolOpBase<Context>::ceil_mode_;       \
  using PoolOpBase<Context>::kshape_;          \
  using PoolOpBase<Context>::strides_;         \
  using PoolOpBase<Context>::pads_begin_;      \
  using PoolOpBase<Context>::pads_end_;        \
  using PoolOpBase<Context>::axis_;            \
  using PoolOpBase<Context>::num_axes_;        \
  using PoolOpBase<Context>::in_dims_;         \
  using PoolOpBase<Context>::out_dims_;        \
  using PoolOpBase<Context>::out_shape_;

#ifdef USE_CUDNN
template <class Context>
class CuDNNPoolOpBase : public PoolOpBase<Context> {
 public:
  CuDNNPoolOpBase(const OperatorDef& def, Workspace* ws)
      : PoolOpBase<Context>(def, ws) {
    GetBaseArguments();
    CuDNNCreateTensorDesc(&input_desc_);
    CuDNNCreateTensorDesc(&output_desc_);
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc_));
    if (mode_ == "MAX") {
      pool_mode_ = CUDNN_POOLING_MAX_DETERMINISTIC;
    } else if (mode_ == "AVG") {
      pool_mode_ = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    } else {
      LOG(FATAL) << "Unknown Mode: " << mode_;
    }
  }
  USE_OPERATOR_FUNCTIONS;
  USE_POOL_FUNCTIONS;

  ~CuDNNPoolOpBase() {
    CuDNNDestroyTensorDesc(input_desc_);
    CuDNNDestroyTensorDesc(output_desc_);
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_desc_));
  }

 protected:
  void SetPoolDesc() {
    if (num_axes_ == 1 || num_axes_ == 2) {
      CUDNN_CHECK(cudnnSetPooling2dDescriptor(
          pool_desc_,
          pool_mode_,
          CUDNN_PROPAGATE_NAN,
          kshape_[0],
          num_axes_ == 1 ? 1 : kshape_[1],
          pads_begin_[0],
          num_axes_ == 1 ? 0 : pads_begin_[1],
          strides_[0],
          num_axes_ == 1 ? 1 : strides_[1]));
    } else {
      CUDNN_CHECK(cudnnSetPoolingNdDescriptor(
          pool_desc_,
          pool_mode_,
          CUDNN_NOT_PROPAGATE_NAN,
          num_axes_,
          vec32_t{kshape_.begin(), kshape_.end()}.data(),
          vec32_t{pads_begin_.begin(), pads_begin_.end()}.data(),
          vec32_t{strides_.begin(), strides_.end()}.data()));
    }
  }

  cudnnPoolingMode_t pool_mode_;
  cudnnPoolingDescriptor_t pool_desc_;
  cudnnTensorDescriptor_t input_desc_, output_desc_;
};
#endif // USE_CUDNN

#ifdef USE_MPS
#ifdef __OBJC__
typedef MPSGraphPooling2DOpDescriptor* MPSGraphPooling2DOpDescriptor_t;
#else
struct MPSGraphPooling2DOpDescriptor;
typedef MPSGraphPooling2DOpDescriptor* MPSGraphPooling2DOpDescriptor_t;
#endif

template <class Context>
class MPSPoolOpBase : public PoolOpBase<Context> {
 public:
  MPSPoolOpBase(const OperatorDef& def, Workspace* ws);
  USE_OPERATOR_FUNCTIONS;
  USE_POOL_FUNCTIONS;

  ~MPSPoolOpBase() {
    NSReleaseObject(graph_);
    NSReleaseObject(pool2d_desc_);
  }

 protected:
  void SetPoolDesc();

  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
  MPSGraphPooling2DOpDescriptor_t pool2d_desc_;
};
#endif // USE_MPS

#ifdef USE_MLU
template <class Context>
class CNNLPoolOpBase : public PoolOpBase<Context> {
 public:
  CNNLPoolOpBase(const OperatorDef& def, Workspace* ws)
      : PoolOpBase<Context>(def, ws) {
    GetBaseArguments();
    CNNLCreateTensorDesc(&input_desc_);
    CNNLCreateTensorDesc(&output_desc_);
    CNNLCreateTensorDesc(&index_desc_);
    CNNL_CHECK(cnnlCreatePoolingDescriptor(&pool_desc_));
    if (mode_ == "MAX") {
      pool_mode_ = CNNL_POOLING_MAX;
    } else if (mode_ == "AVG") {
      pool_mode_ = CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    } else {
      LOG(FATAL) << "Unknown Mode: " << mode_;
    }
  }
  USE_OPERATOR_FUNCTIONS;
  USE_POOL_FUNCTIONS;

  ~CNNLPoolOpBase() {
    CNNLDestroyTensorDesc(input_desc_);
    CNNLDestroyTensorDesc(output_desc_);
    CNNLDestroyTensorDesc(index_desc_);
    CNNL_CHECK(cnnlDestroyPoolingDescriptor(pool_desc_));
  }

 protected:
  void SetPoolDesc() {
    if (num_axes_ == 1 || num_axes_ == 2) {
      CNNL_CHECK(cnnlSetPooling2dDescriptor_v2(
          pool_desc_,
          pool_mode_,
          CNNL_NOT_PROPAGATE_NAN,
          kshape_[0],
          num_axes_ == 1 ? 1 : kshape_[1],
          pads_begin_[0],
          pads_end_[0],
          num_axes_ == 1 ? 0 : pads_begin_[1],
          num_axes_ == 1 ? 0 : pads_end_[1],
          strides_[0],
          num_axes_ == 1 ? 1 : strides_[1],
          1, // dilation_h
          1, // dilation_w
          ceil_mode_ > 0));
    } else {
      vec32_t pads(num_axes_ * 2, 0);
      for (int i = 0; i < pads_begin_.size(); ++i) {
        pads[i * 2] = pads_begin_[i];
        pads[i * 2 + 1] = pads_end_[i];
      }
      CNNL_CHECK(cnnlSetPoolingNdDescriptor_v2(
          pool_desc_,
          pool_mode_,
          CNNL_NOT_PROPAGATE_NAN,
          num_axes_,
          vec32_t({kshape_.begin(), kshape_.end()}).data(),
          pads.data(),
          vec32_t({strides_.begin(), strides_.end()}).data(),
          vec32_t(num_axes_, 1).data(),
          ceil_mode_ > 0));
    }
  }

  cnnlPoolingMode_t pool_mode_;
  cnnlPoolingDescriptor_t pool_desc_;
  cnnlTensorDescriptor_t input_desc_, output_desc_, index_desc_;
};
#endif // USE_MLU

} // namespace dragon

#endif // RAGON_OPERATORS_VISION_POOL_OP_BASE_H_
