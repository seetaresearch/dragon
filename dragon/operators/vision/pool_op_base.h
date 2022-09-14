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

  cudnnPoolingDescriptor_t pool_desc_;
  cudnnPoolingMode_t pool_mode_;
};

#define USE_CUDNN_POOL_FUNCTIONS               \
  using CuDNNPoolOpBase<Context>::SetPoolDesc; \
  using CuDNNPoolOpBase<Context>::pool_desc_;  \
  using CuDNNPoolOpBase<Context>::pool_mode_
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
    NSReleaseObject(pool2d_desc_);
  }

 protected:
  void SetPoolDesc();

  MPSGraphPooling2DOpDescriptor_t pool2d_desc_;
};

#define USE_MPS_POOL_FUNCTIONS               \
  using MPSPoolOpBase<Context>::SetPoolDesc; \
  using MPSPoolOpBase<Context>::pool2d_desc_;

#endif // USE_MPS

} // namespace dragon

#endif // RAGON_OPERATORS_VISION_POOL_OP_BASE_H_
