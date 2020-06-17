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

#ifndef DRAGON_OPERATORS_VISION_POOL_OP_H_
#define DRAGON_OPERATORS_VISION_POOL_OP_H_

#include "dragon/operators/vision/pool_op_base.h"

namespace dragon {

template <class Context>
class Pool2dOp final : public PoolOpBase<Context> {
 public:
  Pool2dOp(const OperatorDef& def, Workspace* ws)
      : PoolOpBase<Context>(def, ws) {
    Setup(2);
  }
  USE_OPERATOR_FUNCTIONS;
  USE_POOLING_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class Pool2dGradientOp final : public PoolOpBase<Context> {
 public:
  Pool2dGradientOp(const OperatorDef& def, Workspace* ws)
      : PoolOpBase<Context>(def, ws) {
    Setup(2);
  }
  USE_OPERATOR_FUNCTIONS;
  USE_POOLING_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

#ifdef USE_CUDNN

template <class Context>
class CuDNNPool2dOp final : public PoolOpBase<Context> {
 public:
  CuDNNPool2dOp(const OperatorDef& def, Workspace* ws)
      : PoolOpBase<Context>(def, ws) {
    Setup(2);
    CuDNNCreateTensorDesc(&input_desc_);
    CuDNNCreateTensorDesc(&output_desc_);
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc_));
    if (mode_ == "MAX") {
#if CUDNN_VERSION_MIN(6, 0, 0)
      pool_mode_ = CUDNN_POOLING_MAX_DETERMINISTIC;
#else
      pool_mode_ = CUDNN_POOLING_MAX;
#endif
    } else if (mode_ == "AVG") {
      pool_mode_ = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    } else {
      LOG(FATAL) << "Unknown Mode: " << mode_;
    }
  }
  USE_OPERATOR_FUNCTIONS;
  USE_POOLING_FUNCTIONS;

  ~CuDNNPool2dOp() {
    CuDNNDestroyTensorDesc(&input_desc_);
    CuDNNDestroyTensorDesc(&output_desc_);
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_desc_));
  }

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  cudnnTensorDescriptor_t input_desc_;
  cudnnTensorDescriptor_t output_desc_;
  cudnnPoolingDescriptor_t pool_desc_;
  cudnnPoolingMode_t pool_mode_;
};

template <class Context>
class CuDNNPool2dGradientOp final : public PoolOpBase<Context> {
 public:
  CuDNNPool2dGradientOp(const OperatorDef& def, Workspace* ws)
      : PoolOpBase<Context>(def, ws) {
    Setup(2);
    CuDNNCreateTensorDesc(&input_desc_);
    CuDNNCreateTensorDesc(&output_desc_);
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc_));
    if (mode_ == "MAX") {
#if CUDNN_VERSION_MIN(6, 0, 0)
      pool_mode_ = CUDNN_POOLING_MAX_DETERMINISTIC;
#else
      pool_mode_ = CUDNN_POOLING_MAX;
#endif
    } else if (mode_ == "AVG") {
      pool_mode_ = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    } else {
      LOG(FATAL) << "Unknown Mode: " << mode_;
    }
  }
  USE_OPERATOR_FUNCTIONS;
  USE_POOLING_FUNCTIONS;

  ~CuDNNPool2dGradientOp() {
    CuDNNDestroyTensorDesc(&input_desc_);
    CuDNNDestroyTensorDesc(&output_desc_);
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_desc_));
  }

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  cudnnTensorDescriptor_t input_desc_;
  cudnnTensorDescriptor_t output_desc_;
  cudnnPoolingDescriptor_t pool_desc_;
  cudnnPoolingMode_t pool_mode_;
};

#endif // USE_CUDNN

} // namespace dragon

#endif // DRAGON_OPERATORS_VISION_POOLING_OP_H_
