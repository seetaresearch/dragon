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

#ifndef DRAGON_OPERATORS_VISION_POOL_OP_H_
#define DRAGON_OPERATORS_VISION_POOL_OP_H_

#include "dragon/operators/vision/pool_op_base.h"

namespace dragon {

template <class Context>
class PoolOp final : public PoolOpBase<Context> {
 public:
  explicit PoolOp(const OperatorDef& def, Workspace* ws)
      : PoolOpBase<Context>(def, ws) {
    GetBaseArguments();
  }
  USE_OPERATOR_FUNCTIONS;
  USE_POOL_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class PoolGradientOp final : public PoolOpBase<Context> {
 public:
  PoolGradientOp(const OperatorDef& def, Workspace* ws)
      : PoolOpBase<Context>(def, ws) {
    GetBaseArguments();
  }
  USE_OPERATOR_FUNCTIONS;
  USE_POOL_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

#ifdef USE_CUDNN

template <class Context>
class CuDNNPoolOp final : public CuDNNPoolOpBase<Context> {
 public:
  CuDNNPoolOp(const OperatorDef& def, Workspace* ws)
      : CuDNNPoolOpBase<Context>(def, ws) {
    CuDNNCreateTensorDesc(&input_desc_);
    CuDNNCreateTensorDesc(&output_desc_);
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc_));
  }
  USE_OPERATOR_FUNCTIONS;
  USE_POOL_FUNCTIONS;
  USE_CUDNN_POOL_FUNCTIONS;

  ~CuDNNPoolOp() {
    CuDNNDestroyTensorDesc(&input_desc_);
    CuDNNDestroyTensorDesc(&output_desc_);
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_desc_));
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  cudnnTensorDescriptor_t input_desc_;
  cudnnTensorDescriptor_t output_desc_;
};

template <class Context>
class CuDNNPoolGradientOp final : public CuDNNPoolOpBase<Context> {
 public:
  CuDNNPoolGradientOp(const OperatorDef& def, Workspace* ws)
      : CuDNNPoolOpBase<Context>(def, ws) {
    CuDNNCreateTensorDesc(&input_desc_);
    CuDNNCreateTensorDesc(&output_desc_);
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc_));
  }
  USE_OPERATOR_FUNCTIONS;
  USE_POOL_FUNCTIONS;
  USE_CUDNN_POOL_FUNCTIONS;

  ~CuDNNPoolGradientOp() {
    CuDNNDestroyTensorDesc(&input_desc_);
    CuDNNDestroyTensorDesc(&output_desc_);
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_desc_));
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  cudnnTensorDescriptor_t input_desc_;
  cudnnTensorDescriptor_t output_desc_;
};

#endif // USE_CUDNN

#ifdef USE_MPS

template <class Context>
class MPSPoolOp final : public MPSPoolOpBase<Context> {
 public:
  MPSPoolOp(const OperatorDef& def, Workspace* ws)
      : MPSPoolOpBase<Context>(def, ws) {
    graph_ = MPSCreateGraph();
  }
  USE_OPERATOR_FUNCTIONS;
  USE_POOL_FUNCTIONS;
  USE_MPS_POOL_FUNCTIONS;

  ~MPSPoolOp() {
    NSReleaseObject(graph_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
};

template <class Context>
class MPSPoolGradientOp final : public MPSPoolOpBase<Context> {
 public:
  MPSPoolGradientOp(const OperatorDef& def, Workspace* ws)
      : MPSPoolOpBase<Context>(def, ws) {
    graph_ = MPSCreateGraph();
  }
  USE_OPERATOR_FUNCTIONS;
  USE_POOL_FUNCTIONS;
  USE_MPS_POOL_FUNCTIONS;

  ~MPSPoolGradientOp() {
    NSReleaseObject(graph_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
};

#endif // USE_MPS

} // namespace dragon

#endif // DRAGON_OPERATORS_VISION_POOL_OP_H_
