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

#ifndef DRAGON_OPERATORS_ACTIVATION_SOFTMAX_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_SOFTMAX_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class SoftmaxOp : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(SoftmaxOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class SoftmaxGradientOp : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(SoftmaxGradientOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

#ifdef USE_CUDNN

template <class Context>
class CuDNNSoftmaxOp final : public SoftmaxOp<Context> {
 public:
  CuDNNSoftmaxOp(const OperatorDef& def, Workspace* ws)
      : SoftmaxOp<Context>(def, ws) {
    CuDNNCreateTensorDesc(&input_desc_);
  }
  USE_OPERATOR_FUNCTIONS;

  ~CuDNNSoftmaxOp() {
    CuDNNDestroyTensorDesc(&input_desc_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  cudnnTensorDescriptor_t input_desc_;
};

template <class Context>
class CuDNNSoftmaxGradientOp final : public SoftmaxGradientOp<Context> {
 public:
  CuDNNSoftmaxGradientOp(const OperatorDef& def, Workspace* ws)
      : SoftmaxGradientOp<Context>(def, ws) {
    CuDNNCreateTensorDesc(&input_desc_);
  }
  USE_OPERATOR_FUNCTIONS;

  ~CuDNNSoftmaxGradientOp() {
    CuDNNDestroyTensorDesc(&input_desc_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  cudnnTensorDescriptor_t input_desc_;
};

#endif // USE_CUDNN

} // namespace dragon

#endif // DRAGON_OPERATORS_ACTIVATION_SOFTMAX_OP_H_
