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
class SoftmaxOp final : public Operator<Context> {
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
class SoftmaxGradientOp final : public Operator<Context> {
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
class CuDNNSoftmaxOp final : public Operator<Context> {
 public:
  CuDNNSoftmaxOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    CuDNNCreateTensorDesc(&input_desc_);
  }
  USE_OPERATOR_FUNCTIONS;

  ~CuDNNSoftmaxOp() {
    CuDNNDestroyTensorDesc(input_desc_);
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
class CuDNNSoftmaxGradientOp final : public Operator<Context> {
 public:
  CuDNNSoftmaxGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    CuDNNCreateTensorDesc(&input_desc_);
  }
  USE_OPERATOR_FUNCTIONS;

  ~CuDNNSoftmaxGradientOp() {
    CuDNNDestroyTensorDesc(input_desc_);
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

#ifdef USE_MLU
template <class Context>
class CNNLSoftmaxOp : public Operator<Context> {
 public:
  CNNLSoftmaxOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    CNNLCreateTensorDesc(&input_desc_);
  }
  USE_OPERATOR_FUNCTIONS;

  ~CNNLSoftmaxOp() {
    CNNLDestroyTensorDesc(input_desc_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  cnnlTensorDescriptor_t input_desc_;
};

template <class Context>
class CNNLSoftmaxGradientOp final : public CNNLSoftmaxOp<Context> {
 public:
  CNNLSoftmaxGradientOp(const OperatorDef& def, Workspace* ws)
      : CNNLSoftmaxOp<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_ACTIVATION_SOFTMAX_OP_H_
