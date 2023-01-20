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

#ifndef DRAGON_OPERATORS_ACTIVATION_LOG_SOFTMAX_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_LOG_SOFTMAX_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class LogSoftmaxOp : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(LogSoftmaxOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class LogSoftmaxGradientOp : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(LogSoftmaxGradientOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

#ifdef USE_MLU
template <class Context>
class CNNLLogSoftmaxOp : public Operator<Context> {
 public:
  CNNLLogSoftmaxOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    CNNLCreateTensorDesc(&input_desc_);
  }
  USE_OPERATOR_FUNCTIONS;

  ~CNNLLogSoftmaxOp() {
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
class CNNLLogSoftmaxGradientOp final : public CNNLLogSoftmaxOp<Context> {
 public:
  CNNLLogSoftmaxGradientOp(const OperatorDef& def, Workspace* ws)
      : CNNLLogSoftmaxOp<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_ACTIVATION_LOG_SOFTMAX_OP_H_
