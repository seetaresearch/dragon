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

} // namespace dragon

#endif // DRAGON_OPERATORS_ACTIVATION_LOG_SOFTMAX_OP_H_
