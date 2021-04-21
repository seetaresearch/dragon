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

#ifndef DRAGON_OPERATORS_ACTIVATION_GELU_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_GELU_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class GeluOp : public Operator<Context> {
 public:
  GeluOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        approximate_(OP_SINGLE_ARG(int64_t, "approximate", 0)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t approximate_;
};

template <class Context>
class GeluGradientOp : public Operator<Context> {
 public:
  GeluGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        approximate_(OP_SINGLE_ARG(int64_t, "approximate", 0)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t approximate_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_ACTIVATION_GELU_OP_H_
