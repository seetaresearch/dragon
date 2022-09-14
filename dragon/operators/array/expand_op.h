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

#ifndef DRAGON_OPERATORS_ARRAY_EXPAND_OP_H_
#define DRAGON_OPERATORS_ARRAY_EXPAND_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class ExpandOp final : public Operator<Context> {
 public:
  ExpandOp(const OperatorDef& def, Workspace* ws) : Operator<Context>(def, ws) {
    INITIALIZE_OP_REPEATED_ARG(int64_t, dims);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Generic>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_OP_REPEATED_ARG(int64_t, dims);
};

template <class Context>
class ExpandGradientOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(ExpandGradientOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

DEFINE_OP_REPEATED_ARG(int64_t, ExpandOp, dims);

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_EXPAND_OP_H_
