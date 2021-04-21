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

#ifndef DRAGON_OPERATORS_ARRAY_ROLL_OP_H_
#define DRAGON_OPERATORS_ARRAY_ROLL_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class RollOp final : public Operator<Context> {
 public:
  RollOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws), axes_(OP_REPEATED_ARG(int64_t, "axes")) {
    INITIALIZE_OP_REPEATED_ARG(int64_t, shifts);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Generic>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  vec64_t axes_;
  DECLARE_OP_REPEATED_ARG(int64_t, shifts);
};

template <class Context>
class RollGradientOp : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(RollGradientOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

DEFINE_OP_REPEATED_ARG(int64_t, RollOp, shifts);

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_ROLL_OP_H_
