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

#ifndef DRAGON_OPERATORS_ARRAY_SPLIT_OP_H_
#define DRAGON_OPERATORS_ARRAY_SPLIT_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class SplitOp final : public Operator<Context> {
 public:
  SplitOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        copy_chunks_(OP_SINGLE_ARG(int64_t, "copy", 1)),
        keep_dims_(OP_SINGLE_ARG(int64_t, "keepdims", 1)) {
    INITIALIZE_OP_REPEATED_ARG(int64_t, split);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Generic>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t copy_chunks_, keep_dims_;
  DECLARE_OP_REPEATED_ARG(int64_t, split);
};

template <class Context>
class SplitGradientOp final : public Operator<Context> {
 public:
  SplitGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    INITIALIZE_OP_REPEATED_ARG(int64_t, split);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input("X_spec"));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_OP_REPEATED_ARG(int64_t, split);
};

DEFINE_OP_REPEATED_ARG(int64_t, SplitOp, split);
DEFINE_OP_REPEATED_ARG(int64_t, SplitGradientOp, split);

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_SPLIT_OP_H_
