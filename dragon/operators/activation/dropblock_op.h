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

#ifndef DRAGON_OPERATORS_ACTIVATION_DROPBLOCK_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_DROPBLOCK_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class DropBlockOp final : public Operator<Context> {
 public:
  DropBlockOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        block_size_(OP_SINGLE_ARG(int64_t, "block_size", 7)) {
    INITIALIZE_OP_SINGLE_ARG(float, ratio, 0.1f);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t block_size_;
  DECLARE_OP_SINGLE_ARG(float, ratio);
};

template <class Context>
class DropBlockGradientOp final : public Operator<Context> {
 public:
  DropBlockGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    INITIALIZE_OP_SINGLE_ARG(float, ratio, 0.1f);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_OP_SINGLE_ARG(float, ratio);
};

DEFINE_OP_SINGLE_ARG(float, DropBlockOp, ratio);
DEFINE_OP_SINGLE_ARG(float, DropBlockGradientOp, ratio);

} // namespace dragon

#endif // DRAGON_OPERATORS_ACTIVATION_DROPBLOCK_OP_H_
