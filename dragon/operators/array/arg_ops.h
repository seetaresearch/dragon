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

#ifndef DRAGON_OPERATORS_ARRAY_ARG_OPS_H_
#define DRAGON_OPERATORS_ARRAY_ARG_OPS_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class ArgMaxOp final : public Operator<Context> {
 public:
  ArgMaxOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        keep_dims_(OP_SINGLE_ARG(int64_t, "keepdims", 0)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t keep_dims_;
};

template <class Context>
class ArgMinOp final : public Operator<Context> {
 public:
  ArgMinOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        keep_dims_(OP_SINGLE_ARG(int64_t, "keepdims", 0)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t keep_dims_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_ARG_OPS_H_
