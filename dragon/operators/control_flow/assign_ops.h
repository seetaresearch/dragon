/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *    <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_CONTROL_FLOW_ASSIGN_OPS_H_
#define DRAGON_OPERATORS_CONTROL_FLOW_ASSIGN_OPS_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class AssignOp final : public Operator<Context> {
 public:
  AssignOp(const OperatorDef& def, Workspace* ws) : Operator<Context>(def, ws) {
    GET_ARGS_WITH_DESC(int64_t, starts);
    GET_ARGS_WITH_DESC(int64_t, sizes);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_ARGS_WITH_DESC(int64_t, starts);
  DECLARE_ARGS_WITH_DESC(int64_t, sizes);
};

template <class Context>
class MaskedAssignOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(MaskedAssignOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

DEFINE_ARGS_WITH_DESC(int64_t, AssignOp, starts);
DEFINE_ARGS_WITH_DESC(int64_t, AssignOp, sizes);

} // namespace dragon

#endif // DRAGON_OPERATORS_CONTROL_FLOW_ASSIGN_OPS_H_
