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

#ifndef DRAGON_OPERATORS_ACTIVATION_DROP_PATH_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_DROP_PATH_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class DropPathOp final : public Operator<Context> {
 public:
  DropPathOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    INITIALIZE_OP_SINGLE_ARG(float, ratio, 0.2f);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_OP_SINGLE_ARG(float, ratio);
};

template <class Context>
class DropPathGradientOp final : public Operator<Context> {
 public:
  DropPathGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    INITIALIZE_OP_SINGLE_ARG(float, ratio, 0.5f);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_OP_SINGLE_ARG(float, ratio);
};

DEFINE_OP_SINGLE_ARG(float, DropPathOp, ratio);
DEFINE_OP_SINGLE_ARG(float, DropPathGradientOp, ratio);

} // namespace dragon

#endif // DRAGON_OPERATORS_ACTIVATION_DROP_PATH_OP_H_
