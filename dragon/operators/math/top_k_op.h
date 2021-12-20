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

#ifndef DRAGON_OPERATORS_MATH_TOP_K_OP_H_
#define DRAGON_OPERATORS_MATH_TOP_K_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class TopKOp final : public Operator<Context> {
 public:
  TopKOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        k_(OP_SINGLE_ARG(int64_t, "k", 1)),
        largest_(OP_SINGLE_ARG(int64_t, "largest", 1)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t k_, largest_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_MATH_TOP_K_OP_H_
