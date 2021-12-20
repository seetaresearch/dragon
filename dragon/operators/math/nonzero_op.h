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

#ifndef DRAGON_OPERATORS_MATH_NONZERO_OP_H_
#define DRAGON_OPERATORS_MATH_NONZERO_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class NonZeroOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(NonZeroOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

} // namespace dragon

#endif // DRAGON_OPERATORS_MATH_NONZERO_OP_H_
