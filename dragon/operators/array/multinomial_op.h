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

#ifndef DRAGON_OPERATORS_ARRAY_MULTINOMIAL_OP_H_
#define DRAGON_OPERATORS_ARRAY_MULTINOMIAL_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class MultinomialOp final : public Operator<Context> {
 public:
  MultinomialOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        eps_(OpArg<float>("eps", 0.f)),
        normalize_(OpArg<int64_t>("normalize", 0)),
        num_samples_(OpArg<int64_t>("num_samples", 1)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  float eps_;
  int64_t normalize_, num_samples_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_MULTINOMIAL_OP_H_
