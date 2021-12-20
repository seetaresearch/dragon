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

#ifndef DRAGON_OPERATORS_ARRAY_CONCAT_OP_H_
#define DRAGON_OPERATORS_ARRAY_CONCAT_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class ConcatOp : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(ConcatOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Generic>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class ConcatGradientOp : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(ConcatGradientOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_CONCAT_OP_H_
