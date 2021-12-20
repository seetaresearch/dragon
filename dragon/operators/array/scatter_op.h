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

#ifndef DRAGON_OPERATORS_ARRAY_SCATTER_OP_H_
#define DRAGON_OPERATORS_ARRAY_SCATTER_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class ScatterElementsOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(ScatterElementsOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Generic>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class ScatterElementsGradientOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(ScatterElementsGradientOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(1));
  }

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class ScatterAddOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(ScatterAddOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

  template <typename T>
  void DoRunWithTypeAndCast();
};

template <class Context>
class ScatterAddGradientOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(ScatterAddGradientOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(1));
  }

  template <typename T>
  void DoRunWithType();
};

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_SCATTER_OP_H_
