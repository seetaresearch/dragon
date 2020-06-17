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

#ifndef DRAGON_OPERATORS_ARRAY_SELECT_OPS_H_
#define DRAGON_OPERATORS_ARRAY_SELECT_OPS_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class IndexSelectOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(IndexSelectOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class MaskedSelectOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(MaskedSelectOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class IndexSelectGradientOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(IndexSelectGradientOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class MaskedSelectGradientOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(MaskedSelectGradientOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_SELECT_OPS_H_
