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

#ifndef DRAGON_OPERATORS_MATH_MATMUL_OP_H_
#define DRAGON_OPERATORS_MATH_MATMUL_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class MatMulOp final : public Operator<Context> {
 public:
  MatMulOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        transA_(OpArg<int64_t>("transA", 0)),
        transB_(OpArg<int64_t>("transB", 0)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t transA_, transB_;
};

template <class Context>
class MatMulGradientOp final : public Operator<Context> {
 public:
  MatMulGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        transA_(OpArg<int64_t>("transA", 0)),
        transB_(OpArg<int64_t>("transB", 0)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t transA_, transB_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_MATH_MATMUL_OP_H_
