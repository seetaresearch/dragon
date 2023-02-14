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

#ifndef DRAGON_OPERATORS_MATH_AFFINE_OP_H_
#define DRAGON_OPERATORS_MATH_AFFINE_OP_H_

#include "dragon/core/operator.h"
#include "dragon/operators/math/reduce_op_impl_cnnl.h"

namespace dragon {

template <class Context>
class AffineOp final : public Operator<Context> {
 public:
  AffineOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws), axes_(OP_REPEATED_ARG(int64_t, "axes")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  vec64_t axes_;
};

template <class Context>
class AffineGradientOp final : public Operator<Context> {
 public:
  AffineGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws), axes_(OP_REPEATED_ARG(int64_t, "axes")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  vec64_t axes_;
};

#ifdef USE_MLU
template <class Context>
class CNNLAffineGradientOp final : public Operator<Context> {
 public:
  CNNLAffineGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws), axes_(OP_REPEATED_ARG(int64_t, "axes")) {
    dW_impl_.SetReducer(CNNL_REDUCE_ADD);
    dB_impl_.SetReducer(CNNL_REDUCE_ADD);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  vec64_t axes_;
  CNNLReduceOpImpl dW_impl_, dB_impl_;
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_MATH_AFFINE_OP_H_
