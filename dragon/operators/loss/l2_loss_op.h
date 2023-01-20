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

#ifndef DRAGON_OPERATORS_LOSS_L2_LOSS_OP_H_
#define DRAGON_OPERATORS_LOSS_L2_LOSS_OP_H_

#include "dragon/core/operator.h"
#include "dragon/operators/math/reduce_op_impl_cnnl.h"

namespace dragon {

template <class Context>
class L2LossOp final : public Operator<Context> {
 public:
  L2LossOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        reduction_(OP_SINGLE_ARG(string, "reduction", "MEAN")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  string reduction_;
};

template <class Context>
class L2LossGradientOp final : public Operator<Context> {
 public:
  L2LossGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        reduction_(OP_SINGLE_ARG(string, "reduction", "MEAN")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  string reduction_;
};

#ifdef USE_MLU
template <class Context>
class CNNLL2LossOp final : public Operator<Context> {
 public:
  CNNLL2LossOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        reduction_(OP_SINGLE_ARG(string, "reduction", "MEAN")) {
    reduce_impl_.SetReducer(CNNL_REDUCE_ADD);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  string reduction_;
  CNNLReduceOpImpl reduce_impl_;
};

template <class Context>
class CNNLL2LossGradientOp final : public Operator<Context> {
 public:
  CNNLL2LossGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        reduction_(OP_SINGLE_ARG(string, "reduction", "MEAN")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  string reduction_;
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_LOSS_L2_LOSS_OP_H_
