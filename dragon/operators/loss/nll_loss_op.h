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

#ifndef DRAGON_OPERATORS_LOSS_NLL_LOSS_OP_H_
#define DRAGON_OPERATORS_LOSS_NLL_LOSS_OP_H_

#include "dragon/core/operator.h"
#include "dragon/operators/math/reduce_op_impl_cnnl.h"

namespace dragon {

template <class Context>
class NLLLossOp final : public Operator<Context> {
 public:
  NLLLossOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        ignore_index_(OP_SINGLE_ARG(int64_t, "ignore_index", INT_MAX)),
        reduction_(OP_SINGLE_ARG(string, "reduction", "VALID")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Loss>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t ignore_index_;
  string reduction_;
};

template <class Context>
class NLLLossGradientOp final : public Operator<Context> {
 public:
  NLLLossGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        ignore_index_(OP_SINGLE_ARG(int64_t, "ignore_index", INT_MAX)),
        reduction_(OP_SINGLE_ARG(string, "reduction", "VALID")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Loss>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t ignore_index_;
  string reduction_;
};

#ifdef USE_MLU
template <class Context>
class CNNLNLLLossOp : public Operator<Context> {
 public:
  CNNLNLLLossOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        ignore_index_(OP_SINGLE_ARG(int64_t, "ignore_index", INT_MAX)),
        reduction_(OP_SINGLE_ARG(string, "reduction", "MEAN")) {
    CNNLCreateTensorDesc(&input_desc_);
    CNNLCreateTensorDesc(&target_desc_);
    CNNLCreateTensorDesc(&output_desc_);
    reduce_impl_.SetReducer(CNNL_REDUCE_ADD);
  }
  USE_OPERATOR_FUNCTIONS;

  ~CNNLNLLLossOp() {
    CNNLDestroyTensorDesc(input_desc_);
    CNNLDestroyTensorDesc(target_desc_);
    CNNLDestroyTensorDesc(output_desc_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Loss>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t ignore_index_;
  string reduction_;
  cnnlTensorDescriptor_t input_desc_, target_desc_, output_desc_;
  CNNLReduceOpImpl reduce_impl_;
};

template <class Context>
class CNNLNLLLossGradientOp : public CNNLNLLLossOp<Context> {
 public:
  CNNLNLLLossGradientOp(const OperatorDef& def, Workspace* ws)
      : CNNLNLLLossOp<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Loss>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_LOSS_NLL_LOSS_OP_H_
