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

#ifndef DRAGON_OPERATORS_ARRAY_EXPAND_OP_H_
#define DRAGON_OPERATORS_ARRAY_EXPAND_OP_H_

#include "dragon/core/operator.h"
#include "dragon/operators/math/reduce_op_impl_cnnl.h"

namespace dragon {

template <class Context>
class ExpandOp final : public Operator<Context> {
 public:
  ExpandOp(const OperatorDef& def, Workspace* ws) : Operator<Context>(def, ws) {
    INITIALIZE_OP_REPEATED_ARG(int64_t, dims);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Generic>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_OP_REPEATED_ARG(int64_t, dims);
};

template <class Context>
class ExpandGradientOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(ExpandGradientOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

#ifdef USE_MLU
template <class Context>
class CNNLExpandOp final : public Operator<Context> {
 public:
  CNNLExpandOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    INITIALIZE_OP_REPEATED_ARG(int64_t, dims);
    CNNLCreateTensorDesc(&input_desc_);
    CNNLCreateTensorDesc(&output_desc_);
  }
  USE_OPERATOR_FUNCTIONS;

  ~CNNLExpandOp() {
    CNNLDestroyTensorDesc(input_desc_);
    CNNLDestroyTensorDesc(output_desc_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Generic>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_OP_REPEATED_ARG(int64_t, dims);
  cnnlTensorDescriptor_t input_desc_, output_desc_;
};

template <class Context>
class CNNLExpandGradientOp final : public Operator<Context> {
 public:
  CNNLExpandGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    reduce_impl_.SetReducer(CNNL_REDUCE_ADD);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  CNNLReduceOpImpl reduce_impl_;
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_EXPAND_OP_H_
