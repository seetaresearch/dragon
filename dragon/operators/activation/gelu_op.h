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

#ifndef DRAGON_OPERATORS_ACTIVATION_GELU_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_GELU_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class GeluOp : public Operator<Context> {
 public:
  GeluOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        approximate_(OP_SINGLE_ARG(int64_t, "approximate", 0)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t approximate_;
};

template <class Context>
class GeluGradientOp : public Operator<Context> {
 public:
  GeluGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        approximate_(OP_SINGLE_ARG(int64_t, "approximate", 0)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t approximate_;
};

#ifdef USE_MLU
template <class Context>
class CNNLGeluOp : public Operator<Context> {
 public:
  CNNLGeluOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        approximate_(OP_SINGLE_ARG(int64_t, "approximate", 0)) {
    CNNLCreateTensorDesc(&input_desc_);
    CNNL_CHECK(cnnlCreateActivationDescriptor(&act_desc_));
    CNNL_CHECK(cnnlSetActivationDescriptor_v6(
        act_desc_,
        CNNL_ACTIVATION_GELU,
        CNNL_ACTIVATION_FAST,
        CNNL_PROPAGATE_NAN,
        0.f,
        0,
        1.f, // gamma
        1.f, // scale
        true,
        approximate_ > 0));
  }
  USE_OPERATOR_FUNCTIONS;

  ~CNNLGeluOp() {
    CNNLDestroyTensorDesc(input_desc_);
    CNNL_CHECK(cnnlDestroyActivationDescriptor(act_desc_));
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t approximate_;
  cnnlTensorDescriptor_t input_desc_;
  cnnlActivationDescriptor_t act_desc_;
};

template <class Context>
class CNNLGeluGradientOp final : public CNNLGeluOp<Context> {
 public:
  CNNLGeluGradientOp(const OperatorDef& def, Workspace* ws)
      : CNNLGeluOp<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_ACTIVATION_GELU_OP_H_
