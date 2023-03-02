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

#ifndef DRAGON_OPERATORS_ACTIVATION_SILU_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_SILU_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class SiluOp : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(SiluOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class SiluGradientOp : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(SiluGradientOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

#ifdef USE_MLU
template <class Context>
class CNNLSiluOp : public Operator<Context> {
 public:
  CNNLSiluOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    CNNLCreateTensorDesc(&input_desc_);
    CNNL_CHECK(cnnlCreateActivationDescriptor(&act_desc_));
    CNNL_CHECK(cnnlSetActivationDescriptor_v6(
        act_desc_,
        CNNL_ACTIVATION_SILU,
        CNNL_ACTIVATION_FAST,
        CNNL_NOT_PROPAGATE_NAN,
        0.f,
        0,
        1.f, // gamma
        1.f, // scale
        true,
        false));
  }
  USE_OPERATOR_FUNCTIONS;

  ~CNNLSiluOp() {
    CNNLDestroyTensorDesc(input_desc_);
    CNNL_CHECK(cnnlDestroyActivationDescriptor(act_desc_));
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  cnnlTensorDescriptor_t input_desc_;
  cnnlActivationDescriptor_t act_desc_;
};

template <class Context>
class CNNLSiluGradientOp final : public CNNLSiluOp<Context> {
 public:
  CNNLSiluGradientOp(const OperatorDef& def, Workspace* ws)
      : CNNLSiluOp<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_ACTIVATION_SILU_OP_H_
