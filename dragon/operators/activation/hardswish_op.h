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

#ifndef DRAGON_OPERATORS_ACTIVATION_HARDSWISH_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_HARDSWISH_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class HardSwishOp : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(HardSwishOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class HardSwishGradientOp : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(HardSwishGradientOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

#ifdef USE_MLU
template <class Context>
class CNNLHardSwishOp : public Operator<Context> {
 public:
  CNNLHardSwishOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    CNNLCreateTensorDesc(&input_desc_);
    CNNL_CHECK(cnnlCreateActivationDescriptor(&act_desc_));
    CNNL_CHECK(cnnlSetActivationDescriptor_v6(
        act_desc_,
        CNNL_ACTIVATION_HARDSWISH,
        CNNL_ACTIVATION_FAST,
        CNNL_PROPAGATE_NAN,
        0.f,
        0,
        1.f, // gamma
        1.f, // scale
        true,
        false));
  }
  USE_OPERATOR_FUNCTIONS;

  ~CNNLHardSwishOp() {
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
class CNNLHardSwishGradientOp final : public CNNLHardSwishOp<Context> {
 public:
  CNNLHardSwishGradientOp(const OperatorDef& def, Workspace* ws)
      : CNNLHardSwishOp<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_ACTIVATION_HARDSWISH_OP_H_
