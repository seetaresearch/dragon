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

#ifndef DRAGON_OPERATORS_VISION_BIAS_ADD_OP_H_
#define DRAGON_OPERATORS_VISION_BIAS_ADD_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class BiasAddOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(BiasAddOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class BiasAddGradientOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(BiasAddGradientOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

#ifdef USE_CUDNN

template <class Context>
class CuDNNBiasAddGradientOp final : public Operator<Context> {
 public:
  CuDNNBiasAddGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    CuDNNCreateTensorDesc(&bias_desc_);
    CuDNNCreateTensorDesc(&input_desc_);
  }
  USE_OPERATOR_FUNCTIONS;

  ~CuDNNBiasAddGradientOp() {
    CuDNNDestroyTensorDesc(&bias_desc_);
    CuDNNDestroyTensorDesc(&input_desc_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  cudnnTensorDescriptor_t bias_desc_;
  cudnnTensorDescriptor_t input_desc_;
};

#endif // USE_CUDNN

} // namespace dragon

#endif // DRAGON_OPERATORS_VISION_BIAS_ADD_OP_H_
