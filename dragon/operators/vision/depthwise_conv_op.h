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

#ifndef DRAGON_OPERATORS_VISION_DEPTHWISE_CONV_OP_H_
#define DRAGON_OPERATORS_VISION_DEPTHWISE_CONV_OP_H_

#include "dragon/operators/vision/conv_op_base.h"

namespace dragon {

template <class Context>
class DepthwiseConvOp final : public ConvOpBase<Context> {
 public:
  DepthwiseConvOp(const OperatorDef& def, Workspace* ws)
      : ConvOpBase<Context>(def, ws) {
    GetBaseArguments();
  }
  USE_OPERATOR_FUNCTIONS;
  USE_CONV_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::TypesBase<float16, float>>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  bool HasBias() override {
    return InputSize() > 2;
  }
};

template <class Context>
class DepthwiseConvGradientOp final : public ConvOpBase<Context> {
 public:
  DepthwiseConvGradientOp(const OperatorDef& def, Workspace* ws)
      : ConvOpBase<Context>(def, ws) {
    GetBaseArguments();
  }
  USE_OPERATOR_FUNCTIONS;
  USE_CONV_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::TypesBase<float16, float>>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  bool HasBias() override {
    return Output(2)->has_name();
  }
};

#ifdef USE_CUDNN
template <class Context>
class CuDNNDepthwiseConvOp final : public ConvOpBase<Context> {
 public:
  CuDNNDepthwiseConvOp(const OperatorDef& def, Workspace* ws)
      : ConvOpBase<Context>(def, ws) {
    GetBaseArguments();
    CuDNNCreateTensorDesc(&bias_desc_);
    CuDNNCreateTensorDesc(&output_desc_);
  }
  USE_OPERATOR_FUNCTIONS;
  USE_CONV_FUNCTIONS;

  ~CuDNNDepthwiseConvOp() {
    CuDNNDestroyTensorDesc(bias_desc_);
    CuDNNDestroyTensorDesc(output_desc_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::TypesBase<float16, float>>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  bool HasBias() override {
    return InputSize() > 2;
  }

  cudnnTensorDescriptor_t bias_desc_;
  cudnnTensorDescriptor_t output_desc_;
};

template <class Context>
class CuDNNDepthwiseConvGradientOp final : public ConvOpBase<Context> {
 public:
  CuDNNDepthwiseConvGradientOp(const OperatorDef& def, Workspace* ws)
      : ConvOpBase<Context>(def, ws) {
    GetBaseArguments();
    CuDNNCreateTensorDesc(&bias_desc_);
    CuDNNCreateTensorDesc(&input_desc_);
  }
  USE_OPERATOR_FUNCTIONS;
  USE_CONV_FUNCTIONS;

  ~CuDNNDepthwiseConvGradientOp() {
    CuDNNDestroyTensorDesc(bias_desc_);
    CuDNNDestroyTensorDesc(input_desc_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::TypesBase<float16, float>>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  bool HasBias() override {
    return Output(2)->has_name();
  }

  cudnnTensorDescriptor_t bias_desc_;
  cudnnTensorDescriptor_t input_desc_;
};
#endif // USE_CUDNN

} // namespace dragon

#endif // DRAGON_OPERATORS_VISION_DEPTHWISE_CONV_OP_H_
