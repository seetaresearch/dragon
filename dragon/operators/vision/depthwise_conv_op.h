/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *    <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_VISION_DEPTHWISE_CONV_OP_H_
#define DRAGON_OPERATORS_VISION_DEPTHWISE_CONV_OP_H_

#include "dragon/operators/vision/conv_op.h"

namespace dragon {

template <class Context>
class DepthwiseConv2dOp : public ConvOpBase<Context> {
 public:
  DepthwiseConv2dOp(const OperatorDef& def, Workspace* ws)
      : ConvOpBase<Context>(def, ws) {
    Setup(2);
  }
  USE_OPERATOR_FUNCTIONS;
  USE_CONVOLUTION_FUNCTIONS;

  bool Transposed() override {
    return false;
  }

  bool HasBias() override {
    return InputSize() > 2;
  }

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class DepthwiseConv2dGradientOp : public DepthwiseConv2dOp<Context> {
 public:
  DepthwiseConv2dGradientOp(const OperatorDef& def, Workspace* ws)
      : DepthwiseConv2dOp<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;
  USE_CONVOLUTION_FUNCTIONS;

  bool HasBias() override {
    return Output(2)->has_name();
  }

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

#ifdef USE_CUDNN

template <class Context>
class CuDNNDepthwiseConv2dOp final : public DepthwiseConv2dOp<Context> {
 public:
  CuDNNDepthwiseConv2dOp(const OperatorDef& def, Workspace* ws)
      : DepthwiseConv2dOp<Context>(def, ws) {
    CuDNNCreateTensorDesc(&bias_desc_);
    CuDNNCreateTensorDesc(&output_desc_);
  }
  USE_OPERATOR_FUNCTIONS;
  USE_CONVOLUTION_FUNCTIONS;

  ~CuDNNDepthwiseConv2dOp() {
    CuDNNDestroyTensorDesc(&bias_desc_);
    CuDNNDestroyTensorDesc(&output_desc_);
  }

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  cudnnTensorDescriptor_t bias_desc_;
  cudnnTensorDescriptor_t output_desc_;
};

template <class Context>
class CuDNNDepthwiseConv2dGradientOp final : public Conv2dGradientOp<Context> {
 public:
  CuDNNDepthwiseConv2dGradientOp(const OperatorDef& def, Workspace* ws)
      : Conv2dGradientOp<Context>(def, ws) {
    CuDNNCreateTensorDesc(&bias_desc_);
    CuDNNCreateTensorDesc(&input_desc_);
  }
  USE_OPERATOR_FUNCTIONS;
  USE_CONVOLUTION_FUNCTIONS;

  ~CuDNNDepthwiseConv2dGradientOp() {
    CuDNNDestroyTensorDesc(&bias_desc_);
    CuDNNDestroyTensorDesc(&input_desc_);
  }

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  cudnnTensorDescriptor_t bias_desc_;
  cudnnTensorDescriptor_t input_desc_;
};

#endif // USE_CUDNN

} // namespace dragon

#endif // DRAGON_OPERATORS_VISION_DEPTHWISE_CONV_OP_H_
