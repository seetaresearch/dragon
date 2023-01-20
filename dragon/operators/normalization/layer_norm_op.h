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

#ifndef DRAGON_OPERATORS_NORMALIZATION_LAYER_NORM_OP_H_
#define DRAGON_OPERATORS_NORMALIZATION_LAYER_NORM_OP_H_

#include "dragon/operators/normalization/group_norm_op.h"

namespace dragon {

template <class Context>
class LayerNormOp final : public Operator<Context> {
 public:
  LayerNormOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        epsilon_(OP_SINGLE_ARG(double, "epsilon", 1e-5)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  double epsilon_;
};

template <class Context>
class LayerNormGradientOp final : public GroupNormGradientOp<Context> {
 public:
  LayerNormGradientOp(const OperatorDef& def, Workspace* ws)
      : GroupNormGradientOp<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  void GetBaseArguments() override {
    auto& X = Input(0);
    GET_OP_AXIS_ARG(axis, X.ndim(), -1);
    // Set dimensions
    this->N_ = X.count(0, axis);
    this->C_ = this->D_ = X.count(axis);
    this->G_ = this->S_ = 1;
    // Set data format
    this->data_format_ = "NHWC";
  }
};

#ifdef USE_MLU
template <class Context>
class CNNLLayerNormOp final : public Operator<Context> {
 public:
  CNNLLayerNormOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        epsilon_(OP_SINGLE_ARG(double, "epsilon", 1e-5)) {
    CNNLCreateTensorDesc(&input_desc_);
    CNNLCreateTensorDesc(&scale_desc_);
    CNNLCreateTensorDesc(&stats_desc_);
  }
  USE_OPERATOR_FUNCTIONS;

  ~CNNLLayerNormOp() {
    CNNLDestroyTensorDesc(input_desc_);
    CNNLDestroyTensorDesc(scale_desc_);
    CNNLDestroyTensorDesc(stats_desc_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  double epsilon_;
  cnnlTensorDescriptor_t input_desc_, scale_desc_, stats_desc_;
};

template <class Context>
class CNNLLayerNormGradientOp final : public Operator<Context> {
 public:
  CNNLLayerNormGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    CNNLCreateTensorDesc(&input_desc_);
    CNNLCreateTensorDesc(&scale_desc_);
    CNNLCreateTensorDesc(&stats_desc_);
  }
  USE_OPERATOR_FUNCTIONS;

  ~CNNLLayerNormGradientOp() {
    CNNLDestroyTensorDesc(input_desc_);
    CNNLDestroyTensorDesc(scale_desc_);
    CNNLDestroyTensorDesc(stats_desc_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  cnnlTensorDescriptor_t input_desc_, scale_desc_, stats_desc_;
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_NORMALIZATION_LAYER_NORM_OP_H_
