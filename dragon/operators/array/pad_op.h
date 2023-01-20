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

#ifndef DRAGON_OPERATORS_ARRAY_PAD_OP_H_
#define DRAGON_OPERATORS_ARRAY_PAD_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class PadOp final : public Operator<Context> {
 public:
  PadOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        value_(OP_SINGLE_ARG(float, "value", 0.f)),
        mode_(OP_SINGLE_ARG(string, "mode", "CONSTANT")) {
    INITIALIZE_OP_REPEATED_ARG(int64_t, pads);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Generic>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  float value_;
  string mode_;
  DECLARE_OP_REPEATED_ARG(int64_t, pads);
};

template <class Context>
class PadGradientOp final : public Operator<Context> {
 public:
  PadGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        mode_(OP_SINGLE_ARG(string, "mode", "CONSTANT")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  string mode_;
};

#ifdef USE_MPS
template <class Context>
class MPSPadOp final : public Operator<Context> {
 public:
  MPSPadOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        value_(OP_SINGLE_ARG(float, "value", 0.f)),
        mode_(OP_SINGLE_ARG(string, "mode", "CONSTANT")) {
    graph_ = MPSCreateGraph();
    INITIALIZE_OP_REPEATED_ARG(int64_t, pads);
  }
  USE_OPERATOR_FUNCTIONS;

  ~MPSPadOp() {
    NSReleaseObject(graph_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Generic>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  float value_;
  string mode_;
  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
  DECLARE_OP_REPEATED_ARG(int64_t, pads);
};

template <class Context>
class MPSPadGradientOp final : public Operator<Context> {
 public:
  MPSPadGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        value_(OP_SINGLE_ARG(float, "value", 0.f)),
        mode_(OP_SINGLE_ARG(string, "mode", "CONSTANT")) {
    graph_ = MPSCreateGraph();
  }
  USE_OPERATOR_FUNCTIONS;

  ~MPSPadGradientOp() {
    NSReleaseObject(graph_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  float value_;
  string mode_;
  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
};
#endif // USE_MPS

#ifdef USE_MLU
template <class Context>
class CNNLPadOp final : public Operator<Context> {
 public:
  CNNLPadOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        value_(OP_SINGLE_ARG(float, "value", 0.f)),
        mode_(OP_SINGLE_ARG(string, "mode", "CONSTANT")) {
    INITIALIZE_OP_REPEATED_ARG(int64_t, pads);
    CNNLCreateTensorDesc(&input_desc_);
    CNNLCreateTensorDesc(&output_desc_);
  }
  USE_OPERATOR_FUNCTIONS;

  ~CNNLPadOp() {
    CNNLDestroyTensorDesc(input_desc_);
    CNNLDestroyTensorDesc(output_desc_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Generic>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  float value_;
  string mode_;
  DECLARE_OP_REPEATED_ARG(int64_t, pads);
  cnnlTensorDescriptor_t input_desc_, output_desc_;
};

template <class Context>
class CNNLPadGradientOp final : public Operator<Context> {
 public:
  CNNLPadGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        mode_(OP_SINGLE_ARG(string, "mode", "CONSTANT")) {
    CNNLCreateTensorDesc(&input_desc_);
    CNNLCreateTensorDesc(&output_desc_);
  }
  USE_OPERATOR_FUNCTIONS;

  ~CNNLPadGradientOp() {
    CNNLDestroyTensorDesc(input_desc_);
    CNNLDestroyTensorDesc(output_desc_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  string mode_;
  cnnlTensorDescriptor_t input_desc_, output_desc_;
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_PAD_OP_H_
