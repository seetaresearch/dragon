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

DEFINE_OP_REPEATED_ARG(int64_t, PadOp, pads);

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

DEFINE_OP_REPEATED_ARG(int64_t, MPSPadOp, pads);

#endif // USE_MPS

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_PAD_OP_H_
