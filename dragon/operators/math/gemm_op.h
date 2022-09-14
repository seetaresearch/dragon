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

#ifndef DRAGON_OPERATORS_MATH_GEMM_OP_H_
#define DRAGON_OPERATORS_MATH_GEMM_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class GemmOp final : public Operator<Context> {
 public:
  GemmOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        n_(OP_SINGLE_ARG(int64_t, "n", 0)),
        alpha_(OP_SINGLE_ARG(float, "alpha", 1.f)),
        beta_(OP_SINGLE_ARG(float, "beta", 1.f)),
        transA_(OP_SINGLE_ARG(int64_t, "transA", 0)),
        transB_(OP_SINGLE_ARG(int64_t, "transB", 0)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  float alpha_, beta_;
  int64_t n_, transA_, transB_;
};

template <class Context>
class GemmGradientOp final : public Operator<Context> {
 public:
  GemmGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        alpha_(OP_SINGLE_ARG(float, "alpha", 1.f)),
        beta_(OP_SINGLE_ARG(float, "beta", 1.f)),
        transA_(OP_SINGLE_ARG(int64_t, "transA", 0)),
        transB_(OP_SINGLE_ARG(int64_t, "transB", 0)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  float alpha_, beta_;
  int64_t transA_, transB_;
};

#ifdef USE_MPS

template <class Context>
class MPSGemmOp final : public Operator<Context> {
 public:
  MPSGemmOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        n_(OP_SINGLE_ARG(int64_t, "n", 0)),
        alpha_(OP_SINGLE_ARG(float, "alpha", 1.f)),
        beta_(OP_SINGLE_ARG(float, "beta", 1.f)),
        transA_(OP_SINGLE_ARG(int64_t, "transA", 0)),
        transB_(OP_SINGLE_ARG(int64_t, "transB", 0)) {
    graph_ = MPSCreateGraph();
  }
  USE_OPERATOR_FUNCTIONS;

  ~MPSGemmOp() {
    NSReleaseObject(graph_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  float alpha_, beta_;
  int64_t n_, transA_, transB_;
  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
};

template <class Context>
class MPSGemmGradientOp final : public Operator<Context> {
 public:
  MPSGemmGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        alpha_(OP_SINGLE_ARG(float, "alpha", 1.f)),
        beta_(OP_SINGLE_ARG(float, "beta", 1.f)),
        transA_(OP_SINGLE_ARG(int64_t, "transA", 0)),
        transB_(OP_SINGLE_ARG(int64_t, "transB", 0)) {
    graph_ = MPSCreateGraph();
  }
  USE_OPERATOR_FUNCTIONS;

  ~MPSGemmGradientOp() {
    NSReleaseObject(graph_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  float alpha_, beta_;
  int64_t transA_, transB_;
  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
};

#endif // USE_MPS

} // namespace dragon

#endif // DRAGON_OPERATORS_MATH_GEMM_OP_H_
