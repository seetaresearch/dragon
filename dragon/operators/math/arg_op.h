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

#ifndef DRAGON_OPERATORS_MATH_ARG_OP_H_
#define DRAGON_OPERATORS_MATH_ARG_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class ArgMaxOp final : public Operator<Context> {
 public:
  ArgMaxOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        keep_dims_(OP_SINGLE_ARG(int64_t, "keepdims", 0)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Numerical>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t keep_dims_;
};

template <class Context>
class ArgMinOp final : public Operator<Context> {
 public:
  ArgMinOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        keep_dims_(OP_SINGLE_ARG(int64_t, "keepdims", 0)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Numerical>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t keep_dims_;
};

#ifdef USE_MPS

template <class Context>
class MPSArgMaxOp final : public Operator<Context> {
 public:
  MPSArgMaxOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        keep_dims_(OP_SINGLE_ARG(int64_t, "keepdims", 0)) {
    graph_ = MPSCreateGraph();
  }
  USE_OPERATOR_FUNCTIONS;

  ~MPSArgMaxOp() {
    NSReleaseObject(graph_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Numerical>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t keep_dims_;
  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
};

template <class Context>
class MPSArgMinOp final : public Operator<Context> {
 public:
  MPSArgMinOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        keep_dims_(OP_SINGLE_ARG(int64_t, "keepdims", 0)) {
    graph_ = MPSCreateGraph();
  }
  USE_OPERATOR_FUNCTIONS;

  ~MPSArgMinOp() {
    NSReleaseObject(graph_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Numerical>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t keep_dims_;
  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
};

#endif // USE_MPS

} // namespace dragon

#endif // DRAGON_OPERATORS_MATH_ARG_OP_H_
