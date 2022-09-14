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

#ifndef DRAGON_OPERATORS_ARRAY_ONE_HOT_OP_H_
#define DRAGON_OPERATORS_ARRAY_ONE_HOT_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class OneHotOp final : public Operator<Context> {
 public:
  OneHotOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        depth_(OP_SINGLE_ARG(int64_t, "depth", -1)),
        on_value_(OP_SINGLE_ARG(float, "on_value", 1.f)),
        off_value_(OP_SINGLE_ARG(float, "off_value", 0.f)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Numerical>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t depth_;
  float on_value_, off_value_;
};

#ifdef USE_MPS

template <class Context>
class MPSOneHotOp final : public Operator<Context> {
 public:
  MPSOneHotOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        depth_(OP_SINGLE_ARG(int64_t, "depth", -1)),
        on_value_(OP_SINGLE_ARG(float, "on_value", 1.f)),
        off_value_(OP_SINGLE_ARG(float, "off_value", 0.f)) {
    graph_ = MPSCreateGraph();
  }
  USE_OPERATOR_FUNCTIONS;

  ~MPSOneHotOp() {
    NSReleaseObject(graph_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Numerical>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t depth_;
  float on_value_, off_value_;
  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
};

#endif // USE_MPS

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_ONE_HOT_OP_H_
