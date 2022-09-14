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

#ifndef DRAGON_OPERATORS_MATH_TOPK_OP_H_
#define DRAGON_OPERATORS_MATH_TOPK_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class TopKOp final : public Operator<Context> {
 public:
  TopKOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        k_(OP_SINGLE_ARG(int64_t, "k", 1)),
        largest_(OP_SINGLE_ARG(int64_t, "largest", 1)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Numerical>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t k_, largest_;
};

#ifdef USE_MPS

template <class Context>
class MPSTopKOp final : public Operator<Context> {
 public:
  MPSTopKOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        k_(OP_SINGLE_ARG(int64_t, "k", 1)),
        largest_(OP_SINGLE_ARG(int64_t, "largest", 1)) {
    graph_ = MPSCreateGraph();
  }
  USE_OPERATOR_FUNCTIONS;

  ~MPSTopKOp() {
    NSReleaseObject(graph_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Numerical>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t k_, largest_;
  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
};

#endif // USE_MPS

} // namespace dragon

#endif // DRAGON_OPERATORS_MATH_TOPK_OP_H_
