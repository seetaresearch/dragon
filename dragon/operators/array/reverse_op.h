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

#ifndef DRAGON_OPERATORS_ARRAY_REVERSE_OP_H_
#define DRAGON_OPERATORS_ARRAY_REVERSE_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class ReverseOp final : public Operator<Context> {
 public:
  ReverseOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws), axes_(OP_REPEATED_ARG(int64_t, "axes")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Generic>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  vec64_t axes_;
};

#ifdef USE_MPS

template <class Context>
class MPSReverseOp final : public Operator<Context> {
 public:
  MPSReverseOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws), axes_(OP_REPEATED_ARG(int64_t, "axes")) {
    graph_ = MPSCreateGraph();
  }
  USE_OPERATOR_FUNCTIONS;

  ~MPSReverseOp() {
    NSReleaseObject(graph_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Generic>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  vec64_t axes_;
  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
};

#endif

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_REVERSE_OP_H_
