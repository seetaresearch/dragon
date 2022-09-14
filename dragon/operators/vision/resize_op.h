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

#ifndef DRAGON_OPERATORS_VISION_RESIZE_OP_H_
#define DRAGON_OPERATORS_VISION_RESIZE_OP_H_

#include "dragon/operators/vision/resize_op_base.h"

namespace dragon {

template <class Context>
class ResizeOp final : public ResizeOpBase<Context> {
 public:
  ResizeOp(const OperatorDef& def, Workspace* ws)
      : ResizeOpBase<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;
  USE_RESIZE_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Numerical>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class ResizeGradientOp final : public ResizeOpBase<Context> {
 public:
  ResizeGradientOp(const OperatorDef& def, Workspace* ws)
      : ResizeOpBase<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;
  USE_RESIZE_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

#ifdef USE_MPS

template <class Context>
class MPSResizeOp final : public ResizeOpBase<Context> {
 public:
  MPSResizeOp(const OperatorDef& def, Workspace* ws)
      : ResizeOpBase<Context>(def, ws) {
    graph_ = MPSCreateGraph();
  }
  USE_OPERATOR_FUNCTIONS;
  USE_RESIZE_FUNCTIONS;

  ~MPSResizeOp() {
    NSReleaseObject(graph_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Numerical>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
};

template <class Context>
class MPSResizeGradientOp final : public ResizeOpBase<Context> {
 public:
  MPSResizeGradientOp(const OperatorDef& def, Workspace* ws)
      : ResizeOpBase<Context>(def, ws) {
    graph_ = MPSCreateGraph();
  }
  USE_OPERATOR_FUNCTIONS;
  USE_RESIZE_FUNCTIONS;

  ~MPSResizeGradientOp() {
    NSReleaseObject(graph_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
};

#endif // USE_MPS

} // namespace dragon

#endif // DRAGON_OPERATORS_VISION_RESIZE_OP_H_
