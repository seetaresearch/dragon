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

#ifndef DRAGON_OPERATORS_ARRAY_TILE_OP_H_
#define DRAGON_OPERATORS_ARRAY_TILE_OP_H_

#include "dragon/core/operator.h"
#include "dragon/operators/math/reduce_op_impl_cnnl.h"

namespace dragon {

template <class Context>
class TileOp final : public Operator<Context> {
 public:
  TileOp(const OperatorDef& def, Workspace* ws) : Operator<Context>(def, ws) {
    INITIALIZE_OP_REPEATED_ARG(int64_t, repeats);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Generic>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_OP_REPEATED_ARG(int64_t, repeats);
};

template <class Context>
class TileGradientOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(TileGradientOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

#ifdef USE_MPS
template <class Context>
class MPSTileOp final : public Operator<Context> {
 public:
  MPSTileOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    graph_ = MPSCreateGraph();
    INITIALIZE_OP_REPEATED_ARG(int64_t, repeats);
  }
  USE_OPERATOR_FUNCTIONS;

  ~MPSTileOp() {
    NSReleaseObject(graph_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Generic>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
  DECLARE_OP_REPEATED_ARG(int64_t, repeats);
};

template <class Context>
class MPSTileGradientOp final : public Operator<Context> {
 public:
  MPSTileGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    graph_ = MPSCreateGraph();
  }
  USE_OPERATOR_FUNCTIONS;

  ~MPSTileGradientOp() {
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

#ifdef USE_MLU
template <class Context>
class CNNLTileOp final : public Operator<Context> {
 public:
  CNNLTileOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    INITIALIZE_OP_REPEATED_ARG(int64_t, repeats);
    CNNLCreateTensorDesc(&input_desc_);
    CNNLCreateTensorDesc(&output_desc_);
  }
  USE_OPERATOR_FUNCTIONS;

  ~CNNLTileOp() {
    CNNLDestroyTensorDesc(input_desc_);
    CNNLDestroyTensorDesc(output_desc_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Generic>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_OP_REPEATED_ARG(int64_t, repeats);
  cnnlTensorDescriptor_t input_desc_, output_desc_;
};

template <class Context>
class CNNLTileGradientOp final : public Operator<Context> {
 public:
  CNNLTileGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    reduce_impl_.SetReducer(CNNL_REDUCE_ADD);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  CNNLReduceOpImpl reduce_impl_;
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_TILE_OP_H_
