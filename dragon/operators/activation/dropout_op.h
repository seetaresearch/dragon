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

#ifndef DRAGON_OPERATORS_ACTIVATION_DROPOUT_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_DROPOUT_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class DropoutOp : public Operator<Context> {
 public:
  DropoutOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    INITIALIZE_OP_SINGLE_ARG(float, ratio, 0.5f);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_OP_SINGLE_ARG(float, ratio);
};

template <class Context>
class DropoutGradientOp : public Operator<Context> {
 public:
  DropoutGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    INITIALIZE_OP_SINGLE_ARG(float, ratio, 0.5f);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_OP_SINGLE_ARG(float, ratio);
};

DEFINE_OP_SINGLE_ARG(float, DropoutOp, ratio);
DEFINE_OP_SINGLE_ARG(float, DropoutGradientOp, ratio);

#ifdef USE_CUDNN

template <class Context>
class CuDNNDropoutOp final : public DropoutOp<Context> {
 public:
  CuDNNDropoutOp(const OperatorDef& def, Workspace* ws)
      : DropoutOp<Context>(def, ws),
        states_initialized_(false),
        seed_(DEFAULT_RNG_SEED) {
    CuDNNCreateTensorDesc(&input_desc_);
    CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropout_desc_));
  }
  USE_OPERATOR_FUNCTIONS;

  ~CuDNNDropoutOp() {
    CuDNNDestroyTensorDesc(&input_desc_);
    CUDNN_CHECK(cudnnDestroyDropoutDescriptor(dropout_desc_));
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  bool states_initialized_;
  cudnnTensorDescriptor_t input_desc_;
  cudnnDropoutDescriptor_t dropout_desc_;
  unsigned long long seed_;
};

template <class Context>
class CuDNNDropoutGradientOp final : public DropoutGradientOp<Context> {
 public:
  CuDNNDropoutGradientOp(const OperatorDef& def, Workspace* ws)
      : DropoutGradientOp<Context>(def, ws),
        states_initialized_(false),
        seed_(DEFAULT_RNG_SEED) {
    CuDNNCreateTensorDesc(&input_desc_);
    CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropout_desc_));
  }
  USE_OPERATOR_FUNCTIONS;

  ~CuDNNDropoutGradientOp() {
    CuDNNDestroyTensorDesc(&input_desc_);
    CUDNN_CHECK(cudnnDestroyDropoutDescriptor(dropout_desc_));
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  bool states_initialized_;
  cudnnTensorDescriptor_t input_desc_;
  cudnnDropoutDescriptor_t dropout_desc_;
  unsigned long long seed_;
};

#endif // USE_CUDNN

#ifdef USE_MPS

template <class Context>
class MPSDropoutOp final : public Operator<Context> {
 public:
  MPSDropoutOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    graph_ = MPSCreateGraph();
    for (int i = 0; i < 50; ++i) {
      graph_seeds_.push(int64_t((*ctx()->rand_generator())()));
    }
    INITIALIZE_OP_SINGLE_ARG(float, ratio, 0.5f);
  }
  USE_OPERATOR_FUNCTIONS;

  ~MPSDropoutOp() {
    NSReleaseObject(graph_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_OP_SINGLE_ARG(float, ratio);
  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
  std::queue<int64_t> graph_seeds_;
};

DEFINE_OP_SINGLE_ARG(float, MPSDropoutOp, ratio);

#endif // USE_MPS

} // namespace dragon

#endif // DRAGON_OPERATORS_ACTIVATION_DROPOUT_OP_H_
