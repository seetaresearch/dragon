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

#ifndef DRAGON_OPERATORS_ARRAY_SCATTER_OP_H_
#define DRAGON_OPERATORS_ARRAY_SCATTER_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class ScatterElementsOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(ScatterElementsOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Generic>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class ScatterElementsGradientOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(ScatterElementsGradientOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(1));
  }

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class ScatterAddOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(ScatterAddOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    auto& X = Input(0);
    if (X.template IsType<float16>()) {
      DoRunWithTypeAndCast<float16>();
    } else if (X.template IsType<bfloat16>()) {
      DoRunWithTypeAndCast<bfloat16>();
    } else if (X.template IsType<double>()) {
      DoRunWithTypeAndCast<double>();
    } else {
      using Types = dtypes::TypesBase<uint8_t, int8_t, int, int64_t, float>;
      DispatchHelper<Types>::Call(this, X);
    }
  }

  template <typename T>
  void DoRunWithType();

  template <typename T>
  void DoRunWithTypeAndCast();
};

template <class Context>
class ScatterAddGradientOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(ScatterAddGradientOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(1));
  }

  template <typename T>
  void DoRunWithType();
};

#ifdef USE_MPS
template <class Context>
class MPSScatterElementsOp final : public Operator<Context> {
 public:
  MPSScatterElementsOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    graph_ = MPSCreateGraph();
  }
  USE_OPERATOR_FUNCTIONS;

  ~MPSScatterElementsOp() {
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
};

template <class Context>
class MPSScatterElementsGradientOp final : public Operator<Context> {
 public:
  MPSScatterElementsGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    graph_ = MPSCreateGraph();
  }
  USE_OPERATOR_FUNCTIONS;

  ~MPSScatterElementsGradientOp() {
    NSReleaseObject(graph_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(1));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
};

template <class Context>
class MPSScatterAddOp final : public Operator<Context> {
 public:
  MPSScatterAddOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    graph_ = MPSCreateGraph();
  }
  USE_OPERATOR_FUNCTIONS;

  ~MPSScatterAddOp() {
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
class MPSScatterAddGradientOp final : public Operator<Context> {
 public:
  MPSScatterAddGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    graph_ = MPSCreateGraph();
  }
  USE_OPERATOR_FUNCTIONS;

  ~MPSScatterAddGradientOp() {
    NSReleaseObject(graph_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(1));
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
class CNNLScatterElementsOp : public Operator<Context> {
 public:
  CNNLScatterElementsOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    CNNLCreateTensorDesc(&input_desc_);
    CNNLCreateTensorDesc(&index_desc_);
    CNNLCreateTensorDesc(&output_desc_);
  }
  USE_OPERATOR_FUNCTIONS;

  ~CNNLScatterElementsOp() {
    CNNLDestroyTensorDesc(input_desc_);
    CNNLDestroyTensorDesc(index_desc_);
    CNNLDestroyTensorDesc(output_desc_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Generic>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  cnnlTensorDescriptor_t input_desc_, index_desc_, output_desc_;
};

template <class Context>
class CNNLScatterElementsGradientOp final
    : public CNNLScatterElementsOp<Context> {
 public:
  CNNLScatterElementsGradientOp(const OperatorDef& def, Workspace* ws)
      : CNNLScatterElementsOp<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(1));
  }

  template <typename T>
  void DoRunWithType();
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_SCATTER_OP_H_
