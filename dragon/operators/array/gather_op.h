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

#ifndef DRAGON_OPERATORS_ARRAY_GATHER_OP_H_
#define DRAGON_OPERATORS_ARRAY_GATHER_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class GatherOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(GatherOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Generic>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class GatherGradientOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(GatherGradientOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(1));
  }

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class GatherElementsOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(GatherElementsOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Generic>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class GatherElementsGradientOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(GatherElementsGradientOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(1));
  }

  template <typename T>
  void DoRunWithType();
};

#ifdef USE_MPS
template <class Context>
class MPSGatherOp final : public Operator<Context> {
 public:
  MPSGatherOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    graph_ = MPSCreateGraph();
  }
  USE_OPERATOR_FUNCTIONS;

  ~MPSGatherOp() {
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
class MPSGatherGradientOp final : public Operator<Context> {
 public:
  MPSGatherGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    graph_ = MPSCreateGraph();
  }
  USE_OPERATOR_FUNCTIONS;

  ~MPSGatherGradientOp() {
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
class MPSGatherElementsOp final : public Operator<Context> {
 public:
  MPSGatherElementsOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    graph_ = MPSCreateGraph();
  }
  USE_OPERATOR_FUNCTIONS;

  ~MPSGatherElementsOp() {
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
class MPSGatherElementsGradientOp final : public Operator<Context> {
 public:
  MPSGatherElementsGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    graph_ = MPSCreateGraph();
  }
  USE_OPERATOR_FUNCTIONS;

  ~MPSGatherElementsGradientOp() {
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
class CNNLGatherOp : public Operator<Context> {
 public:
  CNNLGatherOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    CNNLCreateTensorDesc(&input_desc_);
    CNNLCreateTensorDesc(&index_desc_);
    CNNLCreateTensorDesc(&output_desc_);
  }
  USE_OPERATOR_FUNCTIONS;

  ~CNNLGatherOp() {
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
class CNNLGatherGradientOp final : public CNNLGatherOp<Context> {
 public:
  CNNLGatherGradientOp(const OperatorDef& def, Workspace* ws)
      : CNNLGatherOp<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(1));
  }

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class CNNLGatherElementsOp : public Operator<Context> {
 public:
  CNNLGatherElementsOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    CNNLCreateTensorDesc(&input_desc_);
    CNNLCreateTensorDesc(&index_desc_);
    CNNLCreateTensorDesc(&output_desc_);
  }
  USE_OPERATOR_FUNCTIONS;

  ~CNNLGatherElementsOp() {
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
class CNNLGatherElementsGradientOp final
    : public CNNLGatherElementsOp<Context> {
 public:
  CNNLGatherElementsGradientOp(const OperatorDef& def, Workspace* ws)
      : CNNLGatherElementsOp<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(1));
  }

  template <typename T>
  void DoRunWithType();
};
#endif

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_GATHER_OP_H_
