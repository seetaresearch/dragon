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

#ifdef USE_MLU
template <class Context>
class CNNLResizeOp final : public ResizeOpBase<Context> {
 public:
  CNNLResizeOp(const OperatorDef& def, Workspace* ws)
      : ResizeOpBase<Context>(def, ws) {
    CHECK_EQ(data_format(), "NHWC");
    CNNLCreateTensorDesc(&input_desc_);
    CNNLCreateTensorDesc(&output_desc_);
    CNNL_CHECK(cnnlCreateInterpDescriptor(&resize_desc_));
  }
  USE_OPERATOR_FUNCTIONS;
  USE_RESIZE_FUNCTIONS;

  ~CNNLResizeOp() {
    CNNLDestroyTensorDesc(input_desc_);
    CNNLDestroyTensorDesc(output_desc_);
    CNNL_CHECK(cnnlDestroyInterpDescriptor(resize_desc_));
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Numerical>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  cnnlInterpDescriptor_t resize_desc_;
  cnnlTensorDescriptor_t input_desc_, output_desc_;
};

template <class Context>
class CNNLResizeGradientOp final : public ResizeOpBase<Context> {
 public:
  CNNLResizeGradientOp(const OperatorDef& def, Workspace* ws)
      : ResizeOpBase<Context>(def, ws) {
    CHECK_EQ(data_format(), "NHWC");
    CNNLCreateTensorDesc(&input_desc_);
    CNNLCreateTensorDesc(&output_desc_);
    CNNL_CHECK(cnnlCreateInterpDescriptor(&resize_desc_));
  }
  USE_OPERATOR_FUNCTIONS;
  USE_RESIZE_FUNCTIONS;

  ~CNNLResizeGradientOp() {
    CNNLDestroyTensorDesc(input_desc_);
    CNNLDestroyTensorDesc(output_desc_);
    CNNL_CHECK(cnnlDestroyInterpDescriptor(resize_desc_));
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  cnnlInterpDescriptor_t resize_desc_;
  cnnlTensorDescriptor_t input_desc_, output_desc_;
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_VISION_RESIZE_OP_H_
