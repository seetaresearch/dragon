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

#ifndef DRAGON_OPERATORS_ARRAY_SLICE_OP_H_
#define DRAGON_OPERATORS_ARRAY_SLICE_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class SliceOp final : public Operator<Context> {
 public:
  SliceOp(const OperatorDef& def, Workspace* ws) : Operator<Context>(def, ws) {
    INITIALIZE_OP_REPEATED_ARG(int64_t, starts);
    INITIALIZE_OP_REPEATED_ARG(int64_t, sizes);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Generic>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_OP_REPEATED_ARG(int64_t, starts);
  DECLARE_OP_REPEATED_ARG(int64_t, sizes);
};

template <class Context>
class SliceGradientOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(SliceGradientOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

#ifdef USE_MLU
template <class Context>
class CNNLSliceOp final : public Operator<Context> {
 public:
  CNNLSliceOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    INITIALIZE_OP_REPEATED_ARG(int64_t, starts);
    INITIALIZE_OP_REPEATED_ARG(int64_t, sizes);
    CNNLCreateTensorDesc(&input_desc_);
    CNNLCreateTensorDesc(&output_desc_);
  }
  USE_OPERATOR_FUNCTIONS;

  ~CNNLSliceOp() {
    CNNLDestroyTensorDesc(input_desc_);
    CNNLDestroyTensorDesc(output_desc_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Generic>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_OP_REPEATED_ARG(int64_t, starts);
  DECLARE_OP_REPEATED_ARG(int64_t, sizes);
  cnnlTensorDescriptor_t input_desc_, output_desc_;
};

template <class Context>
class CNNLSliceGradientOp final : public Operator<Context> {
 public:
  CNNLSliceGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    CNNLCreateTensorDesc(&input_desc_);
    CNNLCreateTensorDesc(&output_desc_);
  }
  USE_OPERATOR_FUNCTIONS;

  ~CNNLSliceGradientOp() {
    CNNLDestroyTensorDesc(input_desc_);
    CNNLDestroyTensorDesc(output_desc_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  cnnlTensorDescriptor_t input_desc_, output_desc_;
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_SLICE_OP_H_
