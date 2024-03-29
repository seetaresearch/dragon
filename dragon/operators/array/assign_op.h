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

#ifndef DRAGON_OPERATORS_ARRAY_ASSIGN_OP_H_
#define DRAGON_OPERATORS_ARRAY_ASSIGN_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class AssignOp final : public Operator<Context> {
 public:
  AssignOp(const OperatorDef& def, Workspace* ws) : Operator<Context>(def, ws) {
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

#ifdef USE_MLU
template <class Context>
class CNNLAssignOp final : public Operator<Context> {
 public:
  CNNLAssignOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    INITIALIZE_OP_REPEATED_ARG(int64_t, starts);
    INITIALIZE_OP_REPEATED_ARG(int64_t, sizes);
    CNNLCreateTensorDesc(&input_desc_);
    CNNLCreateTensorDesc(&output_desc_);
  }
  USE_OPERATOR_FUNCTIONS;

  ~CNNLAssignOp() {
    CNNLDestroyTensorDesc(input_desc_);
    CNNLDestroyTensorDesc(output_desc_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_OP_REPEATED_ARG(int64_t, starts);
  DECLARE_OP_REPEATED_ARG(int64_t, sizes);
  cnnlTensorDescriptor_t input_desc_, output_desc_;
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_ASSIGN_OP_H_
