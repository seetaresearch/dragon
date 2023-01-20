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

#ifndef DRAGON_OPERATORS_ARRAY_TRANSPOSE_OP_H_
#define DRAGON_OPERATORS_ARRAY_TRANSPOSE_OP_H_

#include "dragon/core/operator.h"
#include "dragon/operators/array/transpose_op_impl_cnnl.h"

namespace dragon {

template <class Context>
class TransposeOp final : public Operator<Context> {
 public:
  TransposeOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    INITIALIZE_OP_REPEATED_ARG(int64_t, perm);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Generic>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_OP_REPEATED_ARG(int64_t, perm);
};

#ifdef USE_MLU
template <class Context>
class CNNLTransposeOp final : public Operator<Context> {
 public:
  CNNLTransposeOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    INITIALIZE_OP_REPEATED_ARG(int64_t, perm);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Generic>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_OP_REPEATED_ARG(int64_t, perm);
  CNNLTransposeOpImpl impl_;
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_TRANSPOSE_OP_H_
