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

DEFINE_OP_REPEATED_ARG(int64_t, SliceOp, starts);
DEFINE_OP_REPEATED_ARG(int64_t, SliceOp, sizes);

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_SLICE_OP_H_
