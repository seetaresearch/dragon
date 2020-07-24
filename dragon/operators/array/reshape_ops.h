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

#ifndef DRAGON_OPERATORS_ARRAY_RESHAPE_OPS_H_
#define DRAGON_OPERATORS_ARRAY_RESHAPE_OPS_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class ReshapeGradientOpBase : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(ReshapeGradientOpBase);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    // Simply copy the dY to dX
    Output(0)->ReshapeLike(RESTORE_INPUT_SPEC(0));
    Output(0)->CopyFrom(Input(-1), ctx());
  }
};

template <class Context>
class ReshapeOp final : public Operator<Context> {
 public:
  ReshapeOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    GET_ARGS_WITH_DESC(int64_t, dims);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

 protected:
  DECLARE_ARGS_WITH_DESC(int64_t, dims);
};

template <class Context>
class FlattenOp final : public Operator<Context> {
 public:
  FlattenOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        num_axes_(OpArg<int64_t>("num_axes", -1)),
        keep_axes_(OpArg<int64_t>("keep_axes", INT_MAX)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

 protected:
  int64_t num_axes_, keep_axes_;
};

template <class Context>
class ExpandDimsOp final : public Operator<Context> {
 public:
  ExpandDimsOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws), axes_(OpArgs<int64_t>("axes")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

 protected:
  vec64_t axes_;
};

template <class Context>
class SqueezeOp final : public Operator<Context> {
 public:
  SqueezeOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws), axes_(OpArgs<int64_t>("axes")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

 protected:
  vec64_t axes_;
};

#define DEFINE_GRADIENT_OP(name)                                         \
  template <class Context>                                               \
  class name##GradientOp final : public ReshapeGradientOpBase<Context> { \
   public:                                                               \
    name##GradientOp(const OperatorDef& def, Workspace* ws)              \
        : ReshapeGradientOpBase<Context>(def, ws) {}                     \
  };

DEFINE_GRADIENT_OP(Reshape);
DEFINE_GRADIENT_OP(Flatten);
DEFINE_GRADIENT_OP(ExpandDims);
DEFINE_GRADIENT_OP(Squeeze);
#undef DEFINE_GRADIENT_OP

DEFINE_ARGS_WITH_DESC(int64_t, ReshapeOp, dims);

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_RESHAPE_OPS_H_
