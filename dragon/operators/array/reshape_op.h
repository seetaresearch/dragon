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

#ifndef DRAGON_OPERATORS_ARRAY_RESHAPE_OP_H_
#define DRAGON_OPERATORS_ARRAY_RESHAPE_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class ReshapeGradientOpBase : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(ReshapeGradientOpBase);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    auto &dY = Input(0), *dX = Output(0);
    dX->ReshapeLike(Input("X_spec"))->CopyFrom(dY, ctx());
  }
};

template <class Context>
class IdentityOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(IdentityOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    for (int i = 0; i < InputSize(); ++i) {
      auto &X = Input(i), *Y = Output(i);
      Y->ReshapeLike(X)->CopyFrom(X, ctx());
    }
  }
};

template <class Context>
class ReshapeOp final : public Operator<Context> {
 public:
  ReshapeOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    INITIALIZE_OP_REPEATED_ARG(int64_t, dims);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

 protected:
  DECLARE_OP_REPEATED_ARG(int64_t, dims);
};

template <class Context>
class FlattenOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(FlattenOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;
};

template <class Context>
class SqueezeOp final : public Operator<Context> {
 public:
  SqueezeOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws), axes_(OP_REPEATED_ARG(int64_t, "axes")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

 protected:
  vec64_t axes_;
};

template <class Context>
class UnsqueezeOp final : public Operator<Context> {
 public:
  UnsqueezeOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws), axes_(OP_REPEATED_ARG(int64_t, "axes")) {}
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

DEFINE_GRADIENT_OP(Identity);
DEFINE_GRADIENT_OP(Reshape);
DEFINE_GRADIENT_OP(Flatten);
DEFINE_GRADIENT_OP(Squeeze);
DEFINE_GRADIENT_OP(Unsqueeze);
#undef DEFINE_GRADIENT_OP

DEFINE_OP_REPEATED_ARG(int64_t, ReshapeOp, dims);

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_RESHAPE_OP_H_
