#include "dragon/core/workspace.h"
#include "dragon/operators/math/elementwise_ops.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void AxpbyOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  auto* x = X.template data<T, Context>();
  auto* y = Y->ReshapeLike(X)->template mutable_data<T, Context>();
  if (beta_ == 1.f) {
    if (alpha_ == 1.f) {
      math::Add(X.count(), x, y, y, ctx());
    } else {
      math::Axpy(X.count(), alpha_, x, y, ctx());
    }
  } else {
    if (alpha_ == 0.f) {
      math::Scale(X.count(), beta_, y, y, ctx());
    } else {
      math::Axpby(X.count(), alpha_, x, beta_, y, ctx());
    }
  }
}

template <class Context>
void AxpbyOp<Context>::RunOnDevice() {
  DispatchHelper<MathTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU(Axpby);
#ifdef USE_CUDA
DEPLOY_CUDA(Axpby);
#endif

OPERATOR_SCHEMA(Axpby)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

NO_GRADIENT(Axpby);

} // namespace dragon
