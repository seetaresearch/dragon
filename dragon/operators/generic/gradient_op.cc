#include "dragon/operators/generic/gradient_op.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void GradientFillOp<Context>::DoRunWithType() {
  for (int i = 0; i < OutputSize(); i++) {
    auto &X = Input(i), *Y = Output(i);
    if (!Y->has_name()) continue;
    if (i < values_.size() && values_[i] > -100.f) {
      math::Set(
          X.count(),
          convert::To<T>(values_[i]),
          Y->ReshapeLike(X)->template mutable_data<T, Context>(),
          ctx());
    } else {
      math::Copy(
          X.count(),
          X.template data<T, Context>(),
          Y->ReshapeLike(X)->template mutable_data<T, Context>(),
          ctx());
    }
  }
}

template <class Context>
template <typename T>
void GradientGatherOp<Context>::DoRunWithType() {
  auto* Y = Output(0)->ReshapeLike(*inputs_[0]);
  if (inputs_.size() == 1) {
    Y->CopyFrom(*inputs_[0], ctx());
    return;
  }
  const auto N = Y->count();
  auto* x = inputs_[0]->template data<T, Context>();
  auto* y = Y->template mutable_data<T, Context>();
  for (int i = 1; i < inputs_.size(); ++i) {
    CHECK_EQ(inputs_[i]->count(), N);
    math::Add(N, x, inputs_[i]->template data<T, Context>(), y, ctx());
    x = y; // InplaceAdd
  }
}

DEPLOY_CPU_OPERATOR(GradientFill);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(GradientFill);
#endif

DEPLOY_CPU_OPERATOR(GradientGather);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(GradientGather);
#endif

DEPLOY_CPU_OPERATOR(GradientStop);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(GradientStop);
#endif

OPERATOR_SCHEMA(GradientFill).NumInputs(1, INT_MAX).NumOutputs(1, INT_MAX);
OPERATOR_SCHEMA(GradientGather)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .AllowInplace([](int, int) -> bool { return true; });
OPERATOR_SCHEMA(StopGradient).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});

NO_GRADIENT(GradientStop);

} // namespace dragon
