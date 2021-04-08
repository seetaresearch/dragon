#include "dragon/operators/generic/gradient_ops.h"
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
  auto* Y = Output(0)->ReshapeLike(*grads_[0]);
  if (grads_.size() == 1) {
    math::Copy(
        Y->count(),
        grads_[0]->template data<T, Context>(),
        Y->template mutable_data<T, Context>(),
        ctx());
  } else {
    CHECK_EQ(Y->count(), grads_[1]->count());
    auto* y = Y->template mutable_data<T, Context>();
    math::Add(
        Y->count(),
        grads_[0]->template data<T, Context>(),
        grads_[1]->template data<T, Context>(),
        y,
        ctx());
    for (int i = 2; i < grads_.size(); ++i) {
      CHECK_EQ(Y->count(), grads_[i]->count());
      math::Add(
          Y->count(), y, grads_[i]->template data<T, Context>(), y, ctx());
    }
  }
}

template <class Context>
void GradientGatherOp<Context>::RunOnDevice() {
  grads_.clear();
  for (int i = 0; i < InputSize(); i++) {
    auto* X = &Input(i);
    if (X->has_name()) {
      grads_.push_back(X);
    }
  }
  if (grads_.empty() || !Output(0)->has_name()) return;
  DispatchHelper<dtypes::Floating>::Call(this, *grads_[0]);
}

template <class Context>
template <typename T>
void GradientAddOp<Context>::DoRunWithType() {
  auto* x = Input(1).template data<T, Context>();
  auto* y = Output(0)->template mutable_data<T, Context>();
  math::Add(Output(0)->count(), y, x, y, ctx());
}

template <class Context>
void GradientAddOp<Context>::RunOnDevice() {
  CHECK_EQ(Input(0).name(), Output(0)->name())
      << "\nExcepted Input(0) and Output(0) are the same tensor.";
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

template <class Context>
void StopGradientOp<Context>::RunOnDevice() {
  if (Output(0)->name() != Input(0).name()) {
    Output(0)->ReshapeLike(Input(0));
    Output(0)->CopyFrom(Input(0), ctx());
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

DEPLOY_CPU_OPERATOR(GradientAdd);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(GradientAdd);
#endif

DEPLOY_CPU_OPERATOR(StopGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(StopGradient);
#endif

OPERATOR_SCHEMA(GradientFill).NumInputs(1, INT_MAX).NumOutputs(1, INT_MAX);
OPERATOR_SCHEMA(GradientGather).NumInputs(1, INT_MAX).NumOutputs(1);
OPERATOR_SCHEMA(GradientAdd).NumInputs(2).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(StopGradient).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});

NO_GRADIENT(StopGradient);

} // namespace dragon
