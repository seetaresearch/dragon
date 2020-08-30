#include "dragon/operators/generic/gradient_ops.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void GradientGenerateOp<Context>::DoRunWithType() {
  for (int i = 0; i < OutputSize(); i++) {
    auto* Y = Output(i);
    if (!Y->has_name()) continue;
    Y->ReshapeLike(Input(i));
    math::Set(
        Y->count(),
        cast::to<T>(defaults_[i]),
        Y->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
void GradientGenerateOp<Context>::RunOnDevice() {
  CHECK_EQ(InputSize(), OutputSize());
  CHECK_EQ(defaults_.size(), OutputSize());
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
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
  DispatchHelper<FloatingTensorTypes>::Call(this, *grads_[0]);
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
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
void StopGradientOp<Context>::RunOnDevice() {
  if (Output(0)->name() != Input(0).name()) {
    Output(0)->ReshapeLike(Input(0));
    Output(0)->CopyFrom(Input(0), ctx());
  }
}

DEPLOY_CPU_OPERATOR(GradientGenerate);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(GradientGenerate);
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

OPERATOR_SCHEMA(GradientGenerate)
    /* X1, ... */
    .NumInputs(1, INT_MAX)
    /* Y1, ... */
    .NumOutputs(1, INT_MAX);

OPERATOR_SCHEMA(GradientGather)
    /* X1, ... */
    .NumInputs(1, INT_MAX)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(GradientAdd)
    /* A, B */
    .NumInputs(2)
    /* Y */
    .NumOutputs(1)
    /* A => Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(StopGradient)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

NO_GRADIENT(StopGradient);

} // namespace dragon
