#include "dragon/operators/framework/gradient_ops.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void GradientGenerateOp<Context>::DoRunWithType() {
  for (int i = 0; i < OutputSize(); i++) {
    if (!Output(i)->has_name()) continue;
    Output(i)->ReshapeLike(Input(i));
    auto value = cast::to<T>(defaults[i]);
    auto* y = Output(i)->template mutable_data<T, Context>();
    math::Set(Output(i)->count(), value, y, ctx());
  }
}

template <class Context>
void GradientGenerateOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void GradientGatherOp<Context>::DoRunWithType() {
  int64_t count = Output(0)->count();
  auto* y = Output(0)->template mutable_data<T, Context>();
  if (indices.size() == 1) {
    auto* x = Input(indices[0]).template data<T, Context>();
    math::Copy(count, x, y, ctx());
  } else if (indices.size() == 2) {
    CHECK_EQ(count, Input(indices[1]).count());
    auto* a = Input(indices[0]).template data<T, Context>();
    auto* b = Input(indices[1]).template data<T, Context>();
    math::Add(count, a, b, y, ctx());
  } else {
    size_t i = 1;
    auto* x = Input(indices[0]).template data<T, Context>();
    math::Copy(count, x, y, ctx());
    while (i < indices.size()) {
      if (indices.size() - i >= 2) {
        auto* a = Input(indices[i]).template data<T, Context>();
        auto* b = Input(indices[i + 1]).template data<T, Context>();
        kernel::GradientTwoSum(count, a, b, y, ctx());
        i += 2;
      } else {
        x = Input(indices[i]).template data<T, Context>();
        math::Add(count, y, x, y, ctx());
        break;
      }
    }
  }
}

template <class Context>
void GradientGatherOp<Context>::RunOnDevice() {
  if (indices.size() == 0) return;
  auto& Xi = Input(indices[0]);
  Output(0)->ReshapeLike(Xi);
  DispatchHelper<FloatingTensorTypes>::Call(this, Xi);
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
      << "\nRequires Input(0) == Output(0).";
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
void StopGradientOp<Context>::RunOnDevice() {
  if (Output(0)->name() != Input(0).name()) {
    Output(0)->ReshapeLike(Input(0));
    Output(0)->CopyFrom(Input(0), ctx());
  }
}

DEPLOY_CPU(GradientGenerate);
#ifdef USE_CUDA
DEPLOY_CUDA(GradientGenerate);
#endif

DEPLOY_CPU(GradientGather);
#ifdef USE_CUDA
DEPLOY_CUDA(GradientGather);
#endif

DEPLOY_CPU(GradientAdd);
#ifdef USE_CUDA
DEPLOY_CUDA(GradientAdd);
#endif

DEPLOY_CPU(StopGradient);
#ifdef USE_CUDA
DEPLOY_CUDA(StopGradient);
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
    /* X1, X2 */
    .NumInputs(2)
    /* Y */
    .NumOutputs(1)
    /* X1 => Y */
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
