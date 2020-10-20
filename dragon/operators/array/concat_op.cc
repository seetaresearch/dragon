#include "dragon/operators/array/concat_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void ConcatOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  CANONICALIZE_AXIS_WITH_TENSOR(X);

  // Store for the gradient calculation
  STORE_INPUT_SPEC(0);
  vec64_t Y_dims(X.dims());
  for (int i = 1; i < InputSize(); ++i) {
    CHECK_EQ(X.ndim(), Input(i).ndim())
        << "\nAll inputs should have the same ndim.";
    for (int j = 0; j < X.ndim(); ++j) {
      if (j == axis) continue;
      CHECK_EQ(Y_dims[j], Input(i).dim(j))
          << "\nAll inputs should have the same dims"
          << ", except the concat axis.";
    }
    STORE_INPUT_SPEC(i);
    Y_dims[axis] += Input(i).dim(axis);
  }

  Y->Reshape(Y_dims);
  int64_t output_offset = 0;

  for (int i = 0; i < InputSize(); ++i) {
    const auto& Xi = Input(i);
    math::CopyMatrix(
        Xi.count(0, axis),
        Xi.count(axis),
        Xi.count(axis),
        Y->count(axis),
        Xi.template data<T, Context>(),
        Y->template mutable_data<T, Context>() + output_offset,
        ctx());
    output_offset += Xi.count(axis);
  }
}

template <class Context>
void ConcatOp<Context>::RunOnDevice() {
  DispatchHelper<FullTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void ConcatGradientOp<Context>::DoRunWithType() {
  auto& dY = Input(0);
  CANONICALIZE_AXIS_WITH_TENSOR(dY);

  int64_t input_offset = 0;

  for (int i = 0; i < OutputSize(); ++i) {
    auto &X = RESTORE_INPUT_SPEC(i), *dX = Output(i);
    if (dX->has_name()) {
      math::CopyMatrix(
          dY.count(0, axis),
          X.count(axis),
          dY.count(axis),
          X.count(axis),
          dY.template data<T, Context>() + input_offset,
          dX->ReshapeLike(X)->template mutable_data<T, Context>(),
          ctx());
    }
    input_offset += X.count(axis);
  }
}

template <class Context>
void ConcatGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Concat);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Concat);
#endif

DEPLOY_CPU_OPERATOR(ConcatGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(ConcatGradient);
#endif

OPERATOR_SCHEMA(Concat)
    /* X(0), ... */
    .NumInputs(1, INT_MAX)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(ConcatGradient)
    /* dY */
    .NumInputs(1)
    /* dX(0), ... */
    .NumOutputs(1, INT_MAX);

REGISTER_GRADIENT(Concat, SimpleGradientMaker);

} // namespace dragon
