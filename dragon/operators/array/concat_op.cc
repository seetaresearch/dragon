#include "dragon/operators/array/concat_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/op_kernels.h"

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

  int64_t index = 0;
  auto* y = Y->Reshape(Y_dims)->template mutable_data<T, Context>();

  for (int i = 0; i < InputSize(); i++) {
    kernel::Concat(
        X.count(0, axis),
        X.count(axis + 1),
        Input(i).dim(axis),
        Y_dims[axis],
        index,
        Input(i).template data<T, Context>(),
        y,
        ctx());
    index += Input(i).dim(axis);
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

  int64_t index = 0;
  for (int i = 0; i < OutputSize(); i++) {
    auto &X = RESTORE_INPUT_SPEC(i), *dX = Output(i);
    if (dX->has_name()) {
      kernel::Split(
          dY.count(0, axis),
          dY.count(axis + 1),
          dY.dim(axis),
          X.dim(axis),
          index,
          dY.template data<T, Context>(),
          dX->ReshapeLike(X)->template mutable_data<T, Context>(),
          ctx());
    }
    index += X.dim(axis);
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
