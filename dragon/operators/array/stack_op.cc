#include "dragon/operators/array/stack_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void StackOp<Context>::DoRunWithType() {
  auto &X_ref = Input(0), *Y = Output(0);
  GET_OP_AXIS_ARG(axis, X_ref.ndim() + 1, 0);

  vec64_t Y_dims(X_ref.dims());
  Y_dims.insert(Y_dims.begin() + axis, InputSize());
  for (int i = 1; i < InputSize(); ++i) {
    auto& X = Input(i);
    CHECK_EQ(X_ref.ndim(), X.ndim())
        << "\nAll inputs should have the same number of dimensions.";
    for (int j = 0; j < X.ndim(); ++j) {
      CHECK_EQ(X_ref.dim(j), X.dim(j))
          << "\nAll inputs should have the same dimensions.";
    }
  }

  Y->Reshape(Y_dims);
  int64_t copy_offset = 0;
  for (int i = 0; i < InputSize(); ++i) {
    auto& X = Input(i);
    Output("X_spec:" + str::to(i))->ReshapeLike(X);
    math::CopyMatrix(
        X.count(0, axis), // M
        X.count(axis), // N
        X.count(axis), // ldx
        Y->count(axis), // ldy
        0, // x_offset
        copy_offset, // y_offset
        X.template data<T, Context>(),
        Y->template mutable_data<T, Context>(),
        ctx());
    copy_offset += X.count(axis);
  }
}

template <class Context>
template <typename T>
void StackGradientOp<Context>::DoRunWithType() {
  auto &X_ref = Input("X_spec:0"), &dY = Input(0);
  GET_OP_AXIS_ARG(axis, X_ref.ndim() + 1, 0);

  int64_t copy_offset = 0;
  for (int i = 0; i < OutputSize(); ++i) {
    auto &X = Input("X_spec:" + str::to(i)), *dX = Output(i);
    if (dX->has_name()) {
      math::CopyMatrix(
          dY.count(0, axis), // M
          X.count(axis), // N
          dY.count(axis), // ldx
          X.count(axis), // ldy
          copy_offset, // x_offset
          0, // y_offset
          dY.template data<T, Context>(),
          dX->ReshapeLike(X)->template mutable_data<T, Context>(),
          ctx());
    }
    copy_offset += X.count(axis);
  }
}

DEPLOY_CPU_OPERATOR(Stack);
DEPLOY_CPU_OPERATOR(StackGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Stack);
DEPLOY_CUDA_OPERATOR(StackGradient);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(Stack, Stack);
DEPLOY_MPS_OPERATOR(StackGradient, StackGradient);
#endif
#ifdef USE_MLU
DEPLOY_MLU_OPERATOR(Stack);
DEPLOY_MLU_OPERATOR(StackGradient);
#endif

OPERATOR_SCHEMA(Stack).NumInputs(1, INT_MAX).NumOutputs(1);
OPERATOR_SCHEMA(StackGradient).NumInputs(1).NumOutputs(1, INT_MAX);

REGISTER_GRADIENT(Stack, SimpleGradientMaker);

} // namespace dragon
