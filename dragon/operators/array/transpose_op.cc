#include "dragon/operators/array/transpose_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void TransposeOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  int num_axes, num_dims = X.ndim();
  vec64_t X_strides(num_dims), Y_dims(num_dims);
  perm(0, &num_axes);

  CHECK(num_axes == 0 || num_axes == num_dims)
      << "\nProviding " << num_axes << " dimensions to permute, "
      << "while Tensor(" << X.name() << ")'s dims are " << X.DimString();

  for (int i = 0; i < num_dims; ++i) {
    auto axis = num_axes > 0 ? perm(i) : num_dims - i - 1;
    X_strides[i] = X.stride(axis);
    Y_dims[i] = X.dim(axis);
  }

  // Store for the gradient calculation
  SET_INPUT_SPEC(0);
  Buffer("X_strides")->template CopyFrom<int64_t>(X_strides);
  Buffer("Y_dims")->template CopyFrom<int64_t>(Y_dims);

  kernels::Transpose(
      num_dims,
      X_strides.data(),
      Y_dims.data(),
      X.template data<T, Context>(),
      Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void TransposeOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Generic>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void TransposeGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0);
  dX->ReshapeLike(INPUT_SPEC(0));

  vec64_t X_strides, Y_dims;
  Buffer("X_strides")->template CopyTo<int64_t>(X_strides);
  Buffer("Y_dims")->template CopyTo<int64_t>(Y_dims);

  kernels::TransposeGrad(
      X_strides.size(),
      X_strides.data(),
      Y_dims.data(),
      dY.template data<T, Context>(),
      dX->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void TransposeGradientOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Transpose);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Transpose);
#endif

DEPLOY_CPU_OPERATOR(TransposeGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(TransposeGradient);
#endif

OPERATOR_SCHEMA(Transpose)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(TransposeGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(Transpose, SimpleGradientMaker);

} // namespace dragon
