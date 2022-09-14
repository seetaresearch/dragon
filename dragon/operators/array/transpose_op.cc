#include "dragon/operators/array/transpose_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void TransposeOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});

  int num_axes, num_dims = X.ndim();
  perm(0, &num_axes);

  CHECK(num_axes == 0 || num_axes == num_dims)
      << "\nProviding " << num_axes << " dimensions to permute, "
      << "while Tensor(" << X.name() << ")'s dims are " << X.DimString();

  vec64_t Y_axes(num_dims), Y_dims(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    Y_axes[i] = num_axes > 0 ? perm(i) : num_dims - i - 1;
  }

  if (def().type() == "TransposeGradient") {
    const auto X_axes(Y_axes);
    for (int i = 0; i < num_dims; ++i) {
      Y_axes[X_axes[i]] = i;
    }
  }

  for (int i = 0; i < num_dims; ++i) {
    Y_dims[i] = X.dim(Y_axes[i]);
  }

  auto* scratch = ((void*)&X == (void*)Y)
      ? ctx()->workspace()->template data<T, Context>(X.count())
      : Y->Reshape(Y_dims)->template mutable_data<T, Context>();

  math::Transpose(
      num_dims,
      X.dims().data(),
      Y_axes.data(),
      X.template data<T, Context>(),
      scratch,
      ctx());

  if ((void*)&X == (void*)Y) {
    math::Copy(
        X.count(),
        scratch,
        Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
        ctx());
  }
}

DEPLOY_CPU_OPERATOR(Transpose);
REGISTER_CPU_OPERATOR(TransposeGradient, TransposeOp<CPUContext>);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Transpose);
REGISTER_CUDA_OPERATOR(TransposeGradient, TransposeOp<CUDAContext>);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(Transpose, Transpose);
REGISTER_MPS_OPERATOR(TransposeGradient, TransposeOp<MPSContext>);
#endif

OPERATOR_SCHEMA(Transpose).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(TransposeGradient)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}});

REGISTER_GRADIENT(Transpose, SimpleGradientMaker);

} // namespace dragon
