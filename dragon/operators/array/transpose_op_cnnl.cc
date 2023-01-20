#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/array/transpose_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLTransposeOp<Context>::DoRunWithType() {
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

  impl_.Setup<T>(X.dims(), Y_axes, ctx());
  auto* data = ((void*)&X == (void*)Y)
      ? ctx()->workspace()->template data<T, Context>(X.count())
      : Y->Reshape(Y_dims)->template mutable_data<T, Context>();
  auto* scratch = ctx()->workspace()->template data<Context>(
      impl_.scratch_size(), "BufferKernel");
  impl_.Compute<T>(X.template data<T, Context>(), data, scratch, ctx());

  if ((void*)&X == (void*)Y) {
    math::Copy(
        X.count(),
        data,
        Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
        ctx());
  }
}

DEPLOY_CNNL_OPERATOR(Transpose);
REGISTER_CNNL_OPERATOR(TransposeGradient, CNNLTransposeOp<MLUContext>);

DEFINE_OP_REPEATED_ARG(int64_t, CNNLTransposeOp, perm);

} // namespace dragon

#endif // USE_MLU
