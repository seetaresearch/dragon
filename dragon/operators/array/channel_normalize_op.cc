#include "dragon/operators/array/channel_normalize_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename Tx, typename Ty>
void ChannelNormalizeOp<Context>::DoRunWithTypeAndCast() {
  auto &X = Input(0), *Y = Output(0);
  CANONICALIZE_AXIS_WITH_TENSOR(X);

  int num_axes, num_dims = X.ndim();
  vec64_t X_strides(num_dims), Y_dims(num_dims);
  perm(0, &num_axes);

  if (num_axes > 0) {
    CHECK_EQ(num_dims, num_axes)
        << "\nProviding " << num_axes << " dimensions to permute, "
        << "while Tensor(" << X.name() << ")'s dims are " << X.DimString();
  }

  for (int i = 0; i < num_dims; ++i) {
    auto j = num_axes > 0 ? perm(i) : i;
    X_strides[i] = X.stride(j);
    Y_dims[i] = X.dim(j);
  }

  CHECK_LE(Y_dims[axis], X_mean_.count())
      << "\nProviding " << X_mean_.count() << " values to normalize Dimension("
      << Y_dims[axis] << ").";

  kernel::ChannelNormalize(
      axis,
      num_dims,
      X_strides.data(),
      Y_dims.data(),
      X.template data<Tx, Context>(),
      X_mean_.template data<float, Context>(),
      X_std_.template data<float, Context>(),
      Y->Reshape(Y_dims)->template mutable_data<Ty, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void ChannelNormalizeOp<Context>::DoRunWithType() {
  if (dtype() == "float16") {
    DoRunWithTypeAndCast<T, float16>();
  } else if (dtype() == "float32") {
    DoRunWithTypeAndCast<T, float>();
  } else if (dtype() == "float64") {
    DoRunWithTypeAndCast<T, double>();
  } else {
    LOG(FATAL) << MessageForUnsupported(
        dtype(), {"float16", "float32", "float64"});
  }
}

template <class Context>
void ChannelNormalizeOp<Context>::RunOnDevice() {
  DispatchHelper<NumericalTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(ChannelNormalize);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(ChannelNormalize);
#endif

OPERATOR_SCHEMA(ChannelNormalize)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

NO_GRADIENT(ChannelNormalize);

} // namespace dragon
