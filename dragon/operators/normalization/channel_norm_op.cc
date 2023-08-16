#include "dragon/operators/normalization/channel_norm_op.h"
#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"

namespace dragon {

template <class Context>
template <typename InputT, typename OutputT>
void ChannelNormOp<Context>::DoRunWithTypeAndCast() {
  auto &X = Input(0), *Y = Output(0);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);

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

  kernels::ChannelNorm(
      axis,
      num_dims,
      X_strides.data(),
      Y_dims.data(),
      X.template data<InputT, Context>(),
      X_mean_.template data<float, Context>(),
      X_std_.template data<float, Context>(),
      Y->Reshape(Y_dims)->template mutable_data<OutputT, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void ChannelNormOp<Context>::DoRunWithType() {
  if (data_type() == "float16") {
    DoRunWithTypeAndCast<T, float16>();
  } else if (data_type() == "bfloat16") {
    DoRunWithTypeAndCast<T, bfloat16>();
  } else if (data_type() == "float32") {
    DoRunWithTypeAndCast<T, float>();
  } else if (data_type() == "float64") {
    DoRunWithTypeAndCast<T, double>();
  } else {
    LOG(FATAL) << MessageForUnsupported(
        data_type(), {"float16", "bfloat16", "float32", "float64"});
  }
}

DEPLOY_CPU_OPERATOR(ChannelNorm);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(ChannelNorm);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(ChannelNorm, ChannelNorm);
#endif

DEFINE_OP_REPEATED_ARG(int64_t, ChannelNormOp, perm);

OPERATOR_SCHEMA(ChannelNorm).NumInputs(1).NumOutputs(1);

NO_GRADIENT(ChannelNorm);

} // namespace dragon
