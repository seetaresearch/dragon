#include "dragon/operators/array/expand_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void ExpandOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  Output("X_spec")->ReshapeLike(X);

  int num_dims;
  dims(0, &num_dims);

  vec64_t X_dims(num_dims), Y_dims;
  for (int i = 0; i < num_dims; ++i) {
    const auto new_dim = dims(i);
    X_dims[i] = (new_dim < 0 ? X.dim(i - num_dims) : new_dim);
  }

  if (math::utils::IsBinaryBroadcast(X.dims(), X_dims, Y_dims)) {
    math::Set(
        X.ndim(),
        X.dims().data(),
        Y_dims.size(),
        Y_dims.data(),
        X.template data<T, Context>(),
        Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
        ctx());
  } else {
    LOG(FATAL) << "Could not broadcast together with shapes: " << X.DimString()
               << " " << Tensor::DimString(X_dims);
  }
}

template <class Context>
template <typename T>
void ExpandGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0)->ReshapeLike(Input("X_spec"));

  vec64_t broadcast_axes, _;
  math::utils::ComputeBroadcastAxes(
      dX->dims(), dY.dims(), dY.dims(), broadcast_axes, _);

  if (broadcast_axes.empty()) {
    dX->CopyFrom(dY, ctx());
  } else {
    math::ReduceSum(
        dY.ndim(),
        dY.dims().data(),
        broadcast_axes.size(),
        broadcast_axes.data(),
        1.f,
        dY.template data<T, Context>(),
        dX->template mutable_data<T, Context>(),
        ctx());
  }
}

DEPLOY_CPU_OPERATOR(Expand);
DEPLOY_CPU_OPERATOR(ExpandGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Expand);
DEPLOY_CUDA_OPERATOR(ExpandGradient);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(Expand, Expand);
DEPLOY_MPS_OPERATOR(ExpandGradient, ExpandGradient);
#endif

DEFINE_OP_REPEATED_ARG(int64_t, ExpandOp, dims);

OPERATOR_SCHEMA(Expand).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(ExpandGradient).NumInputs(1).NumOutputs(1);

REGISTER_GRADIENT(Expand, SimpleGradientMaker);

} // namespace dragon
