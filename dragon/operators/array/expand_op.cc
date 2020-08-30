#include "dragon/operators/array/expand_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void ExpandOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  int num_dims;
  dims(0, &num_dims);

  vec64_t X_dims(num_dims), Y_dims;
  for (int i = 0; i < num_dims; i++)
    X_dims[i] = dims(i);

  // Store for the gradient calculation
  STORE_INPUT_SPEC(0);

  if (utils::math::IsBinaryBroadcast(X.dims(), X_dims, Y_dims)) {
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
void ExpandOp<Context>::RunOnDevice() {
  DispatchHelper<FullTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void ExpandGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0);
  dX->ReshapeLike(RESTORE_INPUT_SPEC(0));

  vec32_t X_broadcast_axes, _;
  vec32_t Y_dims(dY.dims().begin(), dY.dims().end());
  utils::math::ComputeBinaryBroadcastAxes(
      dX->dims(), dY.dims(), dY.dims(), X_broadcast_axes, _);

  if (X_broadcast_axes.empty()) {
    dX->CopyFrom(dY, ctx());
    return; // Just copy the contents
  } else {
    math::ReduceSum(
        Y_dims.size(),
        Y_dims.data(),
        X_broadcast_axes.size(),
        X_broadcast_axes.data(),
        1.f,
        dY.template data<T, Context>(),
        dX->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
void ExpandGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Expand);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Expand);
#endif

DEPLOY_CPU_OPERATOR(ExpandGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(ExpandGradient);
#endif

OPERATOR_SCHEMA(Expand)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(ExpandGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(Expand, SimpleGradientMaker);

} // namespace dragon
