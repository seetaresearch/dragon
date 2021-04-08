#include "dragon/core/workspace.h"
#include "dragon/operators/math/elementwise_ops.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void SubOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1);
  SET_INPUT_SPEC(0);
  SET_INPUT_SPEC(1);

  vec64_t Y_dims(A.dims());
  if (A.dims() == B.dims()) {
    math::Sub(
        A.count(),
        A.template data<T, Context>(),
        B.template data<T, Context>(),
        Output(0, {0, 1})->Reshape(Y_dims)->template mutable_data<T, Context>(),
        ctx());
  } else if (math::utils::IsBinaryBroadcast(A.dims(), B.dims(), Y_dims)) {
    auto* Y = Output(0, CheckOutputAliases(A, B, Output(0), Y_dims));
    math::Sub(
        A.ndim(),
        A.dims().data(),
        B.ndim(),
        B.dims().data(),
        A.template data<T, Context>(),
        B.template data<T, Context>(),
        Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
        ctx());
  } else {
    LOG(FATAL) << "Could not broadcast together with shapes: " << A.DimString()
               << " " << B.DimString();
  }
}

template <class Context>
void SubOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Numerical>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void SubGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dA = Output(0), *dB = Output(1);
  auto &A = INPUT_SPEC(0), &B = INPUT_SPEC(1);

  vec32_t A_broadcast_axes, B_broadcast_axes;
  vec32_t Y_dims(dY.dims().begin(), dY.dims().end());
  math::utils::ComputeBinaryBroadcastAxes(
      A.dims(), B.dims(), dY.dims(), A_broadcast_axes, B_broadcast_axes);

  if (dA->has_name()) {
    if (A_broadcast_axes.empty()) {
      dA->ReshapeLike(A)->CopyFrom(dY, ctx());
    } else {
      math::ReduceSum(
          Y_dims.size(),
          Y_dims.data(),
          A_broadcast_axes.size(),
          A_broadcast_axes.data(),
          1.f,
          dY.template data<T, Context>(),
          dA->ReshapeLike(A)->template mutable_data<T, Context>(),
          ctx());
    }
  }

  if (dB->has_name()) {
    if (B_broadcast_axes.empty()) {
      math::Neg(
          B.count(),
          dY.template data<T, Context>(),
          dB->ReshapeLike(B)->template mutable_data<T, Context>(),
          ctx());
    } else {
      math::ReduceSum(
          Y_dims.size(),
          Y_dims.data(),
          B_broadcast_axes.size(),
          B_broadcast_axes.data(),
          -1.f,
          dY.template data<T, Context>(),
          dB->ReshapeLike(B)->template mutable_data<T, Context>(),
          ctx());
    }
  }
}

template <class Context>
void SubGradientOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Sub);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Sub);
#endif

DEPLOY_CPU_OPERATOR(SubGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SubGradient);
#endif

OPERATOR_SCHEMA(Sub)
    /* A, B */
    .NumInputs(2)
    /* Y */
    .NumOutputs(1)
    /* A => Y, B => Y */
    .AllowInplace({{0, 0}, {1, 0}});

OPERATOR_SCHEMA(SubGradient)
    /* dY */
    .NumInputs(1)
    /* dA, dB */
    .NumOutputs(2)
    /* dY => dA, dY => dB */
    .AllowInplace({{0, 0}, {0, 1}});

REGISTER_GRADIENT(Sub, SimpleGradientMaker);

} // namespace dragon
