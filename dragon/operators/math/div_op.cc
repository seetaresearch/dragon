#include "dragon/core/workspace.h"
#include "dragon/operators/math/elementwise_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void DivOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1);
  Output("A_spec")->ReshapeLike(A);
  Output("B_spec")->ReshapeLike(B);

  vec64_t Y_dims(A.dims());
  if (A.dims() == B.dims()) {
    math::Div(
        A.count(),
        A.template data<T, Context>(),
        B.template data<T, Context>(),
        Output(0, {0, 1})->Reshape(Y_dims)->template mutable_data<T, Context>(),
        ctx());
  } else if (math::utils::IsBinaryBroadcast(A.dims(), B.dims(), Y_dims)) {
    auto* Y = Output(0, CheckOutputAliases(A, B, Output(0), Y_dims));
    math::Div(
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
void DivOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Numerical>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void DivGradientOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1), &dY = Input(2);
  auto &A_spec = Input("A_spec"), &B_spec = Input("B_spec");
  auto *dA = Output(0), *dB = Output(1);

  vec64_t A_broadcast_axes, B_broadcast_axes;
  math::utils::ComputeBroadcastAxes(
      A_spec.dims(),
      B_spec.dims(),
      dY.dims(),
      A_broadcast_axes,
      B_broadcast_axes);

  // Scratch to save the intermediates.
  T* scratch = nullptr;
  if ((dA->has_name() && !A_broadcast_axes.empty()) ||
      (dB->has_name() && !B_broadcast_axes.empty())) {
    scratch = ctx()->workspace()->template data<T, Context>(dY.count());
  }

  if (dA->has_name()) {
    if (A_broadcast_axes.empty()) {
      if (B_broadcast_axes.empty()) {
        math::Div(
            B_spec.count(),
            dY.template data<T, Context>(),
            B.template data<T, Context>(),
            dA->ReshapeLike(A_spec)->template mutable_data<T, Context>(),
            ctx());
      } else {
        math::Div(
            dY.ndim(),
            dY.dims().data(),
            B_spec.ndim(),
            B_spec.dims().data(),
            dY.template data<T, Context>(),
            B.template data<T, Context>(),
            dA->ReshapeLike(A_spec)->template mutable_data<T, Context>(),
            ctx());
      }
    } else {
      if (B_broadcast_axes.empty()) {
        math::Div(
            B_spec.count(),
            dY.template data<T, Context>(),
            B.template data<T, Context>(),
            scratch,
            ctx());
      } else {
        math::Div(
            dY.ndim(),
            dY.dims().data(),
            B_spec.ndim(),
            B_spec.dims().data(),
            dY.template data<T, Context>(),
            B.template data<T, Context>(),
            scratch,
            ctx());
      }
      math::ReduceSum(
          dY.ndim(),
          dY.dims().data(),
          A_broadcast_axes.size(),
          A_broadcast_axes.data(),
          1.f,
          scratch,
          dA->ReshapeLike(A_spec)->template mutable_data<T, Context>(),
          ctx());
    }
  }

  if (dB->has_name()) {
    if (B_broadcast_axes.empty()) {
      if (A_broadcast_axes.empty()) {
        math::Mul(
            A_spec.count(),
            dY.template data<T, Context>(),
            A.template data<T, Context>(),
            dB->ReshapeLike(B_spec)->template mutable_data<T, Context>(),
            ctx());
      } else {
        math::Mul(
            dY.ndim(),
            dY.dims().data(),
            A_spec.ndim(),
            A_spec.dims().data(),
            dY.template data<T, Context>(),
            A.template data<T, Context>(),
            dB->ReshapeLike(B_spec)->template mutable_data<T, Context>(),
            ctx());
      }
    } else {
      if (A_broadcast_axes.empty()) {
        math::Mul(
            A_spec.count(),
            dY.template data<T, Context>(),
            A.template data<T, Context>(),
            scratch,
            ctx());
      } else {
        math::Mul(
            dY.ndim(),
            dY.dims().data(),
            A_spec.ndim(),
            A_spec.dims().data(),
            dY.template data<T, Context>(),
            A.template data<T, Context>(),
            scratch,
            ctx());
      }
      math::ReduceSum(
          dY.ndim(),
          dY.dims().data(),
          B_broadcast_axes.size(),
          B_broadcast_axes.data(),
          1.f,
          scratch,
          dB->ReshapeLike(B_spec)->template mutable_data<T, Context>(),
          ctx());
    }
    math::Div(
        B_spec.count(),
        dB->template data<T, Context>(),
        B.template data<T, Context>(),
        dB->template mutable_data<T, Context>(),
        ctx());
    math::Div(
        B_spec.count(),
        dB->template data<T, Context>(),
        B.template data<T, Context>(),
        dB->template mutable_data<T, Context>(),
        ctx());
    math::Neg(
        B_spec.count(),
        dB->template data<T, Context>(),
        dB->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
void DivGradientOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Floating>::Call(this, Input(2));
}

DEPLOY_CPU_OPERATOR(Div);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Div);
#endif

DEPLOY_CPU_OPERATOR(DivGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(DivGradient);
#endif

OPERATOR_SCHEMA(Div)
    /* A, B */
    .NumInputs(2)
    /* Y */
    .NumOutputs(1)
    /* A => Y, B => Y */
    .AllowInplace({{0, 0}, {1, 0}});

OPERATOR_SCHEMA(DivGradient)
    /* A, B, dY */
    .NumInputs(3)
    /* dA, dB */
    .NumOutputs(2)
    /* dY => dA, dY => dB */
    .AllowInplace({{2, 0}, {2, 1}});

REGISTER_GRADIENT(Div, GenericGradientMaker);

} // namespace dragon
