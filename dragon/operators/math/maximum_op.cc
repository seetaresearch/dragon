#include "dragon/core/workspace.h"
#include "dragon/operators/math/elementwise_ops.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void MaximumGradientOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1), &dY = Input(2);
  auto *dA = Output(0), *dB = Output(1);

  vec32_t A_broadcast_axes, B_broadcast_axes;
  vec32_t Y_dims(dY.dims().begin(), dY.dims().end());
  math::utils::ComputeBinaryBroadcastAxes(
      A.dims(), B.dims(), dY.dims(), A_broadcast_axes, B_broadcast_axes);

  // Temporal space to store the intermediate gradient
  bool* mask = nullptr;
  T* scratch = nullptr;

  if (dA->has_name()) {
    auto scratches = ctx()->workspace()->template data<Context>(
        {dY.size() * sizeof(T), dY.size() * sizeof(bool)});
    mask = (bool*)scratches[1], scratch = (T*)scratches[0];
    if (A_broadcast_axes.empty()) {
      if (B_broadcast_axes.empty()) {
        math::Greater(
            A.count(),
            A.template data<T, Context>(),
            B.template data<T, Context>(),
            mask,
            ctx());
      } else {
        math::Greater(
            A.ndim(),
            A.dims().data(),
            B.ndim(),
            B.dims().data(),
            A.template data<T, Context>(),
            B.template data<T, Context>(),
            mask,
            ctx());
      }
      math::Cast(dY.count(), mask, scratch, ctx());
      math::Mul(
          dY.count(),
          dY.template data<T, Context>(),
          scratch,
          dA->ReshapeLike(A)->template mutable_data<T, Context>(),
          ctx());
    } else {
      math::Greater(
          A.ndim(),
          A.dims().data(),
          B.ndim(),
          B.dims().data(),
          A.template data<T, Context>(),
          B.template data<T, Context>(),
          mask,
          ctx());
      math::Cast(dY.count(), mask, scratch, ctx());
      math::Mul(
          dY.count(), dY.template data<T, Context>(), scratch, scratch, ctx());
      math::ReduceSum(
          Y_dims.size(),
          Y_dims.data(),
          A_broadcast_axes.size(),
          A_broadcast_axes.data(),
          1.f,
          scratch,
          dA->ReshapeLike(A)->template mutable_data<T, Context>(),
          ctx());
    }
  }

  if (dB->has_name()) {
    if (mask == nullptr) {
      auto scratches = ctx()->workspace()->template data<Context>(
          {dY.size() * sizeof(T), dY.size() * sizeof(bool)});
      mask = (bool*)scratches[1], scratch = (T*)scratches[0];
    }
    if (B_broadcast_axes.empty()) {
      if (A_broadcast_axes.empty()) {
        math::LessEqual(
            A.count(),
            A.template data<T, Context>(),
            B.template data<T, Context>(),
            mask,
            ctx());
      } else {
        math::LessEqual(
            A.ndim(),
            A.dims().data(),
            B.ndim(),
            B.dims().data(),
            A.template data<T, Context>(),
            B.template data<T, Context>(),
            mask,
            ctx());
      }
      math::Cast(dY.count(), mask, scratch, ctx());
      math::Mul(
          dY.count(),
          dY.template data<T, Context>(),
          scratch,
          dB->ReshapeLike(B)->template mutable_data<T, Context>(),
          ctx());
    } else {
      math::LessEqual(
          A.ndim(),
          A.dims().data(),
          B.ndim(),
          B.dims().data(),
          A.template data<T, Context>(),
          B.template data<T, Context>(),
          mask,
          ctx());
      math::Cast(dY.count(), mask, scratch, ctx());
      math::Mul(
          dY.count(), dY.template data<T, Context>(), scratch, scratch, ctx());
      math::ReduceSum(
          Y_dims.size(),
          Y_dims.data(),
          B_broadcast_axes.size(),
          B_broadcast_axes.data(),
          1.f,
          scratch,
          dB->ReshapeLike(B)->template mutable_data<T, Context>(),
          ctx());
    }
  }
}

template <class Context>
void MaximumGradientOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Floating>::Call(this, Input(2));
}

DEPLOY_CPU_OPERATOR(MaximumGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(MaximumGradient);
#endif

OPERATOR_SCHEMA(MaximumGradient)
    /* A, B, dY */
    .NumInputs(3)
    /* dA, dB */
    .NumOutputs(2);

REGISTER_GRADIENT(Maximum, GenericGradientMaker);

} // namespace dragon
