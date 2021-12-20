#include "dragon/core/workspace.h"
#include "dragon/operators/math/elementwise_op.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void MaximumGradientOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1), &dY = Input(2);
  auto *dA = Output(0), *dB = Output(1);

  vec64_t A_broadcast_axes, B_broadcast_axes;
  math::utils::ComputeBroadcastAxes(
      A.dims(), B.dims(), dY.dims(), A_broadcast_axes, B_broadcast_axes);

  // Scratch to save the intermediates.
  size_t scratch_size = 0, scratch_offset = 0;
  if (dA->has_name() || dB->has_name()) scratch_size += dY.size();
  if ((dA->has_name() && !A_broadcast_axes.empty()) ||
      (dB->has_name() && !B_broadcast_axes.empty())) {
    scratch_size += (scratch_offset = dY.size() * sizeof(T));
  }
  void* scratch = ctx()->workspace()->template data<Context>(scratch_size);
  auto* mask = (bool*)scratch + scratch_offset;

  if (dA->has_name()) {
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
      math::ApplyMask(
          dY.count(),
          1.f,
          (uint8_t*)mask,
          dY.template data<T, Context>(),
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
      math::ApplyMask(
          dY.count(),
          1.f,
          (uint8_t*)mask,
          dY.template data<T, Context>(),
          (T*)scratch,
          ctx());
      math::ReduceSum(
          dY.ndim(),
          dY.dims().data(),
          A_broadcast_axes.size(),
          A_broadcast_axes.data(),
          1.f,
          (T*)scratch,
          dA->ReshapeLike(A)->template mutable_data<T, Context>(),
          ctx());
    }
  }

  if (dB->has_name()) {
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
      math::ApplyMask(
          dY.count(),
          1.f,
          (uint8_t*)mask,
          dY.template data<T, Context>(),
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
      math::ApplyMask(
          dY.count(),
          1.f,
          (uint8_t*)mask,
          dY.template data<T, Context>(),
          (T*)scratch,
          ctx());
      math::ReduceSum(
          dY.ndim(),
          dY.dims().data(),
          B_broadcast_axes.size(),
          B_broadcast_axes.data(),
          1.f,
          (T*)scratch,
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
