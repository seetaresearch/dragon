#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/math/elementwise_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLAddGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dA = Output(0), *dB = Output(1);
  auto &A_spec = Input("A_spec"), &B_spec = Input("B_spec");

  vec64_t A_broadcast_axes, B_broadcast_axes;
  math::utils::ComputeBroadcastAxes(
      A_spec.dims(),
      B_spec.dims(),
      dY.dims(),
      A_broadcast_axes,
      B_broadcast_axes);

  if (dA->has_name()) {
    if (A_broadcast_axes.empty()) {
      dA->ReshapeLike(A_spec)->CopyFrom(dY, ctx());
    } else {
      reduce_impl_.Setup<T>(dY.dims(), A_broadcast_axes, ctx());
      const auto scratch_size = reduce_impl_.scratch_size();
      reduce_impl_.Compute<T>(
          dY.template data<T, Context>(),
          dA->ReshapeLike(A_spec)->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(scratch_size),
          ctx());
    }
  }

  if (dB->has_name()) {
    if (B_broadcast_axes.empty()) {
      dB->ReshapeLike(B_spec)->CopyFrom(dY, ctx());
    } else {
      reduce_impl_.Setup<T>(dY.dims(), B_broadcast_axes, ctx());
      const auto scratch_size = reduce_impl_.scratch_size();
      reduce_impl_.Compute<T>(
          dY.template data<T, Context>(),
          dB->ReshapeLike(B_spec)->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(scratch_size),
          ctx());
    }
  }
}

template <class Context>
template <typename T>
void CNNLSubGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dA = Output(0), *dB = Output(1);
  auto &A_spec = Input("A_spec"), &B_spec = Input("B_spec");

  vec64_t A_broadcast_axes, B_broadcast_axes;
  math::utils::ComputeBroadcastAxes(
      A_spec.dims(),
      B_spec.dims(),
      dY.dims(),
      A_broadcast_axes,
      B_broadcast_axes);

  if (dA->has_name()) {
    if (A_broadcast_axes.empty()) {
      dA->ReshapeLike(A_spec)->CopyFrom(dY, ctx());
    } else {
      reduce_impl_.Setup<T>(dY.dims(), A_broadcast_axes, ctx());
      const auto scratch_size = reduce_impl_.scratch_size();
      reduce_impl_.Compute<T>(
          dY.template data<T, Context>(),
          dA->ReshapeLike(A_spec)->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(scratch_size),
          ctx());
    }
  }

  if (dB->has_name()) {
    if (B_broadcast_axes.empty()) {
      math::Neg(
          B_spec.count(),
          dY.template data<T, Context>(),
          dB->ReshapeLike(B_spec)->template mutable_data<T, Context>(),
          ctx());
    } else {
      reduce_impl_.Setup<T>(dY.dims(), B_broadcast_axes, ctx());
      const auto scratch_size = reduce_impl_.scratch_size();
      reduce_impl_.Compute<T>(
          dY.template data<T, Context>(),
          dB->ReshapeLike(B_spec)->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(scratch_size),
          ctx(),
          -1.f);
    }
  }
}

template <class Context>
template <typename T>
void CNNLMulGradientOp<Context>::DoRunWithType() {
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

  // Buffer to save the intermediates.
  T* buffer = nullptr;
  if ((dA->has_name() && !A_broadcast_axes.empty()) ||
      (dB->has_name() && !B_broadcast_axes.empty())) {
    buffer = ctx()->workspace()->template data<T, Context>(
        dY.count(), "BufferKernel");
  }

  if (dA->has_name()) {
    if (A_broadcast_axes.empty()) {
      if (B_broadcast_axes.empty()) {
        math::Mul(
            B_spec.count(),
            dY.template data<T, Context>(),
            B.template data<T, Context>(),
            dA->ReshapeLike(A_spec)->template mutable_data<T, Context>(),
            ctx());
      } else {
        math::Mul(
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
        math::Mul(
            B_spec.count(),
            dY.template data<T, Context>(),
            B.template data<T, Context>(),
            buffer,
            ctx());
      } else {
        math::Mul(
            dY.ndim(),
            dY.dims().data(),
            B_spec.ndim(),
            B_spec.dims().data(),
            dY.template data<T, Context>(),
            B.template data<T, Context>(),
            buffer,
            ctx());
      }
      reduce_impl_.Setup<T>(dY.dims(), A_broadcast_axes, ctx());
      const auto scratch_size = reduce_impl_.scratch_size();
      reduce_impl_.Compute<T>(
          buffer,
          dA->ReshapeLike(A_spec)->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(scratch_size),
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
            buffer,
            ctx());
      } else {
        math::Mul(
            dY.ndim(),
            dY.dims().data(),
            A_spec.ndim(),
            A_spec.dims().data(),
            dY.template data<T, Context>(),
            A.template data<T, Context>(),
            buffer,
            ctx());
      }
      reduce_impl_.Setup<T>(dY.dims(), B_broadcast_axes, ctx());
      const auto scratch_size = reduce_impl_.scratch_size();
      reduce_impl_.Compute<T>(
          buffer,
          dB->ReshapeLike(B_spec)->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(scratch_size),
          ctx());
    }
  }
}

template <class Context>
template <typename T>
void CNNLDivGradientOp<Context>::DoRunWithType() {
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

  // Buffer to save the intermediates.
  T* buffer = nullptr;
  if ((dA->has_name() && !A_broadcast_axes.empty()) ||
      (dB->has_name() && !B_broadcast_axes.empty())) {
    buffer = ctx()->workspace()->template data<T, Context>(
        dY.count(), "BufferKernel");
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
            buffer,
            ctx());
      } else {
        math::Div(
            dY.ndim(),
            dY.dims().data(),
            B_spec.ndim(),
            B_spec.dims().data(),
            dY.template data<T, Context>(),
            B.template data<T, Context>(),
            buffer,
            ctx());
      }
      reduce_impl_.Setup<T>(dY.dims(), A_broadcast_axes, ctx());
      const auto scratch_size = reduce_impl_.scratch_size();
      reduce_impl_.Compute<T>(
          buffer,
          dA->ReshapeLike(A_spec)->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(scratch_size),
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
            buffer,
            ctx());
      } else {
        math::Mul(
            dY.ndim(),
            dY.dims().data(),
            A_spec.ndim(),
            A_spec.dims().data(),
            dY.template data<T, Context>(),
            A.template data<T, Context>(),
            buffer,
            ctx());
      }
      reduce_impl_.Setup<T>(dY.dims(), B_broadcast_axes, ctx());
      const auto scratch_size = reduce_impl_.scratch_size();
      reduce_impl_.Compute<T>(
          buffer,
          dB->ReshapeLike(B_spec)->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(scratch_size),
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
template <typename T>
void CNNLMaximumGradientOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1), &dY = Input(2);
  auto *dA = Output(0), *dB = Output(1);

  vec64_t A_broadcast_axes, B_broadcast_axes;
  math::utils::ComputeBroadcastAxes(
      A.dims(), B.dims(), dY.dims(), A_broadcast_axes, B_broadcast_axes);

  // Buffer to save the intermediates.
  void *buffer, *mask;
  if (dA->has_name() || dB->has_name()) {
    mask = ctx()->workspace()->template data<bool, Context>(
        dY.count(), "BufferKernel2");
  }
  if ((dA->has_name() && !A_broadcast_axes.empty()) ||
      (dB->has_name() && !B_broadcast_axes.empty())) {
    buffer = ctx()->workspace()->template data<T, Context>(
        dY.count(), "BufferKernel");
  }

  if (dA->has_name()) {
    if (A_broadcast_axes.empty()) {
      if (B_broadcast_axes.empty()) {
        math::Greater(
            A.count(),
            A.template data<T, Context>(),
            B.template data<T, Context>(),
            (bool*)mask,
            ctx());
      } else {
        math::Greater(
            A.ndim(),
            A.dims().data(),
            B.ndim(),
            B.dims().data(),
            A.template data<T, Context>(),
            B.template data<T, Context>(),
            (bool*)mask,
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
          (bool*)mask,
          ctx());
      math::ApplyMask(
          dY.count(),
          1.f,
          (uint8_t*)mask,
          dY.template data<T, Context>(),
          (T*)buffer,
          ctx());
      reduce_impl_.Setup<T>(dY.dims(), A_broadcast_axes, ctx());
      const auto scratch_size = reduce_impl_.scratch_size();
      reduce_impl_.Compute<T>(
          (T*)buffer,
          dA->ReshapeLike(A)->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(scratch_size),
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
            (bool*)mask,
            ctx());
      } else {
        math::LessEqual(
            A.ndim(),
            A.dims().data(),
            B.ndim(),
            B.dims().data(),
            A.template data<T, Context>(),
            B.template data<T, Context>(),
            (bool*)mask,
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
          (bool*)mask,
          ctx());
      math::ApplyMask(
          dY.count(),
          1.f,
          (uint8_t*)mask,
          dY.template data<T, Context>(),
          (T*)buffer,
          ctx());
      reduce_impl_.Setup<T>(dY.dims(), B_broadcast_axes, ctx());
      const auto scratch_size = reduce_impl_.scratch_size();
      reduce_impl_.Compute<T>(
          (T*)buffer,
          dB->ReshapeLike(B)->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(scratch_size),
          ctx());
    }
  }
}

template <class Context>
template <typename T>
void CNNLMinimumGradientOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1), &dY = Input(2);
  auto *dA = Output(0), *dB = Output(1);

  vec64_t A_broadcast_axes, B_broadcast_axes;
  math::utils::ComputeBroadcastAxes(
      A.dims(), B.dims(), dY.dims(), A_broadcast_axes, B_broadcast_axes);

  // Buffer to save the intermediates.
  void *buffer, *mask;
  if (dA->has_name() || dB->has_name()) {
    mask = ctx()->workspace()->template data<bool, Context>(
        dY.size(), "BufferKernel2");
  }
  if ((dA->has_name() && !A_broadcast_axes.empty()) ||
      (dB->has_name() && !B_broadcast_axes.empty())) {
    buffer = ctx()->workspace()->template data<T, Context>(
        dY.count(), "BufferKernel");
  }

  if (dA->has_name()) {
    if (A_broadcast_axes.empty()) {
      if (B_broadcast_axes.empty()) {
        math::Less(
            A.count(),
            A.template data<T, Context>(),
            B.template data<T, Context>(),
            (bool*)mask,
            ctx());
      } else {
        math::Less(
            A.ndim(),
            A.dims().data(),
            B.ndim(),
            B.dims().data(),
            A.template data<T, Context>(),
            B.template data<T, Context>(),
            (bool*)mask,
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
      math::Less(
          A.ndim(),
          A.dims().data(),
          B.ndim(),
          B.dims().data(),
          A.template data<T, Context>(),
          B.template data<T, Context>(),
          (bool*)mask,
          ctx());
      math::ApplyMask(
          dY.count(),
          1.f,
          (uint8_t*)mask,
          dY.template data<T, Context>(),
          (T*)buffer,
          ctx());
      reduce_impl_.Setup<T>(dY.dims(), A_broadcast_axes, ctx());
      const auto scratch_size = reduce_impl_.scratch_size();
      reduce_impl_.Compute<T>(
          (T*)buffer,
          dA->ReshapeLike(A)->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(scratch_size),
          ctx());
    }
  }

  if (dB->has_name()) {
    if (B_broadcast_axes.empty()) {
      if (A_broadcast_axes.empty()) {
        math::GreaterEqual(
            A.count(),
            A.template data<T, Context>(),
            B.template data<T, Context>(),
            (bool*)mask,
            ctx());
      } else {
        math::GreaterEqual(
            A.ndim(),
            A.dims().data(),
            B.ndim(),
            B.dims().data(),
            A.template data<T, Context>(),
            B.template data<T, Context>(),
            (bool*)mask,
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
      math::GreaterEqual(
          A.ndim(),
          A.dims().data(),
          B.ndim(),
          B.dims().data(),
          A.template data<T, Context>(),
          B.template data<T, Context>(),
          (bool*)mask,
          ctx());
      math::ApplyMask(
          dY.count(),
          1.f,
          (uint8_t*)mask,
          dY.template data<T, Context>(),
          (T*)buffer,
          ctx());
      reduce_impl_.Setup<T>(dY.dims(), B_broadcast_axes, ctx());
      const auto scratch_size = reduce_impl_.scratch_size();
      reduce_impl_.Compute<T>(
          (T*)buffer,
          dB->ReshapeLike(B)->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(scratch_size),
          ctx());
    }
  }
}

DEPLOY_CNNL_OPERATOR(AddGradient);
DEPLOY_CNNL_OPERATOR(SubGradient);
DEPLOY_CNNL_OPERATOR(MulGradient);
DEPLOY_CNNL_OPERATOR(DivGradient);
DEPLOY_CNNL_OPERATOR(MaximumGradient);
DEPLOY_CNNL_OPERATOR(MinimumGradient);

} // namespace dragon

#endif // USE_MLU
