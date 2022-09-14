/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_MATH_ELEMENTWISE_GRADIENT_IMPL_H_
#define DRAGON_OPERATORS_MATH_ELEMENTWISE_GRADIENT_IMPL_H_

#include "dragon/core/workspace.h"
#include "dragon/operators/math/elementwise_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void AddGradientOp<Context>::DoRunWithType() {
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
      math::ReduceSum(
          dY.ndim(),
          dY.dims().data(),
          A_broadcast_axes.size(),
          A_broadcast_axes.data(),
          1.f,
          dY.template data<T, Context>(),
          dA->ReshapeLike(A_spec)->template mutable_data<T, Context>(),
          ctx());
    }
  }

  if (dB->has_name()) {
    if (B_broadcast_axes.empty()) {
      dB->ReshapeLike(B_spec)->CopyFrom(dY, ctx());
    } else {
      math::ReduceSum(
          dY.ndim(),
          dY.dims().data(),
          B_broadcast_axes.size(),
          B_broadcast_axes.data(),
          1.f,
          dY.template data<T, Context>(),
          dB->ReshapeLike(B_spec)->template mutable_data<T, Context>(),
          ctx());
    }
  }
}

template <class Context>
template <typename T>
void SubGradientOp<Context>::DoRunWithType() {
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
      math::ReduceSum(
          dY.ndim(),
          dY.dims().data(),
          A_broadcast_axes.size(),
          A_broadcast_axes.data(),
          1.f,
          dY.template data<T, Context>(),
          dA->ReshapeLike(A_spec)->template mutable_data<T, Context>(),
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
      math::ReduceSum(
          dY.ndim(),
          dY.dims().data(),
          B_broadcast_axes.size(),
          B_broadcast_axes.data(),
          -1.f,
          dY.template data<T, Context>(),
          dB->ReshapeLike(B_spec)->template mutable_data<T, Context>(),
          ctx());
    }
  }
}

template <class Context>
template <typename T>
void MulGradientOp<Context>::DoRunWithType() {
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
            scratch,
            ctx());
      } else {
        math::Mul(
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
  }
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
template <typename T>
void MinimumGradientOp<Context>::DoRunWithType() {
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
        math::Less(
            A.count(),
            A.template data<T, Context>(),
            B.template data<T, Context>(),
            mask,
            ctx());
      } else {
        math::Less(
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
      math::Less(
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
        math::GreaterEqual(
            A.count(),
            A.template data<T, Context>(),
            B.template data<T, Context>(),
            mask,
            ctx());
      } else {
        math::GreaterEqual(
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
      math::GreaterEqual(
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

} // namespace dragon

#endif // DRAGON_OPERATORS_MATH_ELEMENTWISE_GRADIENT_IMPL_H_
