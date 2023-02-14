#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/math/affine_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLAffineGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), &dY = Input(2);
  auto *dX = Output(0), *dW = Output(1), *dB = Output(2);

  // Compute axes.
  vec64_t reduce_axes, affine_axes;
  for (int i = 0; i < X.ndim(); ++i) {
    bool keep = true;
    for (auto axis : axes_) {
      axis = axis < 0 ? axis + X.ndim() : axis;
      if (axis == i) keep = false;
      if (i == 0) affine_axes.push_back(axis);
    }
    if (keep) reduce_axes.push_back(i);
  }

  // Scratch to save the intermediates.
  T* data = nullptr;
  if (dW->has_name() && X.count() != W.count()) {
    data = ctx()->workspace()->template data<T, Context>(
        X.count(), "BufferKernel");
  }

  // dW = dY * X
  if (dW->has_name()) {
    if (X.count() == W.count()) {
      math::Mul(
          X.count(),
          dY.template data<T, Context>(),
          X.template data<T, Context>(),
          dW->ReshapeLike(W)->template mutable_data<T, Context>(),
          ctx());
    } else {
      math::Mul(
          X.count(),
          dY.template data<T, Context>(),
          X.template data<T, Context>(),
          data,
          ctx());
      dW_impl_.Setup<T>(X.dims(), reduce_axes, ctx());
      dW_impl_.Compute<T>(
          data,
          dW->ReshapeLike(W)->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(dW_impl_.scratch_size()),
          ctx());
    }
  }

  // dB = dY
  if (dB->has_name()) {
    if (X.count() == W.count()) {
      dB->ReshapeLike(W)->CopyFrom(dY, ctx());
    } else {
      dB_impl_.Setup<T>(X.dims(), reduce_axes, ctx());
      dB_impl_.Compute<T>(
          dY.template data<T, Context>(),
          dB->ReshapeLike(W)->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(dB_impl_.scratch_size()),
          ctx());
    }
  }

  // dX = dY * W
  if (dX->has_name()) {
    math::Affine(
        X.ndim(),
        X.dims().data(),
        affine_axes.size(),
        affine_axes.data(),
        dY.template data<T, Context>(),
        W.template data<T, Context>(),
        (const T*)nullptr,
        dX->ReshapeLike(X)->template mutable_data<T, Context>(),
        ctx());
  }
}

DEPLOY_CNNL_OPERATOR(AffineGradient);

} // namespace dragon

#endif // USE_MLU
