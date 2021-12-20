#include "dragon/core/workspace.h"
#include "dragon/operators/training/update_op.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
T UpdateOpBase<Context>::GetHyper(const string& key) {
  auto* X = workspace()->GetTensor(name() + "/" + key);
  return X->template data<T, CPUContext>()[0];
}

template <class Context>
Tensor* UpdateOpBase<Context>::Slot(const string& key) {
  const string& weight_name = Output(weight_index_)->name();
  return workspace()->CreateTensor(name() + "/" + weight_name + "/" + key);
}

template <class Context>
template <typename T>
void UpdateOpBase<Context>::TransformGrad(Tensor* dX, Tensor* X) {
  // Scale.
  if (grad_scale_ != 1.f) {
    auto* dx = dX->template mutable_data<T, Context>();
    math::Scale(dX->count(), grad_scale_, dx, dx, ctx());
  }
  // Clip.
  if (clip_norm_ > 0.f) {
    auto* dx = dX->template mutable_data<T, Context>();
    float grad_norm = std::sqrt(math::Dot(dX->count(), dx, dx, ctx()));
    if (grad_norm > clip_norm_) {
      math::Scale(dX->count(), clip_norm_ / grad_norm, dx, dx, ctx());
    }
  } else if (clip_value_ > 0.f) {
    auto* dx = dX->template mutable_data<T, Context>();
    kernels::Clip(dX->count(), -clip_value_, clip_value_, dx, dx, ctx());
  }
  // Penalty.
  if (weight_decay_ > 0.f) {
    math::Axpy(
        X->count(),
        weight_decay_,
        X->template data<T, Context>(),
        dX->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
template <typename T>
void UpdateOpBase<Context>::ApplyUpdate(Tensor* dX, Tensor* X) {
  math::Sub(
      X->count(),
      X->template data<T, Context>(),
      dX->template data<T, Context>(),
      X->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void UpdateOpBase<Context>::RunOnDevice() {
  GetArguments();
  for (int i = 0; i < InputSize(); ++i) {
    weight_index_ = i;
    auto &dX = Input(i), *X = Output(i);
    if (dX.count() == 0 || X->count() == 0) return;
    CHECK(dX.dims() == X->dims())
        << "\nWeight and grad should have the same dimensions."
        << "\nGot" << X->DimString() << " and " << dX.DimString();
    if (dX.template IsType<float>()) {
      TransformGrad<float>(&dX, X);
      ComputeUpdate(&dX, X);
      ApplyUpdate<float>(&dX, X);
    } else if (dX.template IsType<float16>()) {
      auto* X_master = workspace()->CreateTensor(X->name() + "_master");
      auto* X_grad = ctx()->workspace()->CreateTensor("BufferShared");
      if (X_master->count() != X->count()) {
        math::Cast(
            X->count(),
            X->template data<float16, Context>(),
            X_master->ReshapeLike(*X)->template mutable_data<float, Context>(),
            ctx());
      }
      math::Cast(
          dX.count(),
          dX.template data<float16, Context>(),
          X_grad->ReshapeLike(dX)->template mutable_data<float, Context>(),
          ctx());
      TransformGrad<float>(X_grad, X_master);
      ComputeUpdate(X_grad, X_master);
      ApplyUpdate<float>(X_grad, X_master);
      math::Cast(
          X->count(),
          X_master->template data<float, Context>(),
          X->template mutable_data<float16, Context>(),
          ctx());
    } else {
      LOG(FATAL) << MessageForUnsupported(
          dtypes::to_string(dX.meta()), {"float16", "float32"});
    }
  }
}

template class UpdateOpBase<CPUContext>;
#ifdef USE_CUDA
template class UpdateOpBase<CUDAContext>;
#endif

} // namespace dragon
