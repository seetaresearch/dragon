#include "dragon/core/workspace.h"
#include "dragon/operators/training/update_ops.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
Tensor* UpdateOpBase<Context>::Slot(const string& name) {
  return Buffer(Output(0)->name() + "/" + name);
}

template <class Context>
float UpdateOpBase<Context>::Parameter(const string& name) const {
  return workspace()
      ->GetTensor("/share/hyper/" + handle() + "/" + name)
      ->template mutable_data<float, CPUContext>()[0];
}

template <class Context>
template <typename T>
void UpdateOpBase<Context>::AdjustGradient(Tensor* dX, Tensor* X) {
  // Scale
  auto scale = Parameter("scale");
  if (scale != 1.f) {
    auto* dx = dX->template mutable_data<T, Context>();
    math::Scale(dX->count(), scale, dx, dx, ctx());
  }
  // Clip
  auto clip_norm = Parameter("clip_norm");
  if (clip_norm > 0.f) {
    auto* dx = dX->template mutable_data<T, Context>();
    auto grad_norm = std::sqrt(math::Dot(dX->count(), dx, dx, ctx()));
    if (grad_norm > clip_norm) {
      math::Scale(dX->count(), clip_norm / grad_norm, dx, dx, ctx());
    }
  }
  // Penalty
  auto weight_decay = Parameter("weight_decay");
  if (weight_decay > 0.f && decay_mult_ > 0.f) {
    math::Axpy(
        X->count(),
        weight_decay * decay_mult_,
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
  auto &dX = Input(0), *X = Output(0);

  // Skip empty param or grad
  if (dX.count() == 0 || X->count() == 0) return;

  CHECK(dX.dims() == X->dims())
      << "\nParam and grad should have the same dimensions."
      << "\nGot" << X->DimString() << " and " << dX.DimString();

  if (dX.template IsType<float>()) {
    AdjustGradient<float>(&dX, X);
    ComputeUpdate(&dX);
    ApplyUpdate<float>(&dX, X);
  } else if (dX.template IsType<float16>()) {
    auto* X_master = workspace()->CreateTensor(X->name() + "[float32]");
    auto* dX_copy = ctx()->workspace()->CreateTensor("/share/buffer/data:0");
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
        dX_copy->ReshapeLike(dX)->template mutable_data<float, Context>(),
        ctx());
    AdjustGradient<float>(dX_copy, X_master);
    ComputeUpdate(dX_copy);
    ApplyUpdate<float>(dX_copy, X_master);
    math::Cast(
        X->count(),
        X_master->template data<float, Context>(),
        X->template mutable_data<float16, Context>(),
        ctx());
  } else {
    LOG(FATAL) << MessageForUnsupported(
        types::to_string(dX.meta()), {"float16", "float32"});
  }
}

template class UpdateOpBase<CPUContext>;
#ifdef USE_CUDA
template class UpdateOpBase<CUDAContext>;
#endif

} // namespace dragon
