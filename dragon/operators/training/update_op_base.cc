#include "dragon/core/workspace.h"
#include "dragon/operators/training/update_ops.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
float UpdateOpBase<Context>::Hyper(const string& name) {
  auto* hyper = workspace()->GetTensor(handle() + "/" + name);
  return hyper->template mutable_data<float, CPUContext>()[0];
}

template <class Context>
Tensor* UpdateOpBase<Context>::Slot(const string& name) {
  const string& var_name = Output(input_index_)->name();
  return workspace()->CreateTensor(handle() + "/" + var_name + "/" + name);
}

template <class Context>
template <typename T>
void UpdateOpBase<Context>::AdjustGradient(Tensor* dX, Tensor* X) {
  // Scale
  if (scale_ != 1.f) {
    auto* dx = dX->template mutable_data<T, Context>();
    math::Scale(dX->count(), scale_, dx, dx, ctx());
  }
  // Clip
  if (clip_norm_ > 0.f) {
    auto* dx = dX->template mutable_data<T, Context>();
    auto norm = std::sqrt(math::Dot(dX->count(), dx, dx, ctx()));
    if (norm > clip_norm_) {
      math::Scale(dX->count(), clip_norm_ / norm, dx, dx, ctx());
    }
  }
  // Penalty
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
    auto &dX = Input(i), *X = Output(i);
    if (dX.count() == 0 || X->count() == 0) return;
    CHECK(dX.dims() == X->dims())
        << "\nParam and grad should have the same dimensions."
        << "\nGot" << X->DimString() << " and " << dX.DimString();
    input_index_ = i;
    if (dX.template IsType<float>()) {
      AdjustGradient<float>(&dX, X);
      ComputeUpdate(&dX, X);
      ApplyUpdate<float>(&dX, X);
    } else if (dX.template IsType<float16>()) {
      auto* X_master = workspace()->CreateTensor(X->name() + "_master");
      auto* dX_copy = ctx()->workspace()->CreateTensor("shared/buffer/data:0");
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
      ComputeUpdate(dX_copy, X_master);
      ApplyUpdate<float>(dX_copy, X_master);
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
