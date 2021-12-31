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
Tensor* UpdateOpBase<Context>::GetState(const string& key) {
  const string& weight_name = Output(weight_index_)->name();
  return workspace()->CreateTensor(name() + "/" + weight_name + "/" + key);
}

template <class Context>
template <typename T>
void UpdateOpBase<Context>::TransformGrad(Tensor* dX) {
  if (grad_scale_ != 1.f) {
    auto* dx = dX->template mutable_data<T, Context>();
    math::Scale(dX->count(), grad_scale_, dx, dx, ctx());
  }
  if (clip_norm_ > 0.f) {
    auto* dx = dX->template mutable_data<T, Context>();
    float norm = std::sqrt(math::Dot(dX->count(), dx, dx, ctx()));
    if (norm > clip_norm_) {
      math::Scale(dX->count(), clip_norm_ / norm, dx, dx, ctx());
    }
  } else if (clip_value_ > 0.f) {
    auto* dx = dX->template mutable_data<T, Context>();
    kernels::Clip(dX->count(), -clip_value_, clip_value_, dx, dx, ctx());
  }
}

template <class Context>
void UpdateOpBase<Context>::RunOnDevice() {
  GetArguments();
  for (weight_index_ = 0; weight_index_ < InputSize(); ++weight_index_) {
    auto &dX = Input(weight_index_), *X = Output(weight_index_);
    if (dX.count() == 0 || X->count() == 0) continue;
    CHECK(dX.dims() == X->dims())
        << "\nWeight and grad should have the same dimensions."
        << "\nGot" << X->DimString() << " and " << dX.DimString();
    if (dX.template IsType<float16>()) {
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
      TransformGrad<float>(X_grad);
      ApplyUpdate(X_grad, X_master, X);
    } else if (dX.template IsType<float>()) {
      TransformGrad<float>(&dX);
      ApplyUpdate(&dX, X, nullptr);
    } else if (dX.template IsType<double>()) {
      TransformGrad<double>(&dX);
      ApplyUpdate(&dX, X, nullptr);
    } else {
      LOG(FATAL) << MessageForUnsupported(
          dtypes::to_string(dX.meta()), {"float16", "float32", "float64"});
    }
  }
}

template class UpdateOpBase<CPUContext>;
#ifdef USE_CUDA
template class UpdateOpBase<CUDAContext>;
#endif

} // namespace dragon
