#include "dragon/operators/training/update_op_base.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/cast.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
float UpdateOpBase<Context>::param(const string& name) const {
  return ws()
      ->GetTensor(slot_ + "/" + name)
      ->template mutable_data<float, CPUContext>()[0];
}

template <class Context>
template <typename T>
void UpdateOpBase<Context>::Process(Tensor* dX, Tensor* X) {
  // Scale
  auto scale_factor = param("scale_gradient");
  if (scale_factor != 1.f) {
    auto* dx = dX->template mutable_data<T, Context>();
    math::Scale(dX->count(), scale_factor, dx, dx, ctx());
  }
  // Clip
  auto clip_thresh = param("clip_gradient");
  if (clip_thresh > 0.f) {
    T sumsq_grad;
    auto* dx = dX->template mutable_data<T, Context>();
    math::Dot(dX->count(), dx, dx, &sumsq_grad, ctx());
    auto l2_norm = sqrt(cast::to<float>(sumsq_grad));
    if (l2_norm > clip_thresh) {
      math::Scale(dX->count(), clip_thresh / l2_norm, dx, dx, ctx());
    }
  }
  // L2 Decay
  auto l2_decay = param("l2_decay") * decay_mult_;
  if (l2_decay > 0) {
    if (XIsType((*X), float16)) {
      kernel::MixedPrecL2Decay(
          X->count(),
          l2_decay,
          X->template data<float16, Context>(),
          dX->template mutable_data<float, Context>(),
          ctx());
    } else {
      math::Axpy(
          X->count(),
          l2_decay,
          X->template data<T, Context>(),
          dX->template mutable_data<T, Context>(),
          ctx());
    }
  }
}

template <class Context>
template <typename T>
void UpdateOpBase<Context>::Apply(Tensor* dX, Tensor* X) {
  if (XIsType((*X), float16)) {
    kernel::MixedPrecUpdate(
        X->count(),
        dX->template data<float, Context>(),
        X->template mutable_data<float16, Context>(),
        ctx());
  } else {
    math::Axpy(
        X->count(),
        -1.f,
        dX->template data<T, Context>(),
        X->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
void UpdateOpBase<Context>::RunOnDevice() {
  auto &dX = Input(0), *X = Output(0);

  // Skip empty param or grad
  if (dX.count() == 0 || X->count() == 0) return;

  CHECK(dX.dims() == X->dims())
      << "\nParam and grad should have the same dimensions."
      << "\nGot" << X->DimString() << " and " << dX.DimString();

  if (XIsType(dX, float)) {
    Process<float>(&dX, X);
    Compute(&dX);
    Apply<float>(&dX, X);
  } else if (XIsType(dX, float16)) {
    auto* dX_fp32 = ws()->CreateTensor(dX.name() + "/fp32");
    kernel::Cast(
        dX.count(),
        dX.template data<float16, Context>(),
        dX_fp32->ReshapeLike(dX)->template mutable_data<float, Context>(),
        ctx());
    Process<float>(dX_fp32, X);
    Compute(dX_fp32);
    Apply<float>(dX_fp32, X);
  } else {
    LOG(FATAL) << TypeString(dX, {"float16", "float32"});
  }
}

template class UpdateOpBase<CPUContext>;
#ifdef USE_CUDA
template class UpdateOpBase<CUDAContext>;
#endif

} // namespace dragon
