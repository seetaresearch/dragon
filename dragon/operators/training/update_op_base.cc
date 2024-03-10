#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/operators/training/update_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
T UpdateOpBase<Context>::GetHyper(const string& key) {
  auto* X = workspace()->GetTensor(name() + "/" + key);
  return X->template data<T, CPUContext>()[0];
}

template <class Context>
Tensor* UpdateOpBase<Context>::GetState(const string& key) {
  const string& src_name = Output(src_index_)->name();
  return workspace()->CreateTensor(name() + "/" + src_name + "/" + key);
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
template <typename T>
void UpdateOpBase<Context>::DoRunWithTensor(Tensor* dX, Tensor* X) {
  TransformGrad<T>(dX);
  ApplyUpdate(dX, X, nullptr);
}

template <class Context>
template <typename T>
void UpdateOpBase<Context>::DoRunWithTensor(Tensor* dX, Tensor* X, Tensor* Y) {
  using AccT = float;
  auto* X_master = workspace()->CreateTensor(X->name() + "/master");
  auto* dX_master = ctx()->workspace()->CreateTensor("BufferShared");
  if (X_master->count() != X->count()) {
    auto* x = X_master->ReshapeLike(*X)->template mutable_data<AccT, Context>();
    math::Cast(X->count(), X->template data<T, Context>(), x, ctx());
  }
  auto* dx = dX_master->ReshapeLike(*X)->template mutable_data<AccT, Context>();
  math::Cast(dX->count(), dX->template data<T, Context>(), dx, ctx());
  TransformGrad<AccT>(dX_master);
  ApplyUpdate(dX_master, X_master, Y);
}

template <class Context>
void UpdateOpBase<Context>::RunOnDevice() {
  GetArguments();
  for (src_index_ = 0; src_index_ < InputSize(); ++src_index_) {
    auto &dX = Input(src_index_), *X = Output(src_index_);
    if (dX.count() == 0 || X->count() == 0) continue;
    CHECK(dX.dims() == X->dims())
        << "\nGrad and Param should have the same dimensions."
        << "\nGot" << dX.DimString() << " and " << X->DimString();
    if (dX.template IsType<float16>()) {
      DoRunWithTensor<float16>(&dX, X, X);
    } else if (dX.template IsType<bfloat16>()) {
      DoRunWithTensor<bfloat16>(&dX, X, X);
    } else if (dX.template IsType<float>()) {
      DoRunWithTensor<float>(&dX, X);
    } else if (dX.template IsType<double>()) {
      DoRunWithTensor<double>(&dX, X);
    } else {
      vector<string> supps({"float16", "bfloat16", "float32", "float64"});
      LOG(FATAL) << MessageForUnsupported(dtypes::to_string(dX.meta()), supps);
    }
  }
}

template class UpdateOpBase<CPUContext>;
#ifdef USE_CUDA
template class UpdateOpBase<CUDAContext>;
#endif
#ifdef USE_MPS
template class UpdateOpBase<MPSContext>;
#endif
#ifdef USE_MLU
template class UpdateOpBase<MLUContext>;
#endif

} // namespace dragon
