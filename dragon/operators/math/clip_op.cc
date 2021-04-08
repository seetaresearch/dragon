#include "dragon/operators/math/clip_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
pair<float, float> ClipOp<Context>::ComputeBoundsWithType() {
  auto meta = TypeMeta::Make<T>();
  if (meta.template Match<int8_t>()) {
    return std::make_pair(std::max(low_, -128.f), std::min(high_, 127.f));
  } else if (meta.template Match<uint8_t>()) {
    return std::make_pair(std::max(low_, 0.f), std::min(high_, 255.f));
  } else if (meta.template Match<int>()) {
    return std::make_pair(
        std::max(low_, -214748e4f), std::min(high_, 214748e4f));
  } else if (meta.template Match<int64_t>()) {
    return std::make_pair(
        std::max(low_, -922337e13f), std::min(high_, 922337e13f));
  } else if (meta.template Match<float16>()) {
    return std::make_pair(std::max(low_, -65505.f), std::min(high_, 65504.f));
  } else {
    return std::make_pair(std::max(low_, -FLT_MAX), std::min(high_, FLT_MAX));
  }
}

template <class Context>
template <typename T>
void ClipOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  auto bounds = ComputeBoundsWithType<T>();
  kernels::Clip(
      X.count(),
      bounds.first,
      bounds.second,
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void ClipOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Numerical>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void ClipGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dY = Input(1), *dX = Output(0);
  auto bounds = this->template ComputeBoundsWithType<T>();
  kernels::ClipGrad(
      X.count(),
      bounds.first,
      bounds.second,
      dY.template data<T, Context>(),
      X.template data<T, Context>(),
      dX->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void ClipGradientOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Clip);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Clip);
#endif

DEPLOY_CPU_OPERATOR(ClipGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(ClipGradient);
#endif

OPERATOR_SCHEMA(Clip)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(ClipGradient)
    /* X, dY */
    .NumInputs(2)
    /* X, dX */
    .NumOutputs(1);

REGISTER_GRADIENT(Clip, GenericGradientMaker);

} // namespace dragon
