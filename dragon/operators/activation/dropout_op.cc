#include "dragon/operators/activation/dropout_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void DropoutOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});
  auto scale = use_scale_ ? 1.f / (1.f - prob()) : 1.f;

  if (phase() == "TEST") {
    Y->ReshapeLike(X)->CopyFrom(X, ctx());
    if (!use_scale_) {
      math::Scale(
          X.count(),
          1.f - prob(),
          Y->template data<T, Context>(),
          Y->template mutable_data<T, Context>(),
          ctx());
    }
  } else if (phase() == "TRAIN") {
    Buffer("mask")->ReshapeLike(X);
    kernel::Dropout(
        X.count(),
        prob(),
        scale,
        X.template data<T, Context>(),
        Buffer("mask")->template mutable_data<uint8_t, Context>(),
        Y->ReshapeLike(X)->template mutable_data<T, Context>(),
        ws()->template data<uint32_t, Context>({X.count()})[0],
        ctx());
  } else {
    LOG(FATAL) << "Unknown Phase: " << phase();
  }
}

template <class Context>
void DropoutOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void DropoutGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0);
  auto scale = use_scale_ ? 1.f / (1.f - prob()) : 1.f;

  if (phase() == "TEST") {
    NOT_IMPLEMENTED;
  } else if (phase() == "TRAIN") {
    kernel::ApplyMask(
        dY.count(),
        scale,
        dY.template data<T, Context>(),
        Buffer("mask")->template data<uint8_t, Context>(),
        dX->ReshapeLike(dY)->template mutable_data<T, Context>(),
        ctx());
  } else {
    LOG(FATAL) << "Unknown Phase: " << phase();
  }
}

template <class Context>
void DropoutGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU(Dropout);
#ifdef USE_CUDA
DEPLOY_CUDA(Dropout);
#endif

DEPLOY_CPU(DropoutGradient);
#ifdef USE_CUDA
DEPLOY_CUDA(DropoutGradient);
#endif

OPERATOR_SCHEMA(Dropout)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .Inplace({{0, 0}});

OPERATOR_SCHEMA(DropoutGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1)
    /* dY => dX */
    .Inplace({{0, 0}});

REGISTER_GRADIENT(Dropout, SimpleGradientMaker);

} // namespace dragon
