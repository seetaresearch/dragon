#include "dragon/operators/activation/dropout_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void DropoutOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});
  if (phase() == "TEST") {
    Y->ReshapeLike(X)->CopyFrom(X, ctx());
  } else if (phase() == "TRAIN") {
    auto drop_ratio = ratio();
    auto* X_mask = Output("X_mask")->ReshapeLike(X);
    kernels::Dropout(
        X.count(),
        drop_ratio,
        1.f / (1.f - drop_ratio),
        X.template data<T, Context>(),
        Y->ReshapeLike(X)->template mutable_data<T, Context>(),
        X_mask->template mutable_data<uint8_t, Context>(),
        ctx()->workspace()->template data<uint32_t, Context>(X.count()),
        ctx());
  } else {
    LOG(FATAL) << "Unknown Phase: " << phase();
  }
}

template <class Context>
void DropoutOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void DropoutGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0);
  if (phase() == "TEST") {
    NOT_IMPLEMENTED;
  } else if (phase() == "TRAIN") {
    math::ApplyMask(
        dY.count(),
        1.f / (1.f - ratio()),
        Input("X_mask").template data<uint8_t, Context>(),
        dY.template data<T, Context>(),
        dX->ReshapeLike(dY)->template mutable_data<T, Context>(),
        ctx());
  } else {
    LOG(FATAL) << "Unknown Phase: " << phase();
  }
}

template <class Context>
void DropoutGradientOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Dropout);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Dropout);
#endif

DEPLOY_CPU_OPERATOR(DropoutGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(DropoutGradient);
#endif

OPERATOR_SCHEMA(Dropout)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(DropoutGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1)
    /* dY => dX */
    .AllowInplace({{0, 0}});

REGISTER_GRADIENT(Dropout, SimpleGradientMaker);

} // namespace dragon
