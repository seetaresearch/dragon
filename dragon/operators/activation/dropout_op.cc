#include "dragon/operators/activation/dropout_op.h"
#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void DropoutOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});
  if (phase() == "TEST") {
    Y->ReshapeLike(X)->CopyFrom(X, ctx());
  } else if (phase() == "TRAIN") {
    const auto N = X.count();
    const auto drop_ratio = ratio();
    auto* X_mask = Output("X_mask")->ReshapeLike(X);
    auto* scratch = ctx()->workspace()->template data<float, Context>(N);
    math::RandomUniform(N, 0.f, 1.f, scratch, ctx());
    kernels::Dropout(
        X.count(),
        drop_ratio,
        1.f / (1.f - drop_ratio),
        scratch,
        X.template data<T, Context>(),
        Y->ReshapeLike(X)->template mutable_data<T, Context>(),
        X_mask->template mutable_data<uint8_t, Context>(),
        ctx());
  } else {
    LOG(FATAL) << "Unsupported phase: " << phase();
  }
}

template <class Context>
template <typename T>
void DropoutGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0);
  if (phase() == "TRAIN") {
    math::ApplyMask(
        dY.count(),
        1.f / (1.f - ratio()),
        Input("X_mask").template data<uint8_t, Context>(),
        dY.template data<T, Context>(),
        dX->ReshapeLike(dY)->template mutable_data<T, Context>(),
        ctx());
  } else {
    LOG(FATAL) << "Unsupported phase: " << phase();
  }
}

DEPLOY_CPU_OPERATOR(Dropout);
DEPLOY_CPU_OPERATOR(DropoutGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Dropout);
DEPLOY_CUDA_OPERATOR(DropoutGradient);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(DropoutGradient, DropoutGradient);
#endif
#ifdef USE_MLU
DEPLOY_MLU_OPERATOR(Dropout);
DEPLOY_MLU_OPERATOR(DropoutGradient);
#endif

DEFINE_OP_SINGLE_ARG(float, DropoutOp, ratio);
DEFINE_OP_SINGLE_ARG(float, DropoutGradientOp, ratio);

OPERATOR_SCHEMA(Dropout).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(DropoutGradient)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}});

REGISTER_GRADIENT(Dropout, SimpleGradientMaker);

} // namespace dragon
