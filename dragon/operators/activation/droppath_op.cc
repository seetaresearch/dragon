#include "dragon/operators/activation/droppath_op.h"
#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void DropPathOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});
  if (phase() == "TEST") {
    Y->ReshapeLike(X)->CopyFrom(X, ctx());
  } else if (phase() == "TRAIN") {
    const auto N = X.dim(0);
    const auto drop_ratio = ratio();
    auto* X_mask = Output("X_mask")->Reshape({X.dim(0)});
    auto* scratch = ctx()->workspace()->template data<float, Context>(N);
    math::RandomUniform(N, 0.f, 1.f, scratch, ctx());
    kernels::DropPath(
        X.dim(0),
        X.stride(0),
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
void DropPathGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0);
  if (phase() == "TRAIN") {
    kernels::DropPathGrad(
        dY.dim(0),
        dY.stride(0),
        1.f / (1.f - ratio()),
        Input("X_mask").template data<uint8_t, Context>(),
        dY.template data<T, Context>(),
        dX->ReshapeLike(dY)->template mutable_data<T, Context>(),
        ctx());
  } else {
    LOG(FATAL) << "Unsupported phase: " << phase();
  }
}

DEPLOY_CPU_OPERATOR(DropPath);
DEPLOY_CPU_OPERATOR(DropPathGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(DropPath);
DEPLOY_CUDA_OPERATOR(DropPathGradient);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(DropPathGradient, DropPathGradient);
#endif
#ifdef USE_MLU
DEPLOY_MLU_OPERATOR(DropPath);
DEPLOY_MLU_OPERATOR(DropPathGradient);
#endif

OPERATOR_SCHEMA(DropPath).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(DropPathGradient)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}});

REGISTER_GRADIENT(DropPath, SimpleGradientMaker);

} // namespace dragon
