#include "dragon/operators/vision/bias_add_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/filler.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void BiasAddOp<Context>::DoRunWithType() {
  auto &X = Input(0), &B = Input(1), *Y = Output(0, {0});

  int64_t N, C, S;
  if (data_format() == "NCHW") {
    N = X.dim(0), C = X.dim(1), S = X.count(2);
  } else if (data_format() == "NHWC") {
    N = X.count() / X.dim(-1), C = X.dim(-1), S = 1;
  } else {
    LOG(FATAL) << "Unknown DataFormat: " << data_format();
  }

  TENSOR_FILL(B, vec64_t({C}));
  kernel::BiasAdd(
      N,
      C,
      S,
      X.template data<T, Context>(),
      B.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void BiasAddOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void BiasAddGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0), *dB = Output(1);

  if (dX->has_name()) {
    dX->ReshapeLike(dY)->CopyFrom(dY, ctx());
  }

  if (dB->has_name()) {
    vec32_t dims, axes;
    if (data_format() == "NCHW") {
      dims = {(int)dY.dim(0), (int)dY.dim(1), (int)dY.count(2)};
      axes = {0, 2};
      dB->Reshape({dY.dim(1)});
    } else if (data_format() == "NHWC") {
      dims = {(int)(dY.count() / dY.dim(-1)), (int)dY.dim(-1)};
      axes = {0};
      dB->Reshape({dY.dim(-1)});
    }
    math::ReduceSum(
        dims.size(),
        dims.data(),
        axes.size(),
        axes.data(),
        1.f,
        dY.template data<T, Context>(),
        dB->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
void BiasAddGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU(BiasAdd);
#ifdef USE_CUDA
DEPLOY_CUDA(BiasAdd);
#endif

DEPLOY_CPU(BiasAddGradient);
#ifdef USE_CUDA
DEPLOY_CUDA(BiasAddGradient);
#endif

OPERATOR_SCHEMA(BiasAdd)
    /* X, B */
    .NumInputs(2)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(BiasAddGradient)
    /* dY */
    .NumInputs(1)
    /* dX, dB */
    .NumOutputs(2)
    /* dY => dX */
    .AllowInplace({{0, 0}});

REGISTER_GRADIENT(BiasAdd, SimpleGradientMaker);

} // namespace dragon
