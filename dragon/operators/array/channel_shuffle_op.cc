#include "dragon/operators/array/channel_shuffle_op.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void ChannelShuffleOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  CANONICALIZE_AXIS_WITH_TENSOR(X);

  CHECK_EQ(X.dim(axis) % group_, 0)
      << "\nThe " << X.dim(axis) << " channels "
      << "can not be split into " << group_ << " groups.";

  kernel::ChannelShuffle(
      X.count(0, axis),
      X.count(axis + 1),
      X.dim(axis),
      group_,
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void ChannelShuffleOp<Context>::RunOnDevice() {
  DispatchHelper<AllTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void ChannelShuffleGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0);
  CANONICALIZE_AXIS_WITH_TENSOR(dY);

  kernel::ChannelShuffle(
      dY.count(0, axis),
      dY.count(axis + 1),
      dY.dim(axis),
      dY.dim(axis) / group_,
      dY.template data<T, Context>(),
      dX->ReshapeLike(dY)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void ChannelShuffleGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU(ChannelShuffle);
#ifdef USE_CUDA
DEPLOY_CUDA(ChannelShuffle);
#endif

DEPLOY_CPU(ChannelShuffleGradient);
#ifdef USE_CUDA
DEPLOY_CUDA(ChannelShuffleGradient);
#endif

OPERATOR_SCHEMA(ChannelShuffle)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(ChannelShuffleGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(ChannelShuffle, SimpleGradientMaker);

} // namespace dragon
