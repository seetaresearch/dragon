#include "dragon/operators/array/shuffle_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void ChannelShuffleOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);

  CHECK_EQ(X.dim(axis) % group_, 0)
      << "\nThe " << X.dim(axis) << " channels "
      << "can not be split into " << group_ << " groups.";
  auto G = group_, K = X.dim(axis) / group_;
  if (def().type() == "ChannelShuffleGradient") std::swap(G, K);

  math::Transpose(
      4,
      vec64_t({X.count(0, axis), G, K, X.count(axis + 1)}).data(),
      vec64_t({0, 2, 1, 3}).data(),
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

DEPLOY_CPU_OPERATOR(ChannelShuffle);
REGISTER_CPU_OPERATOR(ChannelShuffleGradient, ChannelShuffleOp<CPUContext>);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(ChannelShuffle);
REGISTER_CUDA_OPERATOR(ChannelShuffleGradient, ChannelShuffleOp<CUDAContext>);
#endif

OPERATOR_SCHEMA(ChannelShuffle).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(ChannelShuffleGradient).NumInputs(1).NumOutputs(1);

REGISTER_GRADIENT(ChannelShuffle, SimpleGradientMaker);

} // namespace dragon
