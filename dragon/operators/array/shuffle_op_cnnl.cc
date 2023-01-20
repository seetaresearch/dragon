#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/array/shuffle_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLChannelShuffleOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);

  CHECK_EQ(X.dim(axis) % group_, 0)
      << "\nThe " << X.dim(axis) << " channels "
      << "can not be split into " << group_ << " groups.";
  auto G = group_, K = X.dim(axis) / group_;
  if (def().type() == "ChannelShuffleGradient") std::swap(G, K);

  const vec64_t X_dims({X.count(0, axis), G, K, X.count(axis + 1)});
  impl_.Setup<T>(X_dims, {0, 2, 1, 3}, ctx());
  impl_.Compute<T>(
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx()->workspace()->template data<Context>(impl_.scratch_size()),
      ctx());
}

DEPLOY_CNNL_OPERATOR(ChannelShuffle);
REGISTER_CNNL_OPERATOR(
    ChannelShuffleGradient,
    CNNLChannelShuffleOp<MLUContext>);

} // namespace dragon

#endif // USE_MLU
