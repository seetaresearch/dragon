#include "dragon/operators/array/reshape_op.h"

namespace dragon {

template <class Context>
void UnsqueezeOp<Context>::RunOnDevice() {
  auto &X = Input(0), *Y = Output(0, {0});
  Output("X_spec")->ReshapeLike(X);

  auto out_shape = vec64_t(X.ndim() + axes_.size());
  auto out_rank = (int64_t)out_shape.size();
  for (auto i : axes_) {
    CHECK(i >= -out_rank && i < out_rank)
        << "\nExcepted the axis in [-" << out_rank << ", " << out_rank
        << "), got " << i << ".";
    auto canonical_axis = i < 0 ? i + out_rank : i;
    out_shape[canonical_axis] = -1;
  }
  int64_t j = 0;
  for (size_t i = 0; i < out_shape.size(); i++) {
    out_shape[i] = out_shape[i] < 0 ? 1 : X.dim(j++);
  }

  Y->Reshape(out_shape)->CopyFrom(X, ctx());
}

DEPLOY_CPU_OPERATOR(Unsqueeze);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Unsqueeze);
#endif

DEPLOY_CPU_OPERATOR(UnsqueezeGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(UnsqueezeGradient);
#endif

OPERATOR_SCHEMA(Unsqueeze)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(UnsqueezeGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1)
    /* dY => dX */
    .AllowInplace({{0, 0}});

REGISTER_GRADIENT(Unsqueeze, SimpleGradientMaker);

} // namespace dragon
