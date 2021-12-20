#include "dragon/operators/array/reshape_op.h"

namespace dragon {

template <class Context>
void SqueezeOp<Context>::RunOnDevice() {
  auto &X = Input(0), *Y = Output(0, {0});
  Output("X_spec")->ReshapeLike(X);

  vec64_t out_shape;
  for (int i = 0; i < X.ndim(); i++) {
    if (X.dim(i) == 1) {
      bool removed = axes_.empty();
      for (auto j : axes_) {
        auto axis = j < 0 ? j + X.ndim() : j;
        CHECK(axis >= 0) << "\nExcepted the axis in [-" << X.ndim()
                         << ", INT_MAX), got " << j << ".";
        removed = (i == axis ? true : removed);
      }
      if (removed) continue;
    }
    out_shape.push_back(X.dim(i));
  }

  Y->Reshape(out_shape)->CopyFrom(X, ctx());
}

DEPLOY_CPU_OPERATOR(Squeeze);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Squeeze);
#endif

DEPLOY_CPU_OPERATOR(SqueezeGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SqueezeGradient);
#endif

OPERATOR_SCHEMA(Squeeze)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(SqueezeGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1)
    /* dY => dX */
    .AllowInplace({{0, 0}});

REGISTER_GRADIENT(Squeeze, SimpleGradientMaker);

} // namespace dragon
