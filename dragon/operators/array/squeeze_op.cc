#include "dragon/core/workspace.h"
#include "dragon/operators/array/reshape_ops.h"

namespace dragon {

template <class Context>
void SqueezeOp<Context>::RunOnDevice() {
  auto &X = Input(0), *Y = Output(0, {0});

  vec64_t out_shape;
  for (int i = 0; i < X.ndim(); i++) {
    bool removed = false;
    if (X.dim(i) == 1) {
      removed = axes_.empty();
      for (auto j : axes_) {
        auto canonical_axis = j < 0 ? j + X.ndim() : j;
        CHECK(canonical_axis >= 0) << "\nExcepted the axis in [-" << X.ndim()
                                   << ", INT_MAX), got " << j << ".";
        if (i == canonical_axis) {
          removed = true;
        }
      }
    }
    if (!removed) {
      out_shape.push_back(X.dim(i));
    }
  }

  // Store for the gradient calculation
  SET_INPUT_SPEC(0);

  // Maybe copy the contents
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
