#include "dragon/core/workspace.h"
#include "dragon/operators/array/reshape_ops.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
void IdentityOp<Context>::RunOnDevice() {
  auto &X = Input(0), *Y = Output(0, {0});
  // Store for the gradient calculation
  STORE_INPUT_SPEC(0);
  // Maybe copy the contents
  Y->ReshapeLike(X)->CopyFrom(X, ctx());
}

DEPLOY_CPU_OPERATOR(Identity);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Identity);
#endif

DEPLOY_CPU_OPERATOR(IdentityGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(IdentityGradient);
#endif

OPERATOR_SCHEMA(Identity)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(IdentityGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1)
    /* dY => dX */
    .AllowInplace({{0, 0}});

REGISTER_GRADIENT(Identity, SimpleGradientMaker);

} // namespace dragon
