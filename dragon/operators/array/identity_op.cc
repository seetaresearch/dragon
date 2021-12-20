#include "dragon/operators/array/reshape_op.h"

namespace dragon {

DEPLOY_CPU_OPERATOR(Identity);
REGISTER_CPU_OPERATOR(IdentityGradient, IdentityOp<CPUContext>);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Identity);
REGISTER_CUDA_OPERATOR(IdentityGradient, IdentityOp<CUDAContext>);
#endif

OPERATOR_SCHEMA(Identity).AllowInplace([](int, int) -> bool { return true; });
OPERATOR_SCHEMA(IdentityGradient).AllowInplace([](int, int) -> bool {
  return true;
});

REGISTER_GRADIENT(Identity, SimpleGradientMaker);

} // namespace dragon
