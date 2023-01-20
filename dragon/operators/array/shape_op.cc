#include "dragon/operators/array/shape_op.h"

namespace dragon {

template <class Context>
void ShapeOp<Context>::RunOnDevice() {
  Output(0)->template CopyFrom<int64_t>(Input(0).dims());
}

DEPLOY_CPU_OPERATOR(Shape);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Shape);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(Shape, Shape);
#endif
#ifdef USE_MLU
DEPLOY_MLU_OPERATOR(Shape);
#endif

OPERATOR_SCHEMA(Shape).NumInputs(1).NumOutputs(1);

NO_GRADIENT(Shape);

} // namespace dragon
