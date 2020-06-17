#include "dragon/operators/array/shape_op.h"

namespace dragon {

template <class Context>
void ShapeOp<Context>::RunOnDevice() {
  Output(0)->Reshape({Input(0).ndim()});

  auto* y = Output(0)->template mutable_data<int64_t, CPUContext>();

  for (int i = 0; i < Input(0).ndim(); i++)
    y[i] = Input(0).dim(i);
}

DEPLOY_CPU(Shape);
#ifdef USE_CUDA
DEPLOY_CUDA(Shape);
#endif

OPERATOR_SCHEMA(Shape)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

NO_GRADIENT(Shape);

} // namespace dragon
