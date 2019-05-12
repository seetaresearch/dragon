#include "operators/array/shape_op.h"

namespace dragon {

template <class Context>
void ShapeOp<Context>::RunOnDevice() {
    Y(0)->Reshape({ X(0).ndim() });

    auto* y = Y(0)->template
        mutable_data<int64_t, CPUContext>();

    for (int i = 0; i < X(0).ndim(); i++)
        y[i] = X(0).dim(i);
}

DEPLOY_CPU(Shape);
#ifdef WITH_CUDA
DEPLOY_CUDA(Shape);
#endif

OPERATOR_SCHEMA(Shape)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

NO_GRADIENT(Shape);

}  // namespace dragon