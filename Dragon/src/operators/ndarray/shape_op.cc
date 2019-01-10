#include "operators/ndarray/shape_op.h"

namespace dragon {

template <class Context>
void ShapeOp<Context>::RunOnDevice() {
    // Reshape
    Output(0)->Reshape({ Input(0).ndim() });

    // Forward
    auto* Ydata = Output(0)->template mutable_data<int64_t, CPUContext>();
    for (int i = 0; i < Input(0).ndim(); i++) Ydata[i] = Input(0).dim(i);
}

DEPLOY_CPU(Shape);
#ifdef WITH_CUDA
DEPLOY_CUDA(Shape);
#endif
OPERATOR_SCHEMA(Shape).NumInputs(1).NumOutputs(1);

NO_GRADIENT(Shape);

}  // namespace dragon