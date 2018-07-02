#include "operators/ndarray/shape_op.h"

namespace dragon {

template <class Context>
void ShapeOp<Context>::RunOnDevice() {
    //  reshape
    Output(0)->Reshape({ (TIndex)Input(0).ndim() });

    //  forward
    auto* Ydata = Output(0)->template mutable_data<int, CPUContext>();
    for (int i = 0; i < Input(0).ndim(); i++) Ydata[i] = Input(0).dim(i);
}

DEPLOY_CPU(Shape);
#ifdef WITH_CUDA
DEPLOY_CUDA(Shape);
#endif
OPERATOR_SCHEMA(Shape).NumInputs(1).NumOutputs(1);

NO_GRADIENT(Shape);

}    // namespace dragon