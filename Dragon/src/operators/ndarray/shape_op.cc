#include "operators/ndarray/shape_op.h"

namespace dragon {

template <class Context>
void ShapeOp<Context>::RunOnDevice() {
    //  reshape
    output(0)->Reshape(vector<TIndex>(1, input(0).ndim()));

    //  forward
    auto* Ydata = output(0)->template mutable_data<float, CPUContext>();
    for (int i = 0; i < input(0).ndim(); i++) Ydata[i] = input(0).dim(i);
}

DEPLOY_CPU(Shape);
#ifdef WITH_CUDA
DEPLOY_CUDA(Shape);
#endif
OPERATOR_SCHEMA(Shape).NumInputs(1).NumOutputs(1);

NO_GRADIENT(Shape);

}    // namespace dragon