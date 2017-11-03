#include "operators/cast/float2half_op.h"
#include "core/workspace.h"
#include "utils/op_kernel.h"

namespace dragon {

#ifdef WITH_CUDA_FP16

template <class Context>
void FloatToHalfOp<Context>::RunOnDevice() {
    CHECK(input(0).template IsType<float>())
        << "The type of input should be float32.";
    output(0)->ReshapeLike(input(0));
    
    //  cast
    auto* Xdata = input(0).template data<float, Context>();
    auto* Ydata = output(0)->template mutable_data<float16, Context>();
    kernel::Float2Half<float, Context>(output(0)->count(), Xdata, Ydata);

    //  release & share
    input(0).Reset();
    input(0).ReshapeLike(*output(0));
    input(0).Share(*output(0));
}

#ifdef WITH_CUDA
DEPLOY_CUDA(FloatToHalf);
#endif
OPERATOR_SCHEMA(FloatToHalf).NumInputs(1).NumOutputs(1);

NO_GRADIENT(FloatToHalf);

#endif

}    // namespace dragon