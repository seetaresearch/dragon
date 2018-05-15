#include "operators/cast/float2half_op.h"
#include "core/workspace.h"
#include "utils/op_kernel.h"

namespace dragon {

#ifdef WITH_CUDA_FP16

template <class Context>
void FloatToHalfOp<Context>::RunOnDevice() {
    CHECK(Input(0).template IsType<float>())
        << "The type of input should be float32.";
    Output(0)->ReshapeLike(Input(0));
    
    //  cast
    auto* Xdata = Input(0).template data<float, Context>();
    auto* Ydata = Output(0)->template mutable_data<float16, Context>();
    kernel::Float2Half<float, Context>(Output(0)->count(), Xdata, Ydata);
}

#ifdef WITH_CUDA
DEPLOY_CUDA(FloatToHalf);
#endif
OPERATOR_SCHEMA(FloatToHalf).NumInputs(1).NumOutputs(1).Inplace({ { 0, 0 } });

NO_GRADIENT(FloatToHalf);

#endif

}    // namespace dragon