#include "operators/utils/copy_op.h"

namespace dragon {

template <class Context> template <typename T>
void CopyOp<Context>::RunWithType() { 
    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    ctx().template Copy<T, Context, Context>(output(0)->count(), Ydata, Xdata);
}

template <class Context>
void CopyOp<Context>::RunOnDevice() {
    output(0)->ReshapeLike(input(0));
    if (input(0).template IsType<float>()) RunWithType<float>();
    else if (input(0).template IsType<float16>()) RunWithType<float16>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(Copy);
#ifdef WITH_CUDA
DEPLOY_CUDA(Copy);
#endif
OPERATOR_SCHEMA(Copy).NumInputs(1).NumOutputs(1);
NO_GRADIENT(Copy);

}    // namespace dragon