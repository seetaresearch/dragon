#include "operators/control_flow/copy_op.h"

namespace dragon {

template <class Context> template <typename T>
void CopyOp<Context>::RunWithType() { 
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    ctx().template Copy<T, Context, Context>(Output(0)->count(), Ydata, Xdata);
}

template <class Context>
void CopyOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));
    if (Input(0).template IsType<float>()) RunWithType<float>();
    else if (Input(0).template IsType<float16>()) RunWithType<float16>();
    else if (Input(0).template IsType<double>()) RunWithType<double>();
    else if (Input(0).template IsType<int>()) RunWithType<int>();
    else if (Input(0).template IsType<int64_t>()) RunWithType<int64_t>();
    else if (Input(0).template IsType<uint8_t>()) RunWithType<uint8_t>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(Copy);
#ifdef WITH_CUDA
DEPLOY_CUDA(Copy);
#endif
OPERATOR_SCHEMA(Copy).NumInputs(1).NumOutputs(1);
NO_GRADIENT(Copy);

}    // namespace dragon