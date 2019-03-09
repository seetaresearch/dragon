#include "operators/control_flow/copy_op.h"

namespace dragon {

template <class Context> template <typename T>
void CopyOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    ctx()->template Copy<T, Context, Context>(
        Output(0)->count(), Ydata, Xdata);
}

template <class Context>
void CopyOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), bool)) RunWithType<bool>();
    else if (XIsType(Input(0), int8_t)) RunWithType<int8_t>();
    else if (XIsType(Input(0), uint8_t)) RunWithType<uint8_t>();
    else if (XIsType(Input(0), int)) RunWithType<int>();
    else if (XIsType(Input(0), int64_t)) RunWithType<int64_t>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), double)) RunWithType<double>();
    else LOG(FATAL) << DTypeHelper(Input(0), { 
        "bool", "int8", "uint8", "int32", "int64",
            "float16", "float32", "float64",
    });
}

DEPLOY_CPU(Copy);
#ifdef WITH_CUDA
DEPLOY_CUDA(Copy);
#endif
OPERATOR_SCHEMA(Copy).NumInputs(1).NumOutputs(1);
NO_GRADIENT(Copy);

}  // namespace dragon