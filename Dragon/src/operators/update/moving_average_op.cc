#include "utils/math_functions.h"
#include "operators/update/moving_average_op.h"

namespace dragon {

template <class Context> template <typename T>
void MovingAverageOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    math::Axpby(Input(0).count(),
        1.f - decay, Xdata, decay, Ydata, ctx());
}

template <class Context>
void MovingAverageOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), int8_t)) RunWithType<int8_t>();
    else if (XIsType(Input(0), uint8_t)) RunWithType<uint8_t>();
    else if (XIsType(Input(0), int)) RunWithType<int>();
    else if (XIsType(Input(0), int64_t)) RunWithType<int64_t>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), double)) RunWithType<double>();
    else LOG(FATAL) << DTypeHelper(Input(0), {
        "int8", "uint8", "int32", "int64",
            "float16", "float32", "float64",
    });
}

DEPLOY_CPU(MovingAverage);
#ifdef WITH_CUDA
DEPLOY_CUDA(MovingAverage);
#endif
OPERATOR_SCHEMA(MovingAverage).NumInputs(1).NumOutputs(1);

NO_GRADIENT(MovingAverage);

}   // namespace dragon