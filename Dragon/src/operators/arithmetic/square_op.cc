#include "utils/math_functions.h"
#include "operators/arithmetic/square_op.h"

namespace dragon {

template <class Context> template <typename T>
void SquareOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    math::Square(Output(0)->count(), Xdata, Ydata, ctx());
}

template <class Context>
void SquareOp<Context>::RunOnDevice() {
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

DEPLOY_CPU(Square);
#ifdef WITH_CUDA
DEPLOY_CUDA(Square);
#endif
OPERATOR_SCHEMA(Square).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void SquareGradientOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    math::Mul(Output(0)->count(), dYdata, Xdata, dXdata, ctx());
    math::Scale(Output(0)->count(), 2.f, dXdata, dXdata, ctx());
}

template <class Context>
void SquareGradientOp<Context>::RunOnDevice() {
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

DEPLOY_CPU(SquareGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SquareGradient);
#endif

OPERATOR_SCHEMA(SquareGradient)
    .NumInputs(2).NumOutputs(1);

REGISTER_GRADIENT(Square, SimpleGradientMaker);

}  // namespace dragon