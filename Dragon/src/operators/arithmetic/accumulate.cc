#include "core/workspace.h"
#include "utils/math_functions.h"
#include "operators/arithmetic/accumulate_op.h"

namespace dragon {

template <class Context> template <typename T>
void AccumulateOp<Context>::RunWithType(Tensor* X, Tensor* Y) {
    CHECK_EQ(X->count(), Y->count());
    auto* Xdata = X->template data<T, Context>();
    auto* Ydata = Y->template mutable_data<T, Context>();
    if (beta == 1.f) {
        if (alpha == 1.f) {
            math::Add(X->count(), Xdata, Ydata, Ydata, ctx());
        } else {
            math::Axpy(X->count(), alpha, Xdata, Ydata, ctx());
        }
    } else {
        if (alpha == 0.f) {
            math::Scale(X->count(), beta, Ydata, Ydata, ctx());
        } else {
            math::Axpby(X->count(), alpha, Xdata, beta, Ydata, ctx());
        }
    }
}

template <class Context>
void AccumulateOp<Context>::RunOnDevice() {
    for (int i = 0; i < InputSize(); i++) {
        Output(i)->ReshapeLike(Input(i));
        if (XIsType(Input(i), int8_t)) {
            RunWithType<int8_t>(&Input(i), Output(i));
        } else if (XIsType(Input(i), uint8_t)) {
            RunWithType<uint8_t>(&Input(i), Output(i));
        } else if (XIsType(Input(i), int)) {
            RunWithType<int>(&Input(i), Output(i));
        } else if (XIsType(Input(i), int64_t)) {
            RunWithType<int64_t>(&Input(i), Output(i));
        } else if (XIsType(Input(i), float16)) {
            RunWithType<float16>(&Input(i), Output(i));
        } else if (XIsType(Input(i), float)) {
            RunWithType<float>(&Input(i), Output(i));
        } else if (XIsType(Input(i), double)) {
            RunWithType<double>(&Input(i), Output(i));
        } else LOG(FATAL) << DTypeHelper(Input(i), {
            "int8", "uint8", "int32", "int64",
                "float16", "float32", "float64",
        });
    }
}

DEPLOY_CPU(Accumulate);
#ifdef WITH_CUDA
DEPLOY_CUDA(Accumulate);
#endif

OPERATOR_SCHEMA(Accumulate)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1, INT_MAX);
    
NO_GRADIENT(Accumulate);

}  // namespace dragon