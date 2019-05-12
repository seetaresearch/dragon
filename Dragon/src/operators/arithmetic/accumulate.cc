#include "core/workspace.h"
#include "utils/math_functions.h"
#include "operators/arithmetic/accumulate_op.h"

namespace dragon {

template <class Context> template <typename T>
void AccumulateOp<Context>::RunImpl(Tensor* X, Tensor* Y) {
    CHECK_EQ(X->count(), Y->count());
    auto* x = X->template data<T, Context>();
    auto* y = Y->template mutable_data<T, Context>();
    if (beta_ == 1.f) {
        if (alpha_ == 1.f) {
            math::Add(
                X->count(),
                x, y,
                y, ctx()
            );
        } else {
            math::Axpy(
                X->count(),
                alpha_, x,
                y, ctx()
            );
        }
    } else {
        if (alpha_ == 0.f) {
            math::Scale(
                X->count(),
                beta_, y,
                y, ctx()
            );
        } else {
            math::Axpby(
                X->count(),
                alpha_, x,
                beta_, y, ctx()
            );
        }
    }
}

template <class Context>
void AccumulateOp<Context>::RunOnDevice() {
    for (int i = 0; i < XSize(); i++) {
        Y(i)->ReshapeLike(X(i));
        if (XIsType(X(i), int8_t)) {
            RunImpl<int8_t>(&X(i), Y(i));
        } else if (XIsType(X(i), uint8_t)) {
            RunImpl<uint8_t>(&X(i), Y(i));
        } else if (XIsType(X(i), int)) {
            RunImpl<int>(&X(i), Y(i));
        } else if (XIsType(X(i), int64_t)) {
            RunImpl<int64_t>(&X(i), Y(i));
        } else if (XIsType(X(i), float16)) {
            RunImpl<float16>(&X(i), Y(i));
        } else if (XIsType(X(i), float)) {
            RunImpl<float>(&X(i), Y(i));
        } else if (XIsType(X(i), double)) {
            RunImpl<double>(&X(i), Y(i));
        } else LOG(FATAL) << DTypeString(X(i), {
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
     /* X(0), ... */
    .NumInputs(1, INT_MAX)
     /* Y(0), ... */
    .NumOutputs(1, INT_MAX);
    
NO_GRADIENT(Accumulate);

}  // namespace dragon