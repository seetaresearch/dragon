#include "utils/math_functions.h"
#include "operators/arithmetic/square_op.h"

namespace dragon {

template <class Context> template <typename T>
void SquareOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();
    math::Square(X(0).count(), x, y, ctx());
}

template <class Context>
void SquareOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <int8_t, uint8_t, int, int64_t,
            float16, float, double>
    >::Call(this, X(0));
}

template <class Context> template <typename T>
void SquareGradientOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* dy = X(-1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();
    math::Mul(X(0).count(), dy, x, dx, ctx());
    math::Scale(X(0).count(), 2.f, dx, dx, ctx());
}

template <class Context>
void SquareGradientOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <int8_t, uint8_t, int, int64_t,
            float16, float, double>
    >::Call(this, X(0));
}

DEPLOY_CPU(Square);
#ifdef WITH_CUDA
DEPLOY_CUDA(Square);
#endif

DEPLOY_CPU(SquareGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SquareGradient);
#endif

OPERATOR_SCHEMA(Square)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(SquareGradient)
     /* X, dY */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(Square, SimpleGradientMaker);

}  // namespace dragon