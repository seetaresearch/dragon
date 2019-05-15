#include "utils/math_functions.h"
#include "operators/control_flow/copy_op.h"

namespace dragon {

template <class Context> template <typename T>
void CopyOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();
    math::Copy(Y(0)->count(), x, y, ctx());
}

template <class Context>
void CopyOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <bool, int8_t, uint8_t, int, int64_t,
               float16, float, double>
    >::Call(this, X(0));
}

DEPLOY_CPU(Copy);
#ifdef WITH_CUDA
DEPLOY_CUDA(Copy);
#endif

OPERATOR_SCHEMA(Copy)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

NO_GRADIENT(Copy);

}  // namespace dragon