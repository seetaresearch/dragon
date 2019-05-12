#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/array/one_hot_op.h"

namespace dragon {

template <class Context> template <typename T>
void OneHotOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    math::Set(
        Y(0)->count(),
        cast::to<T>((float)off_value_),
        y, ctx()
    );

    kernel::OneHot(
        X(0).count(),
        depth_, on_value_,
        x, y, ctx()
    );
}

template <class Context>
void OneHotOp<Context>::RunOnDevice() {
    auto out_shape = X(0).dims();
    out_shape.push_back(depth_);

    Y(0)->Reshape(out_shape);
   
    if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else if (XIsType(X(0), int)) {
        RunImpl<int>();
    } else if (XIsType(X(0), int64_t)) {
        RunImpl<int64_t>();
    } else {
        LOG(FATAL) << DTypeString(X(0),
            { "float32", "int32", "int64" }
        );
    }
}

DEPLOY_CPU(OneHot);
#ifdef WITH_CUDA
DEPLOY_CUDA(OneHot);
#endif

OPERATOR_SCHEMA(OneHot)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

NO_GRADIENT(OneHot);

}  // namespace dragon