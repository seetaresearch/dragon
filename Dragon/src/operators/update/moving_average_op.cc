#include "operators/update/moving_average_op.h"
#include "utils/math_functions.h"

namespace dragon {

template <class Context> template <typename T>
void MovingAverageOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    math::Axpby<T, Context>(Input(0).count(),
        1.f - decay, Xdata, decay, Ydata);
}

template <class Context>
void MovingAverageOp<Context>::RunOnDevice() {
    CHECK(Input(0).dims() == Output(0)->dims())
        << "\nVariable(" << Output(0)->name() << ") and "
        << "new Value(" << Input(0).name() << ") "
        << "should have same dims.\nGot "
        << Output(0)->dim_string() << " and " << Input(0).dim_string();

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(MovingAverage);
#ifdef WITH_CUDA
DEPLOY_CUDA(MovingAverage);
#endif
OPERATOR_SCHEMA(MovingAverage).NumInputs(1).NumOutputs(1);

NO_GRADIENT(MovingAverage);

}   // namespace dragon