#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/ndarray/one_hot_op.h"

namespace dragon {

template <class Context> template <typename T>
void OneHotOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    math::Set<T, Context>(Output(0)->count(),
        dragon_cast<T, float>(float(off_value)), Ydata, ctx());

    kernel::OneHot<T, Context>(Input(0).count(),
        depth, on_value, Xdata, Ydata, ctx());
}

template <class Context>
void OneHotOp<Context>::RunOnDevice() {
    vector<TIndex> dims = Input(0).dims();
    dims.push_back(depth);
    Output(0)->Reshape(dims);
   
    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(OneHot);
#ifdef WITH_CUDA
DEPLOY_CUDA(OneHot);
#endif
OPERATOR_SCHEMA(OneHot).NumInputs(1).NumOutputs(1);

NO_GRADIENT(OneHot);

}    // namespace dragon