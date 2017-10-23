#include "operators/ndarray/one_hot_op.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void OneHotOp<Context>::RunWithType() {
    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    math::Set<T, Context>(output(0)->count(), 
                          dragon_cast<T, float>(float(off_value)), 
                          Ydata);

    kernel::OneHot<T, Context>(input(0).count(), depth, on_value, Xdata, Ydata);
}

template <class Context>
void OneHotOp<Context>::RunOnDevice() {
    vector<TIndex> dims = input(0).dims();
    dims.push_back(depth);
    output(0)->Reshape(dims);
   
    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(OneHot);
#ifdef WITH_CUDA
DEPLOY_CUDA(OneHot);
#endif
OPERATOR_SCHEMA(OneHot).NumInputs(1).NumOutputs(1);

NO_GRADIENT(OneHot);

}    // namespace dragon