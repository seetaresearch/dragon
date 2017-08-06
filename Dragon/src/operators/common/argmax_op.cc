#include "operators/common/argmax_op.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void ArgmaxOp<Context>::RunWithType() {
    if (top_k != 1) {
        //  it's difficult to implement device code when top_k > 1
        auto* Xdata = input(0).template data<T, CPUContext>();
        auto* Ydata = output(0)->template mutable_data<T, CPUContext>();
        kernel::Argmax<T, CPUContext>(count, input(0).dim(axis), inner_dim,
            top_k, Xdata, Ydata);
    } else {
        auto* Xdata = input(0).template data<T, Context>();
        auto* Ydata = output(0)->template mutable_data<T, Context>();
        kernel::Argmax<T, Context>(count, input(0).dim(axis), inner_dim,
            top_k, Xdata, Ydata);
    }
}

template <class Context>
void ArgmaxOp<Context>::RunOnDevice() {
    count = input(0).count() / input(0).dim(axis);
    inner_dim = input(0).count(axis) / input(0).dim(axis);
    vector<TIndex> dims = input(0).dims();
    if (top_k == 1) dims.erase(dims.begin() + axis);
    else dims[axis] = top_k;
    output(0)->Reshape(dims);

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(Argmax);
#ifdef WITH_CUDA
DEPLOY_CUDA(Argmax);
#endif
OPERATOR_SCHEMA(Argmax).NumInputs(1).NumOutputs(1);

NO_GRADIENT(Argmax);

}    // namespace dragon