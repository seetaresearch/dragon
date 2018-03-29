#include "operators/control_flow/compare_op.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void CompareOp<Context>::EqualRunWithType() {
    auto* X1data = Input(0).template data<T, Context>();
    auto* X2data = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    kernel::Equal<T, Context>(Output(0)->count(), X1data, X2data, Ydata);
}

template <class Context>
void CompareOp<Context>::RunOnDevice() {
    CHECK_EQ(Input(0).count(), Input(1).count())
        << "Both conditioned tensors should have same elements.";
    Output(0)->ReshapeLike(Input(0));

    if (operation == "EQUAL") {
        if (Input(0).template IsType<float>()) EqualRunWithType<float>();
        else LOG(FATAL) << "Unsupported input types.";
    }
    else {
        LOG(FATAL) << "Unsupport operation: [" << operation << "].";
    }
}

DEPLOY_CPU(Compare);
#ifdef WITH_CUDA
DEPLOY_CUDA(Compare);
#endif
OPERATOR_SCHEMA(Compare).NumInputs(2).NumOutputs(1);

NO_GRADIENT(Compare);

}    // namespace dragon