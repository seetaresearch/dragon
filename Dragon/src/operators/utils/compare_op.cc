#include "operators/utils/compare_op.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void CompareOp<Context>::EqualRunWithType() {
    auto* X1data = input(0).template data<T, Context>();
    auto* X2data = input(1).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    kernel::Equal<T, Context>(output(0)->count(), X1data, X2data, Ydata);
}

template <class Context>
void CompareOp<Context>::RunOnDevice() {
    CHECK_EQ(input(0).count(), input(1).count())
        << "both conditioned tensor must have same elements.";
    output(0)->ReshapeLike(input(0));

    if (operation == "EQUAL") {
        if (input(0).template IsType<float>()) EqualRunWithType<float>();
        else LOG(FATAL) << "unsupported input types.";
    }
    else {
        LOG(FATAL) << "unsupport operation: [" << operation << "].";
    }
}

DEPLOY_CPU(Compare);
#ifdef WITH_CUDA
DEPLOY_CUDA(Compare);
#endif
OPERATOR_SCHEMA(Compare).NumInputs(2).NumOutputs(1);

NO_GRADIENT(Compare);

}    // namespace dragon