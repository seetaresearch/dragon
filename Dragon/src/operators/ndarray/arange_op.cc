#include "operators/ndarray/arange_op.h"
#include "core/workspace.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context>
void ArangeOp<Context>::Reshape() {
    //  parse start & step & stop
    Tensor* t = ws()->GetTensor(start_desc);
    CHECK_EQ(t->count(), 1) << "\nThe start should be a scalar";
    CHECK(t->IsType<int>()) << "\nThe type of start should be int32.";
    start = t->template data<int, CPUContext>()[0];

    t = ws()->GetTensor(step_desc);
    CHECK_EQ(t->count(), 1) << "\nThe step should be a scalar";
    CHECK(t->IsType<int>()) << "\nThe type of step should be int32.";
    step = t->template data<int, CPUContext>()[0];

    if (!stop_desc.empty()) {
        t = ws()->GetTensor(stop_desc);
        CHECK_EQ(t->count(), 1) << "\nThe stop should be a scalar";
        CHECK(t->IsType<int>()) << "\nThe type of stop should be int32.";
        stop = t->template data<int, CPUContext>()[0];
    } else { stop = start; start = 0; }

    count = (stop - start - 1) / step + 1;
    output(0)->Reshape(vector<TIndex>(1, count));
}

template <class Context> template <typename T>
void ArangeOp<Context>::RunWithType() {
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    kernel::Arange<T, Context>(count, start, step, Ydata);
}

template <class Context>
void ArangeOp<Context>::RunOnDevice() {
    Reshape();
    if (dtype == "FLOAT32") RunWithType<float>(); 
    else if (dtype == "INT32") RunWithType<int>();
    else LOG(FATAL) << "Unsupported data types";
}

DEPLOY_CPU(Arange);
#ifdef WITH_CUDA
DEPLOY_CUDA(Arange);
#endif
OPERATOR_SCHEMA(Arange).NumInputs(0).NumOutputs(1);

NO_GRADIENT(Arange);

}    // namespace dragon