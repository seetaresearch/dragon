#include "operators/ndarray/arange_op.h"
#include "core/workspace.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context>
void ArangeOp<Context>::Reshape() {
    if (!dynamic_start_.empty()) {
        dynamic_start = ws()->GetTensor(dynamic_start_);
        CHECK_EQ(dynamic_start->count(), 1)
            << "The start should be a scalar";
        if (dynamic_start->IsType<int>()) {
            start = dynamic_start->template data<int, CPUContext>()[0];
        } else if (dynamic_start->IsType<float>()) {
            start = dynamic_start->template data<float, CPUContext>()[0];
        } else {
            LOG(FATAL) << "Unsupported types of start.";
        }
    }
    if (!dynamic_stop_.empty()) {
        dynamic_stop = ws()->GetTensor(dynamic_stop_);
        CHECK_EQ(dynamic_stop->count(), 1)
            << "The stop should be a scalar";
        if (dynamic_stop->IsType<int>()) {
            stop = dynamic_stop->template data<int, CPUContext>()[0];
        } else if (dynamic_stop->IsType<float>()) {
            stop = dynamic_stop->template data<float, CPUContext>()[0];
        } else {
            LOG(FATAL) << "Unsupported types of stop.";
        }
    }
    if (!dynamic_step_.empty()) {
        dynamic_step = ws()->GetTensor(dynamic_step_);
        CHECK_EQ(dynamic_step->count(), 1)
            << "The step should be a scalar";
        if (dynamic_step->IsType<int>()) {
            step = dynamic_step->template data<int, CPUContext>()[0];
        } else if (dynamic_step->IsType<float>()) {
            step = dynamic_step->template data<float, CPUContext>()[0];
        } else {
            LOG(FATAL) << "Unsupported types of step.";
        }
    }
    if (stop == -1) { stop = start; start = 0; }
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