#include "operators/ndarray/arange_op.h"
#include "core/workspace.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void ArangeOp<Context>::RunWithType() {
    TIndex start_ = start(), step_ = step(), stop_ = stop(), count;
    if (stop_ == 0) { stop_ = start_; start_ = 0; }
    count = (stop_ - start_ - 1) / step_ + 1;
    Output(0)->Reshape({ count });
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    kernel::Arange<T, Context>(count, start_, step_, Ydata);
}

template <class Context>
void ArangeOp<Context>::RunOnDevice() {
    if (dtype == "FLOAT32") RunWithType<float>(); 
    else if (dtype == "INT32") RunWithType<int>();
    else LOG(FATAL) << "Unsupported DType: " << dtype;
}

DEPLOY_CPU(Arange);
#ifdef WITH_CUDA
DEPLOY_CUDA(Arange);
#endif
OPERATOR_SCHEMA(Arange).NumInputs(0).NumOutputs(1);

NO_GRADIENT(Arange);

}    // namespace dragon