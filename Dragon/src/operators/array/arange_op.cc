#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "operators/array/arange_op.h"

namespace dragon {

template <class Context> template <typename T>
void ArangeOp<Context>::RunImpl() {
    auto* y = Y(0)->template mutable_data<T, Context>();
    kernel::Arange(dim_, astart_, astep_, y, ctx());
}

template <class Context>
void ArangeOp<Context>::RunOnDevice() {
    astart_ = start(), astop_ = stop(), astep_ = step();
    if (astop_ == 0) { astop_ = astart_; astart_ = 0; }
    dim_ = (astop_ - astart_ - 1) / astep_ + 1;

    CHECK_GT(dim_, 0)
        << "\nInvalid arguments: \n"
        << "start = " << start() << ", "
        << "stop = " << stop() << ", "
        << "step = " << step() << ".";

    Y(0)->Reshape({ dim_ });

    if (dtype() == "int8") {
        RunImpl<int8_t>();
    } else if (dtype() == "uint8") {
        RunImpl<uint8_t>();
    } else if (dtype() == "int32") {
        RunImpl<int>();
    } else if (dtype() == "int64") {
        RunImpl<int64_t>();
    } else if (dtype() == "float16") {
        RunImpl<float16>();
    } else if (dtype() == "float32") {
        RunImpl<float>();
    } else if (dtype() == "float64") {
        RunImpl<double>();
    } else {
        LOG(FATAL) << DTypeString(dtype(), {
            "int8", "uint8", "int32", "int64",
            "float16", "float32", "float64",
        });
    }
}

DEPLOY_CPU(Arange);
#ifdef WITH_CUDA
DEPLOY_CUDA(Arange);
#endif

OPERATOR_SCHEMA(Arange)
    .NumInputs(0).NumOutputs(1);

NO_GRADIENT(Arange);

}  // namespace dragon