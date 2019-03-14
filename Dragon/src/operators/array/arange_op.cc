#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "operators/array/arange_op.h"

namespace dragon {

template <class Context> template <typename T>
void ArangeOp<Context>::RunWithType() {
    astart = start(), astop = stop(), astep = step();
    if (astop == 0) { astop = astart; astart = 0; }
    dim = (astop - astart - 1) / astep + 1;
    CHECK_GT(dim, 0) << "\nInvalid arguments: \n"
                     << "start = " << start() << ", "
                     << "stop = " << stop() << ", "
                     << "step = " << step() << ".";
    Output(0)->Reshape({ dim });
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    kernel::Arange(dim, astart, astep, Ydata, ctx());
}

template <class Context>
void ArangeOp<Context>::RunOnDevice() {
    if (dtype == "int8") RunWithType<int8_t>();
    else if (dtype == "uint8") RunWithType<uint8_t>();
    else if (dtype == "int32") RunWithType<int>();
    else if (dtype == "int64") RunWithType<int64_t>();
    else if (dtype == "float16") RunWithType<float16>();
    else if (dtype == "float32") RunWithType<float>();
    else if (dtype == "float64") RunWithType<double>();
    else LOG(FATAL) << DTypeHelper(Input(0), { 
        "int8", "uint8", "int32", "int64",
            "float16", "float32", "float64",
    });
}

DEPLOY_CPU(Arange);
#ifdef WITH_CUDA
DEPLOY_CUDA(Arange);
#endif
OPERATOR_SCHEMA(Arange).NumInputs(0).NumOutputs(1);

NO_GRADIENT(Arange);

}  // namespace dragon