#include "utils/op_kernel.h"
#include "operators/control_flow/compare_op.h"

namespace dragon {

using kernel::Equal;
using kernel::Less;
using kernel::Greater;

#define ELIGIBLE_DATA_TYPES \
    { "bool", "int8", "uint8", "int32", "int64", \
      "float16", "float32", "float64" }

#define DEFINE_TYPED_CALLER(Operation) \
    if (XIsType(Input(0), bool)) Operation##RunWithType<bool>(); \
    else if (XIsType(Input(0), int8_t)) Operation##RunWithType<int8_t>(); \
    else if (XIsType(Input(0), uint8_t)) Operation##RunWithType<uint8_t>(); \
    else if (XIsType(Input(0), int)) Operation##RunWithType<int>(); \
    else if (XIsType(Input(0), int64_t)) Operation##RunWithType<int64_t>(); \
    else if (XIsType(Input(0), float16)) Operation##RunWithType<float16>(); \
    else if (XIsType(Input(0), float)) Operation##RunWithType<float>(); \
    else if (XIsType(Input(0), double)) Operation##RunWithType<double>(); \
    else LOG(FATAL) << DTypeHelper(Input(0), ELIGIBLE_DATA_TYPES)

#define DEFINE_OP_CALLER(Operation) \
    template <class Context> template <typename T> \
    void CompareOp<Context>::Operation##RunWithType() { \
        auto* Adata = Input(0).template data<T, Context>(); \
        auto* Bdata = Input(1).template data<T, Context>(); \
        auto* Ydata = Output(0)->template mutable_data<bool, Context>(); \
        kernel::Operation(Output(0)->count(), Adata, Bdata, Ydata, ctx()); \
    }

template <class Context>
void CompareOp<Context>::RunOnDevice() {
    CHECK_EQ(Input(0).count(), Input(1).count())
        << "\nBoth A and B should have the same number of elements."
        << "\nDimensions of A and B are " << Input(0).DimString()
        << " and " << Input(1).DimString();

    Output(0)->ReshapeLike(Input(0));

    if (operation == "EQUAL") { DEFINE_TYPED_CALLER(Equal); }
    else if (operation == "LESS") { DEFINE_TYPED_CALLER(Less); }
    else if (operation == "GREATER") { DEFINE_TYPED_CALLER(Greater); }
    else { LOG(FATAL) << "Unsupport operation: " << operation << "."; }

    if (to_uint8) Output(0)->SetMeta(TypeMeta::Make<uint8_t>());
}

DEFINE_OP_CALLER(Equal);
DEFINE_OP_CALLER(Less);
DEFINE_OP_CALLER(Greater);

DEPLOY_CPU(Compare);
#ifdef WITH_CUDA
DEPLOY_CUDA(Compare);
#endif
OPERATOR_SCHEMA(Compare).NumInputs(2).NumOutputs(1);

NO_GRADIENT(Compare);

#undef ELIGIBLE_DATA_TYPES
#undef DEFINE_OP_CALLLER
#undef DEFINE_TYPED_CALLER

}  // namespace dragon