#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
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
        const T* Bdata = nullptr; \
        auto* Ydata = Output(0)->template mutable_data<bool, Context>(); \
        if (Input(1).count() == 1) { \
            auto* WSdata = ws()->template caches<T, Context> \
                    ({ Input(0).count() })[0]; \
            auto* BCdata = Input(1).template data<T, CPUContext>(); \
            math::Set(Input(0).count(), BCdata[0], WSdata, ctx()); \
            Bdata = WSdata; \
        } else { Bdata = Input(1).template data<T, Context>(); } \
        kernel::Operation(Output(0)->count(), Adata, Bdata, Ydata, ctx()); \
    }

template <class Context>
void CompareOp<Context>::RunOnDevice() {
    if (Input(0).count() != Input(1).count()) {
        CHECK_EQ(Input(1).count(), 1)
            << "\nBoth A and B should have the same number of elements."
            << "\nOr the B should be a Scalar."
            << "\nDimensions of A and B are " << Input(0).DimString()
            << " and " << Input(1).DimString();
    }

    Output(0)->ReshapeLike(Input(0));

    if (operation == "EQ") { DEFINE_TYPED_CALLER(Equal); }
    else if (operation == "LT") { DEFINE_TYPED_CALLER(Less); }
    else if (operation == "GT") { DEFINE_TYPED_CALLER(Greater); }
    else if (operation == "LE") { DEFINE_TYPED_CALLER(LessEqual); }
    else if (operation == "GE") { DEFINE_TYPED_CALLER(GreaterEqual); }
    else { LOG(FATAL) << "Unsupport operation: " << operation << "."; }
    if (to_uint8) Output(0)->SetMeta(TypeMeta::Make<uint8_t>());
}

DEFINE_OP_CALLER(Equal);
DEFINE_OP_CALLER(Less);
DEFINE_OP_CALLER(LessEqual);
DEFINE_OP_CALLER(Greater);
DEFINE_OP_CALLER(GreaterEqual);

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