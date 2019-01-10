#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "operators/misc/astype_op.h"

namespace dragon {

#define ELIGIBLE_DATA_TYPES \
    { "bool", "int8", "uint8", "int32", "int64", \
      "float16", "float32", "float64" }

#define DEFINE_TYPE_A_TO_B(type_a, type_bn, type_b) \
    if (dtype == type_bn) { \
        if (InputSize() != 0) { \
            Output(0)->ReshapeLike(Input(0)); \
            auto* Xdata = Input(0).template data<type_a, Context>(); \
            auto* Ydata = Output(0)->template mutable_data<type_b, Context>(); \
            kernel::TypeA2B(Input(0).count(), Xdata, Ydata, ctx()); \
        } else { \
            int64_t count = Output(0)->count(); \
            auto* Xdata = Output(0)->template data<type_a, Context>(); \
            auto* Cdata = ws()->template caches<type_b, Context>({ count })[0]; \
            kernel::TypeA2B(count, Xdata, Cdata, ctx()); \
            ctx()->FinishDeviceCompution(); \
            auto* Ydata = Output(0)->template mutable_data<type_b, Context>(); \
            ctx()->template Copy<type_b, Context, Context>(count, Ydata, Cdata); \
        } \
        return; \
    }

#define DEFINE_TYPE_A_TO_ALL(type_a) \
    DEFINE_TYPE_A_TO_B(type_a, "bool", bool); \
    DEFINE_TYPE_A_TO_B(type_a, "int8", int8_t); \
    DEFINE_TYPE_A_TO_B(type_a, "uint8", uint8_t); \
    DEFINE_TYPE_A_TO_B(type_a, "int32", int); \
    DEFINE_TYPE_A_TO_B(type_a, "int64", int64_t); \
    DEFINE_TYPE_A_TO_B(type_a, "float16", float16); \
    DEFINE_TYPE_A_TO_B(type_a, "float32", float); \
    DEFINE_TYPE_A_TO_B(type_a, "float64", double)

#define DEFINE_TYPED_CALLER(X) \
    if (XIsType(X, bool)) { DEFINE_TYPE_A_TO_ALL(bool); } \
    else if (XIsType(X, int8_t)) { DEFINE_TYPE_A_TO_ALL(int8_t); } \
    else if (XIsType(X, uint8_t)) { DEFINE_TYPE_A_TO_ALL(uint8_t); } \
    else if (XIsType(X, int)) { DEFINE_TYPE_A_TO_ALL(int); } \
    else if (XIsType(X, int64_t)) { DEFINE_TYPE_A_TO_ALL(int64_t); } \
    else if (XIsType(X, float16)) { DEFINE_TYPE_A_TO_ALL(float16); } \
    else if (XIsType(X, float)) { DEFINE_TYPE_A_TO_ALL(float); } \
    else if (XIsType(X, double)) { DEFINE_TYPE_A_TO_ALL(double); } \
    else LOG(FATAL) << DTypeHelper(X, ELIGIBLE_DATA_TYPES)

template <class Context>
void AsTypeOp<Context>::RunOnDevice() {
    if (inplace && InputSize() != 0)
        LOG(FATAL) << "Excepted 0 inputs, got " << InputSize() << ".";

    if (InputSize() != 0) { DEFINE_TYPED_CALLER(Input(0)); } 
    else { DEFINE_TYPED_CALLER((*Output(0))); }
}

DEPLOY_CPU(AsType);
#ifdef WITH_CUDA
DEPLOY_CUDA(AsType);
#endif
OPERATOR_SCHEMA(AsType).NumInputs(0, 1).NumOutputs(1);

NO_GRADIENT(AsType);

#undef ELIGIBLE_DATA_TYPES
#undef DEFINE_TYPE_A_TO_B
#undef DEFINE_TYPE_A_TO_ALL
#undef DEFINE_TYPED_CALLER

}  // namespace dragon