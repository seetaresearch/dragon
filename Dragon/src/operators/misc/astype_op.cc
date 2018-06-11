#include "operators/misc/astype_op.h"
#include "core/workspace.h"
#include "utils/op_kernel.h"

namespace dragon {

#define ELIGIBLE_DTYPES \
    { "float16", "float32", "float64", \
      "int32", "int64", "uint8" }

#define BODY(type_a, type_bn, type_b) \
    if (dtype == type_bn) { \
        if (InputSize() != 0) { \
            Output(0)->ReshapeLike(Input(0)); \
            auto* Xdata = Input(0).template data<type_a, Context>(); \
            auto* Ydata = Output(0)->template mutable_data<type_b, Context>(); \
            kernel::TypeA2B<type_a, type_b, Context>(Input(0).count(), Xdata, Ydata); \
        } else { \
            Tensor* buffer = ws()->GetBuffer(); \
            buffer->ReshapeLike(*Output(0)); \
            auto* Xdata = Output(0)->template data<type_a, Context>(); \
            auto* Ydata = buffer->template mutable_data<type_b, Context>(); \
            kernel::TypeA2B<type_a, type_b, Context>(Output(0)->count(), Xdata, Ydata); \
            Output(0)->template Copy<Context, Context>(*buffer); \
            ws()->ReleaseBuffer(buffer); \
        } \
        return; \
    }

#define BODY_A2ALL(type_a) \
    BODY(type_a, "float16", float16); \
    BODY(type_a, "float32", float); \
    BODY(type_a, "float64", double); \
    BODY(type_a, "int32", int); \
    BODY(type_a, "int64", int64_t); \
    BODY(type_a, "uint8", uint8_t); \
    LOG(FATAL) << DTypeHelper(dtype, ELIGIBLE_DTYPES);

template <class Context>
void AsTypeOp<Context>::RunOnDevice() {
    if (inplace && InputSize() != 0)
        LOG(FATAL) << "Excepted 0 inputs, got " << InputSize() << ".";
    if (InputSize() != 0) {
        if (XIsType(Input(0), float16)) { BODY_A2ALL(float16); }
        else if (XIsType(Input(0), float)) { BODY_A2ALL(float); }
        else if (XIsType(Input(0), double)) { BODY_A2ALL(double); }
        else if (XIsType(Input(0), int)) { BODY_A2ALL(int); }
        else if (XIsType(Input(0), int64_t)) { BODY_A2ALL(int64_t); }
        else if (XIsType(Input(0), uint8_t)) { BODY_A2ALL(uint8_t); }
        else LOG(FATAL) << DTypeHelper(Input(0), ELIGIBLE_DTYPES);
    } else {
        if (XIsType((*Output(0)), float16)) { BODY_A2ALL(float16); }
        else if (XIsType((*Output(0)), float)) { BODY_A2ALL(float); }
        else if (XIsType((*Output(0)), double)) { BODY_A2ALL(double); }
        else if (XIsType((*Output(0)), int)) { BODY_A2ALL(int); }
        else if (XIsType((*Output(0)), int64_t)) { BODY_A2ALL(int64_t); }
        else if (XIsType((*Output(0)), uint8_t)) { BODY_A2ALL(uint8_t); }
        else LOG(FATAL) << DTypeHelper((*Output(0)), ELIGIBLE_DTYPES);
    }
}

DEPLOY_CPU(AsType);
#ifdef WITH_CUDA
DEPLOY_CUDA(AsType);
#endif
OPERATOR_SCHEMA(AsType).NumInputs(0, 1).NumOutputs(1);

NO_GRADIENT(AsType);

}    // namespace dragon