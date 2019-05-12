#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/control_flow/compare_op.h"

namespace dragon {

using kernel::Equal;
using kernel::Less;
using kernel::Greater;

#define ELIGIBLE_DTYPES \
    { "bool", "int8", "uint8", "int32", "int64", \
           "float16", "float32", "float64" }

#define DEFINE_TYPED_IMPL(Op) \
    if (XIsType(X(0), bool)) Op##RunImpl<bool>(); \
    else if (XIsType(X(0), int8_t)) Op##RunImpl<int8_t>(); \
    else if (XIsType(X(0), uint8_t)) Op##RunImpl<uint8_t>(); \
    else if (XIsType(X(0), int)) Op##RunImpl<int>(); \
    else if (XIsType(X(0), int64_t)) Op##RunImpl<int64_t>(); \
    else if (XIsType(X(0), float16)) Op##RunImpl<float16>(); \
    else if (XIsType(X(0), float)) Op##RunImpl<float>(); \
    else if (XIsType(X(0), double)) Op##RunImpl<double>(); \
    else LOG(FATAL) << DTypeString(X(0), ELIGIBLE_DTYPES)

#define DEFINE_OP_IMPL(Op) \
    template <class Context> template <typename T> \
    void CompareOp<Context>::Op##RunImpl() { \
        auto* a = X(0).template data<T, Context>(); \
        const T* b = nullptr; \
        auto* y = Y(0)->template mutable_data<bool, Context>(); \
        if (X(1).count() == 1) { \
            auto* scratch = ws() \
                ->template data<T, Context> \
                    ({ X(0).count() })[0]; \
            auto* bc = X(1).template data<T, CPUContext>(); \
            math::Set(X(0).count(), bc[0], scratch, ctx()); \
            b = scratch; \
        } else { b = X(1).template data<T, Context>(); } \
        kernel::Op(Y(0)->count(), a, b, y, ctx()); \
    }

template <class Context>
void CompareOp<Context>::RunOnDevice() {
    if (X(0).count() != X(1).count()) {
        CHECK_EQ(X(1).count(), 1)
            << "\nBoth A and B should have the same num of elements."
            << "\nOr the B should be a Scalar."
            << "\nDimensions of A and B are "
            << X(0).DimString() << " and " << X(1).DimString();
    }

    Y(0)->ReshapeLike(X(0));

    if (op_str_ == "EQ") { DEFINE_TYPED_IMPL(Equal); }
    else if (op_str_ == "LT") { DEFINE_TYPED_IMPL(Less); }
    else if (op_str_ == "GT") { DEFINE_TYPED_IMPL(Greater); }
    else if (op_str_ == "LE") { DEFINE_TYPED_IMPL(LessEqual); }
    else if (op_str_ == "GE") { DEFINE_TYPED_IMPL(GreaterEqual); }
    else { LOG(FATAL) << "Unknown Operation: " << op_str_ << "."; }

    if (to_uint8_) Y(0)->SetMeta(TypeMeta::Make<uint8_t>());
}

DEFINE_OP_IMPL(Equal);
DEFINE_OP_IMPL(Less);
DEFINE_OP_IMPL(LessEqual);
DEFINE_OP_IMPL(Greater);
DEFINE_OP_IMPL(GreaterEqual);

DEPLOY_CPU(Compare);
#ifdef WITH_CUDA
DEPLOY_CUDA(Compare);
#endif

OPERATOR_SCHEMA(Compare)
     /* A, B */
    .NumInputs(2)
     /* Y */
    .NumOutputs(1);

NO_GRADIENT(Compare);

#undef ELIGIBLE_DTYPES
#undef DEFINE_OP_IMPL
#undef DEFINE_TYPED_IMPL

}  // namespace dragon