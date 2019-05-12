#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/misc/cast_op.h"

namespace dragon {

#define ELIGIBLE_DTYPES \
    { "bool", "int8", "uint8", "int32", "int64", \
          "float16", "float32", "float64" }

#define DEFINE_TYPE_A_TO_B(Ta, type_str, Tb) \
    if (dtype() == type_str) { \
        if (XSize() != 0) { \
            Y(0)->ReshapeLike(X(0)); \
            auto* x = X(0).template data<Ta, Context>(); \
            auto* y = Y(0)->template mutable_data<Tb, Context>(); \
            kernel::TypeA2B(X(0).count(), x, y, ctx()); \
        } else { \
            auto n = Y(0)->count(); \
            auto* x = Y(0)->template data<Ta, Context>(); \
            auto* scratch = ws()->template data<Tb, Context>({ n })[0]; \
            kernel::TypeA2B(n, x, scratch, ctx()); \
            ctx()->FinishDeviceCompution(); \
            auto* y = Y(0)->template mutable_data<Tb, Context>(); \
            math::Copy(n, scratch, y, ctx()); \
        } \
        return; \
    }

#define DEFINE_TYPE_A_TO_ALL(Ta) \
    DEFINE_TYPE_A_TO_B(Ta, "bool", bool); \
    DEFINE_TYPE_A_TO_B(Ta, "int8", int8_t); \
    DEFINE_TYPE_A_TO_B(Ta, "uint8", uint8_t); \
    DEFINE_TYPE_A_TO_B(Ta, "int32", int); \
    DEFINE_TYPE_A_TO_B(Ta, "int64", int64_t); \
    DEFINE_TYPE_A_TO_B(Ta, "float16", float16); \
    DEFINE_TYPE_A_TO_B(Ta, "float32", float); \
    DEFINE_TYPE_A_TO_B(Ta, "float64", double)

#define DEFINE_TYPED_IMPL(X) \
    if (XIsType(X, bool)) { DEFINE_TYPE_A_TO_ALL(bool); } \
    else if (XIsType(X, int8_t)) { DEFINE_TYPE_A_TO_ALL(int8_t); } \
    else if (XIsType(X, uint8_t)) { DEFINE_TYPE_A_TO_ALL(uint8_t); } \
    else if (XIsType(X, int)) { DEFINE_TYPE_A_TO_ALL(int); } \
    else if (XIsType(X, int64_t)) { DEFINE_TYPE_A_TO_ALL(int64_t); } \
    else if (XIsType(X, float16)) { DEFINE_TYPE_A_TO_ALL(float16); } \
    else if (XIsType(X, float)) { DEFINE_TYPE_A_TO_ALL(float); } \
    else if (XIsType(X, double)) { DEFINE_TYPE_A_TO_ALL(double); } \
    else LOG(FATAL) << DTypeString(X, ELIGIBLE_DTYPES)

template <class Context>
void CastOp<Context>::RunOnDevice() {
    if (inplace_ && XSize() > 0) {
        LOG(FATAL) << "Excepted 0 inputs, got "
                   << XSize() << ".";
    }

    if (XSize() > 0) DEFINE_TYPED_IMPL(X(0));
    else DEFINE_TYPED_IMPL((*Y(0)));
}

template <class Context>
void CastGradientOp<Context>::RunOnDevice() {
    this->dtype_ = TypeMetaToString(X(1).meta());
    DEFINE_TYPED_IMPL(X(0));
}

DEPLOY_CPU(Cast);
#ifdef WITH_CUDA
DEPLOY_CUDA(Cast);
#endif

DEPLOY_CPU(CastGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(CastGradient);
#endif

OPERATOR_SCHEMA(Cast)
     /* X */
    .NumInputs(0, 1)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(CastGradient)
     /* dY, X */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GradientMaker);
    vector<OperatorDef> MakeDef() override {
        return SingleDef(def.type() + "Gradient", "",
            // Inversed inputs to reuse the macros
            vector<string>({ GO(0), I(0) }),
            vector<string>({ GI(0) })
        );
    }
};

}  // namespace

REGISTER_GRADIENT(Cast, GradientMaker);

#undef ELIGIBLE_DTYPES
#undef DEFINE_TYPE_A_TO_B
#undef DEFINE_TYPE_A_TO_ALL
#undef DEFINE_TYPED_IMPL

}  // namespace dragon