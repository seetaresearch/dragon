#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/arithmetic/fundamental_op.h"

namespace dragon {

template <class Context> template <typename T>
void RMulOp<Context>::EltwiseRunImpl() {
    auto* a = X(0).template data<T, Context>();
    auto* b = X(1).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();
    math::Mul(Y(0)->count(), a, b, y, ctx());
}

template <class Context> template <typename T>
void RMulOp<Context>::BroadcastRunImpl(int type) {
    auto* a = X(0).template data<T, Context>();
    auto* b = X(1).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();
    math::BroadcastMul(rows_, cols_, type, a, b, y, ctx());
}

template <class Context>
void RMulOp<Context>::RunOnDevice() {
    DECLARE_INPUT_DESC;
    Y(0)->ReshapeLike(X(1));

    if (XIsType(X(0), int8_t)) {
        DEFINE_RFUNDAMENTAL_TYPED_IMPL(int8_t);
    } else if (XIsType(X(0), uint8_t)) {
        DEFINE_RFUNDAMENTAL_TYPED_IMPL(uint8_t);
    } else if (XIsType(X(0), int)) {
        DEFINE_RFUNDAMENTAL_TYPED_IMPL(int);
    } else if (XIsType(X(0), int64_t)) {
        DEFINE_RFUNDAMENTAL_TYPED_IMPL(int64_t);
    } else if (XIsType(X(0), float16)) {
        DEFINE_RFUNDAMENTAL_TYPED_IMPL(float16);
    } else if (XIsType(X(0), float)) {
        DEFINE_RFUNDAMENTAL_TYPED_IMPL(float);
    } else if (XIsType(X(0), double)) {
        DEFINE_RFUNDAMENTAL_TYPED_IMPL(double);
    } else {
        LOG(FATAL) << DTypeString(X(0), {
            "int8", "uint8", "int32", "int64",
                "float16", "float32", "float64",
        });
    }
}

template <class Context> template <typename T>
void RMulGradientOp<Context>::EltwiseRunImpl() {
    auto* dy = X(-1).template data<T, Context>();

    if (Y(1)->name() != "NULL") {
        auto* a = X(0).template data<T, Context>();
        auto* db = Y(1)->template mutable_data<T, Context>();
        math::Mul(Y(1)->count(), dy, a, db, ctx());
    }

    if (Y(0)->name() != "NULL") {
        auto* b = X(1).template data<T, Context>();
        auto* da = Y(0)->template mutable_data<T, Context>();
        math::Mul(Y(0)->count(), dy, b, da, ctx());
    }
}

template <class Context> template <typename T>
void RMulGradientOp<Context>::BroadcastRunImpl(int type) {
    DEFINE_INPUT_DESC;
    auto* dy = X(-1).template data<T, Context>();

    if (Y(0)->name() != "NULL") {
        auto* b = X(1).template data<T, Context>();
        auto* da = Y(0)->template mutable_data<T, Context>();
        auto* scratch = ws()->template data
            <T, Context>({ B->count() })[0];
        math::Mul(B->count(), dy, b, scratch, ctx());
        vec32_t dims = { rows_, cols_ };
        vec32_t axes = { type - 2 };
        kernel::ReduceSum(
            2, dims.data(),
            1, axes.data(),
            1.f, scratch,
            da, ctx()
        );
    }

    if (Y(1)->name() != "NULL") {
        auto* a = X(0).template data<T, Context>();
        auto* db = Y(1)->template mutable_data<T, Context>();
        math::BroadcastMul(rows_, cols_, type - 2, dy, a, db, ctx());
    }
}

template <class Context>
void RMulGradientOp<Context>::RunOnDevice() {
    DEFINE_INPUT_DESC;
    Y(0)->ReshapeLike(*A);
    Y(1)->ReshapeLike(*B);

    if (XIsType(X(-1), int8_t)) {
        DEFINE_RFUNDAMENTAL_TYPED_IMPL(int8_t);
    } else if (XIsType(X(-1), uint8_t)) {
        DEFINE_RFUNDAMENTAL_TYPED_IMPL(uint8_t);
    } else if (XIsType(X(-1), int)) {
        DEFINE_RFUNDAMENTAL_TYPED_IMPL(int);
    } else if (XIsType(X(-1), int64_t)) {
        DEFINE_RFUNDAMENTAL_TYPED_IMPL(int64_t);
    } else if (XIsType(X(-1), float16)) {
        DEFINE_RFUNDAMENTAL_TYPED_IMPL(float16);
    } else if (XIsType(X(-1), float)) {
        DEFINE_RFUNDAMENTAL_TYPED_IMPL(float);
    } else if (XIsType(X(-1), double)) {
        DEFINE_RFUNDAMENTAL_TYPED_IMPL(double);
    } else {
        LOG(FATAL) << DTypeString(X(0), {
            "int8", "uint8", "int32", "int64",
            "float16", "float32", "float64",
        });
    }
}

DEPLOY_CPU(RMul);
#ifdef WITH_CUDA
DEPLOY_CUDA(RMul);
#endif

DEPLOY_CPU(RMulGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(RMulGradient);
#endif

OPERATOR_SCHEMA(RMul)
     /* A, B */
    .NumInputs(2)
     /* Y */
    .NumOutputs(1)
     /* B => Y */
    .Inplace({ { 1, 0 } });

OPERATOR_SCHEMA(RMulGradient)
     /* A, B, dY */
    .NumInputs(3)
     /* dA, dB */
    .NumOutputs(2);

namespace {

class GradientMaker : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GradientMaker);
    vector<OperatorDef> MakeDef() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), GO(0) }),
            vector<string>({ GI(0), GI(1) })
        );
    }
};

}  // namespace

REGISTER_GRADIENT(RMul, GradientMaker);

}  // namespace dragon