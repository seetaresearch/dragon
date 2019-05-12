#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/arithmetic/fundamental_op.h"

namespace dragon {

template <class Context> template <typename T>
void DivOp<Context>::EltwiseRunImpl() {
    auto* a = X(0).template data<T, Context>();
    auto* b = X(1).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();
    math::Div(Y(0)->count(), a, b, y, ctx());
}

template <class Context> template <typename T>
void DivOp<Context>::BroadcastRunImpl(int type) {
    auto* a = X(0).template data<T, Context>();
    auto* b = X(1).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();
    math::BroadcastDiv(rows_, cols_, type, a, b, y, ctx());
}

template <class Context>
void DivOp<Context>::RunOnDevice() {
    DECLARE_INPUT_DESC;
    Y(0)->ReshapeLike(X(0));

    if (XIsType(X(0), int8_t)) {
        DEFINE_FUNDAMENTAL_TYPED_IMPL(int8_t);
    } else if (XIsType(X(0), uint8_t)) {
        DEFINE_FUNDAMENTAL_TYPED_IMPL(uint8_t);
    } else if (XIsType(X(0), int)) {
        DEFINE_FUNDAMENTAL_TYPED_IMPL(int);
    } else if (XIsType(X(0), int64_t)) {
        DEFINE_FUNDAMENTAL_TYPED_IMPL(int64_t);
    } else if (XIsType(X(0), float16)) {
        DEFINE_FUNDAMENTAL_TYPED_IMPL(float16);
    } else if (XIsType(X(0), float)) {
        DEFINE_FUNDAMENTAL_TYPED_IMPL(float);
    } else if (XIsType(X(0), double)) {
        DEFINE_FUNDAMENTAL_TYPED_IMPL(double);
    } else {
        LOG(FATAL) << DTypeString(X(0), {
            "int8", "uint8", "int32", "int64",
            "float16", "float32", "float64",
        });
    }
}

template <class Context> template <typename T>
void DivGradientOp<Context>::EltwiseRunImpl() {
    DEFINE_INPUT_DESC;
    auto* dy = X(-1).template data<T, Context>();

    if (Y(1)->name() != "NULL") {
        auto* a = X(0).template data<T, Context>();
        auto* b = X(1).template data<T, Context>();
        auto* db = Y(1)->template mutable_data<T, Context>();
        auto* scratch = ws()->template data
            <T, Context>({ A->count() })[0];
        // dB = -dY * X1 / (B ** 2)
        math::Mul(A->count(), dy, a, scratch, ctx());
        math::Square(B->count(), b, db, ctx());
        math::Div(B->count(), scratch, db, db, ctx());
        math::Scale(B->count(), -1.f, db, db, ctx());
    }

    if (Y(0)->name() != "NULL") {
        auto* b = X(1).template data<T, Context>();
        auto* da = Y(0)->template mutable_data<T, Context>();
        math::Div(A->count(), dy, b, da, ctx());
    }
}

template <class Context> template <typename T>
void DivGradientOp<Context>::BroadcastRunImpl(int type) {
    DEFINE_INPUT_DESC;
    auto* dy = X(-1).template data<T, Context>();

    if (Y(1)->name() != "NULL") {
        auto* a = X(0).template data<T, Context>();
        auto* b = X(1).template data<T, Context>();
        auto* db = Y(1)->template mutable_data<T, Context>();
        auto buf = ws()->template data<T, Context>
            ({ A->count(), B->count() });
        // dB = -dY * A / (B ** 2)
        math::Mul(A->count(), dy, a, buf[0], ctx());
        math::Square(B->count(), b, buf[1], ctx());
        math::BroadcastDiv(
            rows_, cols_, type,
            buf[0], buf[1],
            buf[0], ctx()
        );
        vec32_t dims = { rows_, cols_ };
        vec32_t axes = { type };
        kernel::ReduceSum(
            2, dims.data(),
            1, axes.data(),
            -1.f, buf[0],
            db, ctx()
        );
    }

    if (Y(0)->name() != "NULL") {
        auto* b = X(1).template data<T, Context>();
        auto* da = Y(0)->template mutable_data<T, Context>();
        math::BroadcastDiv(rows_, cols_, type, dy, b, da, ctx());
    }
}

template <class Context>
void DivGradientOp<Context>::RunOnDevice() {
    DEFINE_INPUT_DESC;
    Y(0)->ReshapeLike(*A);
    Y(1)->ReshapeLike(*B);

    if (XIsType(X(-1), int8_t)) {
        DEFINE_FUNDAMENTAL_TYPED_IMPL(int8_t);
    } else if (XIsType(X(-1), uint8_t)) {
        DEFINE_FUNDAMENTAL_TYPED_IMPL(uint8_t);
    } else if (XIsType(X(-1), int)) {
        DEFINE_FUNDAMENTAL_TYPED_IMPL(int);
    } else if (XIsType(X(-1), int64_t)) {
        DEFINE_FUNDAMENTAL_TYPED_IMPL(int64_t);
    } else if (XIsType(X(-1), float16)) {
        DEFINE_FUNDAMENTAL_TYPED_IMPL(float16);
    } else if (XIsType(X(-1), float)) {
        DEFINE_FUNDAMENTAL_TYPED_IMPL(float);
    } else if (XIsType(X(-1), double)) {
        DEFINE_FUNDAMENTAL_TYPED_IMPL(double);
    } else {
        LOG(FATAL) << DTypeString(X(0), {
            "int8", "uint8", "int32", "int64",
            "float16", "float32", "float64",
        });
    }
}

DEPLOY_CPU(Div);
#ifdef WITH_CUDA
DEPLOY_CUDA(Div);
#endif

DEPLOY_CPU(DivGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(DivGradient);
#endif

OPERATOR_SCHEMA(Div)
     /* A, B */
    .NumInputs(2)
     /* Y */
    .NumOutputs(1)
     /* A => Y */
    .Inplace({ { 0, 0 } });

OPERATOR_SCHEMA(DivGradient)
     /* A, B, dY */
    .NumInputs(3)
     /* dA, dB */
    .NumOutputs(2)
     /* dY => dA */
    .Inplace({ { 2, 0 } });

namespace {

class GradientMaker final : public GradientMakerBase {
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

REGISTER_GRADIENT(Div, GradientMaker);

}  // namespace dragon