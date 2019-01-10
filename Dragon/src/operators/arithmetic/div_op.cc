#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/arithmetic/fundamental_op.h"

namespace dragon {

template <class Context> template <typename T>
void DivOp<Context>::EltwiseRunWithType() {
    auto* x1 = Input(0).template data<T, Context>();
    auto* x2 = Input(1).template data<T, Context>();
    auto* y = Output(0)->template mutable_data<T, Context>();
    math::Div(Output(0)->count(), x1, x2, y, ctx());
}

template <class Context> template <typename T>
void DivOp<Context>::BroadcastRunWithType(int type) {
    auto* x1 = Input(0).template data<T, Context>();
    auto* x2 = Input(1).template data<T, Context>();
    auto* y = Output(0)->template mutable_data<T, Context>();
    math::BroadcastDiv(rows, cols, type, x1, x2, y, ctx());
}

template <class Context>
void DivOp<Context>::RunOnDevice() {
    DECLARE_FUNDAMENTAL_OP_X1X2;
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), int8_t)) {
        DEFINE_FUNDAMENTAL_TYPED_CALLER(int8_t);
    } else if (XIsType(Input(0), uint8_t)) {
        DEFINE_FUNDAMENTAL_TYPED_CALLER(uint8_t);
    } else if (XIsType(Input(0), int)) {
        DEFINE_FUNDAMENTAL_TYPED_CALLER(int);
    } else if (XIsType(Input(0), int64_t)) {
        DEFINE_FUNDAMENTAL_TYPED_CALLER(int64_t);
    } else if (XIsType(Input(0), float16)) {
        DEFINE_FUNDAMENTAL_TYPED_CALLER(float16);
    } else if (XIsType(Input(0), float)) {
        DEFINE_FUNDAMENTAL_TYPED_CALLER(float);
    } else if (XIsType(Input(0), double)) {
        DEFINE_FUNDAMENTAL_TYPED_CALLER(double);
    } else {
        LOG(FATAL) << DTypeHelper(Input(0), {
            "int8", "uint8", "int32", "int64",
                "float16", "float32", "float64",
        });
    }
}

DEPLOY_CPU(Div);
#ifdef WITH_CUDA
DEPLOY_CUDA(Div);
#endif
OPERATOR_SCHEMA(Div)
    .NumInputs(2).NumOutputs(1)
    .Inplace({ { 0, 0 } });

template <class Context> template <typename T>
void DivGradientOp<Context>::EltwiseRunWithType() {
    DEFINE_FUNDAMENTAL_OP_X1X2;
    auto* dy = Input(-1).template data<T, Context>();

    if (Output(1)->name() != "ignore") {
        auto* x1 = Input(0).template data<T, Context>();
        auto* x2 = Input(1).template data<T, Context>();
        auto* dx2 = Output(1)->template mutable_data<T, Context>();
        auto* c = ws()->template caches<T, Context>({ X1->count() })[0];
        // dX2 = -dY * X1 / (X2 ** 2)
        math::Mul(X1->count(), dy, x1, c, ctx());
        math::Square(X2->count(), x2, dx2, ctx());
        math::Div(X2->count(), c, dx2, dx2, ctx());
        math::Scale(X2->count(), -1.f, dx2, dx2, ctx());
    }

    if (Output(0)->name() != "ignore") {
        auto* x2 = Input(1).template data<T, Context>();
        auto* dx1 = Output(0)->template mutable_data<T, Context>();
        math::Div(X1->count(), dy, x2, dx1, ctx());
    }
}

template <class Context> template <typename T>
void DivGradientOp<Context>::BroadcastRunWithType(int type) {
    DEFINE_FUNDAMENTAL_OP_X1X2;
    auto* dy = Input(-1).template data<T, Context>();

    if (Output(1)->name() != "ignore") {
        auto* x1 = Input(0).template data<T, Context>();
        auto* x2 = Input(1).template data<T, Context>();
        auto* dx2 = Output(1)->template mutable_data<T, Context>();
        auto cs = ws()->template caches<T, Context>(
            { X1->count(), X2->count() });
        // dX2 = -dY * X1 / (X2 ** 2)
        math::Mul(X1->count(), dy, x1, cs[0], ctx());
        math::Square(X2->count(), x2, cs[1], ctx());
        math::BroadcastDiv(rows, cols, type, cs[0], cs[1], cs[0], ctx());
        vector<int> dims = { rows, cols }, axes = { type };
        kernel::ReduceSum(2, dims.data(),
            1, axes.data(), -1.f, cs[0], dx2, ctx());
    }

    if (Output(0)->name() != "ignore") {
        auto* x2 = Input(1).template data<T, Context>();
        auto* dx1 = Output(0)->template mutable_data<T, Context>();
        CHECK(dy != dx1) << "\nCan't set inplace if X2 was broadcast.";
        math::BroadcastDiv(rows, cols, type, dy, x2, dx1, ctx());
    }
}

template <class Context>
void DivGradientOp<Context>::RunOnDevice() {
    DEFINE_FUNDAMENTAL_OP_X1X2;
    Output(0)->ReshapeLike(*X1);
    Output(1)->ReshapeLike(*X2);

    if (XIsType(Input(-1), int8_t)) {
        DEFINE_FUNDAMENTAL_TYPED_CALLER(int8_t);
    } else if (XIsType(Input(-1), uint8_t)) {
        DEFINE_FUNDAMENTAL_TYPED_CALLER(uint8_t);
    } else if (XIsType(Input(-1), int)) {
        DEFINE_FUNDAMENTAL_TYPED_CALLER(int);
    } else if (XIsType(Input(-1), int64_t)) {
        DEFINE_FUNDAMENTAL_TYPED_CALLER(int64_t);
    } else if (XIsType(Input(-1), float16)) {
        DEFINE_FUNDAMENTAL_TYPED_CALLER(float16);
    } else if (XIsType(Input(-1), float)) {
        DEFINE_FUNDAMENTAL_TYPED_CALLER(float);
    } else if (XIsType(Input(-1), double)) {
        DEFINE_FUNDAMENTAL_TYPED_CALLER(double);
    } else {
        LOG(FATAL) << DTypeHelper(Input(0), {
            "int8", "uint8", "int32", "int64",
                  "float16", "float32", "float64",
        });
    }
}

DEPLOY_CPU(DivGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(DivGradient);
#endif

OPERATOR_SCHEMA(DivGradient)
    .NumInputs(3).NumOutputs(2)
    .Inplace({ { 2, 0 } });

class GetDivGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetDivGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), GO(0) }),
            vector<string>({ GI(0), GI(1) }));
    }
};

REGISTER_GRADIENT(Div, GetDivGradient);

}  // namespace dragon