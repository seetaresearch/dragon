#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/arithmetic/fundamental_op.h"

namespace dragon {

template <class Context> template <typename T>
void RDivOp<Context>::EltwiseRunWithType() {
    auto* x1 = Input(0).template data<T, Context>();
    auto* x2 = Input(1).template data<T, Context>();
    auto* y = Output(0)->template mutable_data<T, Context>();
    math::Div(Output(0)->count(), x1, x2, y, ctx());
}

template <class Context> template <typename T>
void RDivOp<Context>::BroadcastRunWithType(int type) {
    auto* x1 = Input(0).template data<T, Context>();
    auto* x2 = Input(1).template data<T, Context>();
    auto* y = Output(0)->template mutable_data<T, Context>();
    math::BroadcastDiv(rows, cols, type, x1, x2, y, ctx());
}

template <class Context>
void RDivOp<Context>::RunOnDevice() {
    DECLARE_FUNDAMENTAL_OP_X1X2;
    Output(0)->ReshapeLike(Input(1));

    if (XIsType(Input(0), int8_t)) {
        DEFINE_FUNDAMENTAL_TYPED_RCALLER(int8_t);
    } else if (XIsType(Input(0), uint8_t)) {
        DEFINE_FUNDAMENTAL_TYPED_RCALLER(uint8_t);
    } else if (XIsType(Input(0), int)) {
        DEFINE_FUNDAMENTAL_TYPED_RCALLER(int);
    } else if (XIsType(Input(0), int64_t)) {
        DEFINE_FUNDAMENTAL_TYPED_RCALLER(int64_t);
    } else if (XIsType(Input(0), float16)) {
        DEFINE_FUNDAMENTAL_TYPED_RCALLER(float16);
    } else if (XIsType(Input(0), float)) {
        DEFINE_FUNDAMENTAL_TYPED_RCALLER(float);
    } else if (XIsType(Input(0), double)) {
        DEFINE_FUNDAMENTAL_TYPED_RCALLER(double);
    } else {
        LOG(FATAL) << DTypeHelper(Input(0), {
            "int8", "uint8", "int32", "int64",
                "float16", "float32", "float64",
        });
    }
}

DEPLOY_CPU(RDiv);
#ifdef WITH_CUDA
DEPLOY_CUDA(RDiv);
#endif
OPERATOR_SCHEMA(RDiv)
    .NumInputs(2).NumOutputs(1)
    .Inplace({ { 1, 0 } });

template <class Context> template <typename T>
void RDivGradientOp<Context>::EltwiseRunWithType() {
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
void RDivGradientOp<Context>::BroadcastRunWithType(int type) {
    DEFINE_FUNDAMENTAL_OP_X1X2;
    auto* dy = Input(-1).template data<T, Context>();

    if (Output(0)->name() != "ignore") {
        auto* x2 = Input(1).template data<T, Context>();
        auto* dx1 = Output(0)->template mutable_data<T, Context>();
        auto* c = ws()->template caches<T, Context>({ X2->count() })[0];
        math::Div(X2->count(), dy, x2, c, ctx());
        vector<int> dims = { rows, cols }, axes = { type - 2 };
        kernel::ReduceSum(2, dims.data(),
            1, axes.data(), 1.f, c, dx1, ctx());
    }

    if (Output(1)->name() != "ignore") {
        auto* x1 = Input(0).template data<T, Context>();
        auto* x2 = Input(1).template data<T, Context>();
        auto* dx2 = Output(1)->template mutable_data<T, Context>();
        // dX2 = -dY * X1 / (X2 ** 2)
        math::BroadcastMul(rows, cols, type - 2, dy, x1, dx2, ctx());
        math::Div(X2->count(), dx2, x2, dx2, ctx());
        math::Div(X2->count(), dx2, x2, dx2, ctx());
        math::Scale(X2->count(), -1.f, dx2, dx2, ctx());
    }
}

template <class Context>
void RDivGradientOp<Context>::RunOnDevice() {
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

DEPLOY_CPU(RDivGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(RDivGradient);
#endif

OPERATOR_SCHEMA(RDivGradient)
    .NumInputs(3).NumOutputs(2);

class GetRDivGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetRDivGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), GO(0) }),
            vector<string>({ GI(0), GI(1) }));
    }
};

REGISTER_GRADIENT(RDiv, GetRDivGradient);

}  // namespace dragon