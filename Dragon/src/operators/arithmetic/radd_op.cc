#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/arithmetic/fundamental_op.h"

namespace dragon {

template <class Context> template <typename T>
void RAddOp<Context>::EltwiseRunWithType() {
    auto* x1 = Input(0).template data<T, Context>();
    auto* x2 = Input(1).template data<T, Context>();
    auto* y = Output(0)->template mutable_data<T, Context>();
    math::Add(Output(0)->count(), x1, x2, y, ctx());
}

template <class Context> template <typename T>
void RAddOp<Context>::BroadcastRunWithType(int type) {
    auto* x1 = Input(0).template data<T, Context>();
    auto* x2 = Input(1).template data<T, Context>();
    auto* y = Output(0)->template mutable_data<T, Context>();
    math::BroadcastAdd(rows, cols, type, x1, x2, y, ctx());
}

template <class Context>
void RAddOp<Context>::RunOnDevice() {
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

DEPLOY_CPU(RAdd);
#ifdef WITH_CUDA
DEPLOY_CUDA(RAdd);
#endif
OPERATOR_SCHEMA(RAdd)
    .NumInputs(2).NumOutputs(1)
    .Inplace({ { 1, 0 } });

template <class Context> template <typename T>
void RAddGradientOp<Context>::EltwiseRunWithType() {
    auto* dy = Input(-1).template data<T, Context>();

    if (Output(1)->name() != "ignore") {
        auto* dx2 = Output(1)->template mutable_data<T, Context>();
        ctx()->template Copy<T, Context, Context>(
            Output(1)->count(), dx2, dy);
    }

    if (Output(0)->name() != "ignore") {
        auto* dx1 = Output(0)->template mutable_data<T, Context>();
        ctx()->template Copy<T, Context, Context>(
            Output(0)->count(), dx1, dy);
    }
}

template <class Context> template <typename T>
void RAddGradientOp<Context>::BroadcastRunWithType(int type) {
    DEFINE_FUNDAMENTAL_OP_X1X2;
    auto* dy = Input(-1).template data<T, Context>();

    if (Output(0)->name() != "ignore") {
        auto* dx1 = Output(0)->template mutable_data<T, Context>();
        vector<int> dims = { rows, cols }, axes = { type - 2 };
        kernel::ReduceSum(2, dims.data(),
            1, axes.data(), 1.f, dy, dx1, ctx());
    }

    if (Output(1)->name() != "ignore") {
        auto* dx2 = Output(1)->template mutable_data<T, Context>();
        ctx()->template Copy<T, Context, Context>(
            X2->count(), dx2, dy);
    }
}

template <class Context>
void RAddGradientOp<Context>::RunOnDevice() {
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

DEPLOY_CPU(RAddGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(RAddGradient);
#endif

OPERATOR_SCHEMA(RAddGradient)
    .NumInputs(1).NumOutputs(2);

class GetRAddGradient : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetRAddGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ GO(0) }),
            vector<string>({ GI(0), GI(1) }));
    }
};

REGISTER_GRADIENT(RAdd, GetRAddGradient);

}  // namespace dragon