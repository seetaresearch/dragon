#include "core/workspace.h"
#include "utils/math_functions.h"
#include "operators/arithmetic/fundamental_op.h"

namespace dragon {

template <class Context> template <typename T>
void AddOp<Context>::EltwiseRunWithType() {
    auto* x1 = Input(0).template data<T, Context>();
    auto* x2 = Input(1).template data<T, Context>();
    auto* y = Output(0)->template mutable_data<T, Context>();
    math::Add<T, Context>(Output(0)->count(), x1, x2, y, ctx());
}

template <class Context> template <typename T>
void AddOp<Context>::BroadcastRunWithType(int type) {
    TIndex outer_dim, inner_dim;
    auto* x1 = Input(0).template data<T, Context>();
    auto* x2 = Input(1).template data<T, Context>();
    auto* y = Output(0)->template mutable_data<T, Context>();

    ctx()->template Copy<T, Context, Context>(
        Output(0)->count(), y, x1);

    if (type == 0 || type == 1) {
        if (type == 0) {
            x2 = Input(1).template data<T, CPUContext>();
            math::AddScalar<T, Context>(Output(0)->count(),
                dragon_cast<float, T>(x2[0]), y, ctx());
        } else {
            outer_dim = Input(0).count(0, Input(0).axis(-1));
            inner_dim = Input(0).dim(-1);
            DECLARE_MULTIPLIER(multiplier, outer_dim);
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    outer_dim, inner_dim, 1,
                        1.0, multiplier, x2,
                            1.0, y, ctx());
        }
    } else if (type == 2) {
        outer_dim = Input(0).dim(0);
        inner_dim = Input(0).count(1);
        DECLARE_MULTIPLIER(multiplier, inner_dim);
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                outer_dim, inner_dim, 1,
                    1.0, x2, multiplier,
                        1.0, y, ctx());
    }
}

template <class Context>
void AddOp<Context>::RunOnDevice() {
    DeclareX1X2;
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) {
        RunByX1X2(float);
    } else if (XIsType(Input(0), float16)) {
        RunByX1X2(float16);
    } else {
        LOG(FATAL) << DTypeHelper(Input(0),
            { "float32", "float16" });
    }
}

DEPLOY_CPU(Add);
#ifdef WITH_CUDA
DEPLOY_CUDA(Add);
#endif
OPERATOR_SCHEMA(Add)
    .NumInputs(2).NumOutputs(1)
    .Inplace({ { 0, 0 } });

template <class Context> template <typename T>
void AddGradientOp<Context>::EltwiseRunWithType() {
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
void AddGradientOp<Context>::BroadcastRunWithType(int type) {
    DefineX1X2;
    TIndex outer_dim, inner_dim;
    auto* dy = Input(-1).template data<T, Context>();

    if (Output(1)->name() != "ignore") {
        auto* dx2 = Output(1)->template mutable_data<T, Context>();
        if (type == 0 || type == 1) {
            if (type == 0) {
                outer_dim = X1->count();
                inner_dim = 1;
            } else {
                outer_dim = X1->count(0, X1->axis(-1));
                inner_dim = X1->dim(-1);
            }
            DECLARE_MULTIPLIER(multiplier, outer_dim);
            math::Gemv<T, Context>(
                CblasTrans, outer_dim, inner_dim,
                    1.0, dy, multiplier,
                        0.0, dx2, ctx());
        } else if (type == 2) {
            outer_dim = X1->dim(0);
            inner_dim = X1->count(1);
            DECLARE_MULTIPLIER(multiplier, inner_dim);
            math::Gemv<T, Context>(
                CblasNoTrans, outer_dim, inner_dim,
                    1.0, dy, multiplier,
                        0.0, dx2, ctx());
        }
    }

    if (Output(0)->name() != "ignore") {
        auto* dx1 = Output(0)->template mutable_data<T, Context>();
        ctx()->template Copy<T, Context, Context>(
            X1->count(), dx1, dy);
    }
}

template <class Context>
void AddGradientOp<Context>::RunOnDevice() {
    DefineX1X2;
    Output(0)->ReshapeLike(*X1);
    Output(1)->ReshapeLike(*X2);

    if (XIsType(Input(-1), float)) {
        RunByX1X2(float);
    } else if (XIsType(Input(-1), float16)) {
        RunByX1X2(float16);
    } else {
        LOG(FATAL) << DTypeHelper(Input(-1),
            { "float32", "float16" });
    }
}

DEPLOY_CPU(AddGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(AddGradient);
#endif
OPERATOR_SCHEMA(AddGradient)
    .NumInputs(1).NumOutputs(2)
    .Inplace({ { 0, 0 } });

class GetAddGradient : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetAddGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {GO(0)},
            vector<string> {GI(0), GI(1)});
    }
};
REGISTER_GRADIENT(Add, GetAddGradient);

}    // namespace dragon