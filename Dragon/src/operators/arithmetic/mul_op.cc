#include "core/workspace.h"
#include "utils/math_functions.h"
#include "operators/arithmetic/fundamental_op.h"

namespace dragon {

template <class Context> template <typename T>
void MulOp<Context>::EltwiseRunWithType() {
    auto* x1 = Input(0).template data<T, Context>();
    auto* x2 = Input(1).template data<T, Context>();
    auto* y = Output(0)->template mutable_data<T, Context>();
    math::Mul<T, Context>(Output(0)->count(), x1, x2, y, ctx());
}

template <class Context> template <typename T>
void MulOp<Context>::BroadcastRunWithType(int type) {
    TIndex outer_dim, inner_dim;
    auto* x1 = Input(0).template data<T, Context>();
    auto* x2 = Input(1).template data<T, Context>();
    auto* y = Output(0)->template mutable_data<T, Context>();

    if (type == 0) {
        x2 = Input(1).template data<T, CPUContext>();
        ctx()->template Copy<T, Context, Context>(
            Output(0)->count(), y, x1);
        math::MulScalar<T, Context>(Output(0)->count(),
            dragon_cast<float, T>(x2[0]), y, ctx());
    } else if (type == 1) {
        outer_dim = Input(0).count(0, Input(0).axis(-1));
        inner_dim = Input(0).dim(-1);
        DECLARE_MULTIPLIER(multiplier, outer_dim);
        auto* c = ws()->template caches<T, Context>(
            { Output(0)->count() })[0];
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                outer_dim, inner_dim, 1,
                    1.0, multiplier, x2,
                        0.0, c, ctx());
        math::Mul<T, Context>(
            Output(0)->count(), x1, c, y, ctx());
    } else if (type == 2) {
        outer_dim = Input(0).dim(0);
        inner_dim = Input(0).count(1);
        DECLARE_MULTIPLIER(multiplier, inner_dim);
        auto* c = ws()->template caches<T, Context>(
            { Output(0)->count() })[0];
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                outer_dim, inner_dim, 1,
                    1.0, x2, multiplier,
                        0.0, c, ctx());
        math::Mul<T, Context>(
            Output(0)->count(), x1, c, y, ctx());
    }
}

template <class Context>
void MulOp<Context>::RunOnDevice() {
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

DEPLOY_CPU(Mul);
#ifdef WITH_CUDA
DEPLOY_CUDA(Mul);
#endif
OPERATOR_SCHEMA(Mul)
    .NumInputs(2).NumOutputs(1)
    .Inplace({ { 0, 0 } });

template <class Context> template <typename T>
void MulGradientOp<Context>::EltwiseRunWithType() {
    auto* dy = Input(-1).template data<T, Context>();

    if (Output(1)->name() != "ignore") {
        auto* x1 = Input(0).template data<T, Context>();
        auto* dx2 = Output(1)->template mutable_data<T, Context>();
        math::Mul<T, Context>(Output(1)->count(), dy, x1, dx2, ctx());
    }

    if (Output(0)->name() != "ignore") {
        auto* x2 = Input(1).template data<T, Context>();
        auto* dx1 = Output(0)->template mutable_data<T, Context>();
        math::Mul<T, Context>(Output(0)->count(), dy, x2, dx1, ctx());
    }
}

template <class Context> template <typename T>
void MulGradientOp<Context>::BroadcastRunWithType(int type) {
    DefineX1X2;
    TIndex outer_dim, inner_dim;
    auto* dy = Input(-1).template data<T, Context>();

    if (type == 0) {
        outer_dim = X1->count();
        inner_dim = 1;
    } else if (type == 1) {
        outer_dim = X1->count(0, X1->axis(-1));
        inner_dim = X1->dim(-1);
    } else if (type == 2) {
        outer_dim = X1->dim(0);
        inner_dim = X1->count(1);
    }

    if (Output(1)->name() != "ignore") {
        auto* x1 = Input(0).template data<T, Context>();
        auto* dx2 = Output(1)->template mutable_data<T, Context>();
        auto* c = ws()->template caches<T, Context>({ X1->count() })[0];
        math::Mul<T, Context>(X1->count(), dy, x1, c, ctx());
        if (type == 0 || type == 1) {
            DECLARE_MULTIPLIER(multiplier, outer_dim);
            math::Gemv<T, Context>(
                CblasTrans, outer_dim, inner_dim,
                    1.0, c, multiplier,
                        0.0, dx2, ctx());
        } else if (type == 2) {
            DECLARE_MULTIPLIER(multiplier, inner_dim);
            math::Gemv<T, Context>(
                CblasNoTrans, outer_dim, inner_dim,
                    1.0, c, multiplier,
                        0.0, dx2, ctx());
        }
    }

    if (Output(0)->name() != "ignore") {
        auto* x2 = Input(1).template data<T, Context>();
        auto* dx1 = Output(0)->template mutable_data<T, Context>();
        CHECK(dy != dx1) << "\nCan't set inplace if X2 was broadcast.";
        if (type == 0 || type == 1) {
            DECLARE_MULTIPLIER(multiplier, outer_dim);
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    outer_dim, inner_dim, 1,
                        1.0, multiplier, x2,
                            0.0, dx1, ctx());
        } else if (type == 2) {
            DECLARE_MULTIPLIER(multiplier, inner_dim);
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    outer_dim, inner_dim, 1,
                        1.0, x2, multiplier,
                            0.0, dx1, ctx());
        }
        math::Mul<T, Context>(X1->count(), dy, dx1, dx1, ctx());
    }
}

template <class Context>
void MulGradientOp<Context>::RunOnDevice() {
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

DEPLOY_CPU(MulGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(MulGradient);
#endif
OPERATOR_SCHEMA(MulGradient)
    .NumInputs(3).NumOutputs(2)
    .Inplace({ { 2, 0 } });

class GetMulGradient : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetMulGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0), GI(1)});
    }
};
REGISTER_GRADIENT(Mul, GetMulGradient);

}    // namespace dragon