#include "core/workspace.h"
#include "utils/math_functions.h"
#include "operators/arithmetic/fundamental_op.h"

namespace dragon {

template <class Context> template <typename T>
void RMulOp<Context>::EltwiseRunWithType() {
    auto* x1 = Input(0).template data<T, Context>();
    auto* x2 = Input(1).template data<T, Context>();
    auto* y = Output(0)->template mutable_data<T, Context>();
    math::Mul<T, Context>(Output(0)->count(), x1, x2, y);
}

template <class Context> template <typename T>
void RMulOp<Context>::BroadcastRunWithType(int type) {
    TIndex outer_dim, inner_dim;
    auto* x1 = Input(0).template data<T, Context>();
    auto* x2 = Input(1).template data<T, Context>();
    auto* y = Output(0)->template mutable_data<T, Context>();
    auto* c = ws()->template caches<T, Context>({
        Output(0)->count() })[0];

    if (type == 0 || type == 1) {
        if (type == 0) {
            outer_dim = Input(1).count();
            inner_dim = 1;
        } else {
            outer_dim = Input(1).count(0, Input(1).axis(-1));
            inner_dim = Input(1).dim(-1);
        }
        DECLARE_MULTIPLIER(multiplier, outer_dim);
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                outer_dim, inner_dim, 1,
                    1.0, multiplier, x1,
                        0.0, c, &ctx());
        math::Mul<T, Context>(Output(0)->count(), c, x2, y);
    } else if (type == 2) {
        outer_dim = Input(1).dim(0);
        inner_dim = Input(1).count(1);
        DECLARE_MULTIPLIER(multiplier, inner_dim);
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                outer_dim, inner_dim, 1,
                    1.0, x1, multiplier,
                        0.0, c, &ctx());
        math::Mul<T, Context>(Output(0)->count(), c, x2, y);
    }
}

template <class Context>
void RMulOp<Context>::RunOnDevice() {
    DeclareX1X2;
    Output(0)->ReshapeLike(Input(1));

    if (XIsType(Input(0), float)) {
        RRunByX1X2(float);
    } else if (XIsType(Input(0), float16)) {
        RRunByX1X2(float16);
    } else {
        LOG(FATAL) << DTypeHelper(Input(0),
            { "float32", "float16" });
    }
}

DEPLOY_CPU(RMul);
#ifdef WITH_CUDA
DEPLOY_CUDA(RMul);
#endif
OPERATOR_SCHEMA(RMul)
    .NumInputs(2).NumOutputs(1)
    .Inplace({ { 1, 0 } });

template <class Context> template <typename T>
void RMulGradientOp<Context>::EltwiseRunWithType() {
    auto* dy = Input(-1).template data<T, Context>();

    if (Output(1)->name() != "ignore") {
        auto* x1 = Input(0).template data<T, Context>();
        auto* dx2 = Output(1)->template mutable_data<T, Context>();
        math::Mul<T, Context>(Output(1)->count(), dy, x1, dx2);
    }

    if (Output(0)->name() != "ignore") {
        auto* x2 = Input(1).template data<T, Context>();
        auto* dx1 = Output(0)->template mutable_data<T, Context>();
        math::Mul<T, Context>(Output(0)->count(), dy, x2, dx1);
    }
}

template <class Context> template <typename T>
void RMulGradientOp<Context>::BroadcastRunWithType(int type) {
    DefineX1X2;
    TIndex outer_dim, inner_dim;
    auto* dy = Input(-1).template data<T, Context>();

    if (type == 0) {
        outer_dim = X2->count();
        inner_dim = 1;
    } else if (type == 1) {
        outer_dim = X2->count(0, X2->axis(-1));
        inner_dim = X2->dim(-1);
    } else if (type == 2) {
        outer_dim = X2->dim(0);
        inner_dim = X2->count(1);
    }

    if (Output(0)->name() != "ignore") {
        auto* x2 = Input(1).template data<T, Context>();
        auto* dx1 = Output(0)->template mutable_data<T, Context>();
        auto* c = ws()->template caches<T, Context>({ X2->count() })[0];
        math::Mul<T, Context>(X2->count(), dy, x2, c);
        if (type == 0 || type == 1) {
            DECLARE_MULTIPLIER(multiplier, outer_dim);
            math::Gemv<T, Context>(
                CblasTrans, outer_dim, inner_dim,
                    1.0, c, multiplier,
                        0.0, dx1, &ctx());
        } else if (type == 2) {
            DECLARE_MULTIPLIER(multiplier, inner_dim);
            math::Gemv<T, Context>(
                CblasNoTrans, outer_dim, inner_dim,
                    1.0, c, multiplier,
                        0.0, dx1, &ctx());
        }
    }

    if (Output(1)->name() != "ignore") {
        auto* x1 = Input(0).template data<T, Context>();
        auto* dx2 = Output(1)->template mutable_data<T, Context>();
        if (type == 0 || type == 1) {
            DECLARE_MULTIPLIER(multiplier, outer_dim);
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    outer_dim, inner_dim, 1,
                        1.0, multiplier, x1,
                            0.0, dx2, &ctx());
        } else if (type == 2) {
            DECLARE_MULTIPLIER(multiplier, inner_dim);
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    outer_dim, inner_dim, 1,
                        1.0, x1, multiplier,
                            0.0, dx2, &ctx());
        }
        math::Mul<T, Context>(X2->count(), dy, dx2, dx2);
    }
}

template <class Context>
void RMulGradientOp<Context>::RunOnDevice() {
    DefineX1X2;
    Output(0)->ReshapeLike(*X1);
    Output(1)->ReshapeLike(*X2);

    if (XIsType(Input(-1), float)) {
        RRunByX1X2(float);
    } else if (XIsType(Input(-1), float16)) {
        RRunByX1X2(float16);
    } else {
        LOG(FATAL) << DTypeHelper(Input(-1),
            { "float32", "float16" });
    }
}

DEPLOY_CPU(RMulGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(RMulGradient);
#endif
OPERATOR_SCHEMA(RMulGradient).NumInputs(3).NumOutputs(2);

class GetRMulGradient : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetRMulGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0), GI(1)});
    }
};
REGISTER_GRADIENT(RMul, GetRMulGradient);

}    // namespace dragon