#include "operators/arithmetic/div_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h" 

namespace dragon {

template <class Context> template <typename T>
void DivOp<Context>::EltwiseRunWithType() {
    auto* X1data = Input(0).template data<T, Context>();
    auto* X2data = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    math::Div<T, Context>(Input(0).count(), X1data, X2data, Ydata);
}

template <class Context> template <typename T>
void DivOp<Context>::BroadcastRunWithType(int type) {
    TIndex outer_dim, inner_dim;
    auto* X1data = Input(0).template data<T, Context>();
    auto* X2data = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    if (type == 0 || type == 1) {
        if (type == 0) {
            outer_dim = Input(0).count();
            inner_dim = 1;
        } else {
            outer_dim = Input(0).count(0, Input(0).axis(-1));
            inner_dim = Input(0).dim(-1);
        }
        DECLARE_MULTIPLIER(multiplier, outer_dim);
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                outer_dim, inner_dim, 1,
                    1.0, multiplier, X2data, 0.0, Ydata);
        math::Div<T, Context>(Input(0).count(), X1data, Ydata, Ydata);
    } 
    else if (type == 2) {
        outer_dim = Input(0).dim(0);
        inner_dim = Input(0).count(1);
        DECLARE_MULTIPLIER(multiplier, inner_dim);
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                outer_dim, inner_dim, 1,
                    1.0, X2data, multiplier, 0.0, Ydata);
        math::Div<T, Context>(Input(0).count(), X1data, Ydata, Ydata);
    }
}

template <class Context>
void DivOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) {
        if (Input(0).dims() == Input(1).dims()) 
            EltwiseRunWithType<float>();
        else if (Input(0).dim(0) == Input(1).dim(0) && Input(1).count(1) == 1) 
            BroadcastRunWithType<float>(2);
        else if (Input(0).dim(-1) == Input(1).dim(-1) && 
                 Input(1).count(0, Input(1).axis(-1)) == 1)  
            BroadcastRunWithType<float>(1);
        else if (Input(1).ndim() == 1 && Input(1).dim(0) == 1)
            BroadcastRunWithType<float>(0);
        else LOG(FATAL) << "Could not be broadcast together with shapes "
                        << Input(0).dim_string() << "  " << Input(1).dim_string();
    } else if (XIsType(Input(0), float16)) {
        if (Input(0).dims() == Input(1).dims())
            EltwiseRunWithType<float16>();
        else if (Input(0).dim(0) == Input(1).dim(0) && Input(1).count(1) == 1)
            BroadcastRunWithType<float16>(2);
        else if (Input(0).dim(-1) == Input(1).dim(-1) &&
            Input(1).count(0, Input(1).axis(-1)) == 1)
            BroadcastRunWithType<float16>(1);
        else if (Input(1).ndim() == 1 && Input(1).dim(0) == 1)
            BroadcastRunWithType<float16>(0);
        else LOG(FATAL) << "Could not be broadcast together with shapes "
                        << Input(0).dim_string() << "  " << Input(1).dim_string();
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(Div);
#ifdef WITH_CUDA
DEPLOY_CUDA(Div);
#endif
OPERATOR_SCHEMA(Div).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void DivGradientOp<Context>::EltwiseRunWithType() {
    auto* dYdata = Input(2).template data<T, Context>();
    if (Output(1)->name() != "ignore") {
        auto* X1data = Input(0).template data<T, Context>();
        auto* X2data = Input(1).template data<T, Context>();
        auto* dX1data = Output(0)->template mutable_data<T, Context>();
        auto* dX2data = Output(1)->template mutable_data<T, Context>();
        math::Mul<T, Context>(Input(-1).count(), dYdata, X1data, dX1data); // dY * X_{1}
        math::Square<T, Context>(Input(1).count(), X2data, dX2data); // X_{2}^{2}
        math::Inv<T, Context>(Input(1).count(), -1.0, dX2data, dX2data); // -1 / X_{2}^{2}
        math::Mul<T, Context>(Input(1).count(), dX1data, dX2data, dX2data);
    }
    if (Output(0)->name() != "ignore") {
        auto* X2data = Input(1).template data<T, Context>();
        auto* dX1data = Output(0)->template mutable_data<T, Context>();
        math::Div<T, Context>(Input(0).count(), dYdata, X2data, dX1data);
    }
}

template <class Context> template <typename T>
void DivGradientOp<Context>::BroadcastRunWithType(int type) {
    TIndex outer_dim, inner_dim;
    auto* dYdata = Input(2).template data<T, Context>();
    if (type == 0) {
        outer_dim = Input(0).count();
        inner_dim = 1;
    } else if (type == 1) {
        outer_dim = Input(0).count(0, Input(0).axis(-1));
        inner_dim = Input(0).dim(-1);
    } else if (type == 2) {
        outer_dim = Input(0).dim(0);
        inner_dim = Input(0).count(1);
    }

    if (Output(1)->name() != "ignore") {
        auto* X1data = Input(0).template data<T, Context>();
        auto* X2data = Input(1).template data<T, Context>();
        auto* dX1data = Output(0)->template mutable_data<T, Context>();
        auto* dX2data = Output(1)->template mutable_data<T, Context>();
        auto* Bdata = ws()->template caches<T, Context>({ Input(1).count() })[0];
        math::Mul<T, Context>(Input(-1).count(), dYdata, X1data, dX1data); // dY * X_{1}
        if (type == 0 || type == 1) {
            DECLARE_MULTIPLIER(multiplier, outer_dim);
            math::Square<T, Context>(Input(1).count(), X2data, dX2data); // X_{2}^{2}
            math::Inv<T, Context>(Input(1).count(), -1.0, dX2data, dX2data); // -1 / X_{2}^{2}
            math::Gemv<T, Context>(
                CblasTrans,
                    outer_dim, inner_dim, 1.0,
                        dX1data, multiplier, 0.0, Bdata);
        }
        else if (type == 2) {
            DECLARE_MULTIPLIER(multiplier, inner_dim);
            math::Square<T, Context>(Input(1).count(), X2data, dX2data); // X_{2}^{2}
            math::Inv<T, Context>(Input(1).count(), -1.0, dX2data, dX2data); // -1 / X_{2}^{2}
            math::Gemv<T, Context>(
                CblasNoTrans,
                    outer_dim, inner_dim, 1.0,
                        dX1data, multiplier, 0.0, Bdata);
        }
        math::Mul<T, Context>(Input(1).count(), Bdata, dX2data, dX2data);
    }

    if (Output(0)->name() != "ignore") {
        auto* X2data = Input(1).template data<T, Context>();
        auto* dX1data = Output(0)->template mutable_data<T, Context>();
        if (type == 0 || type == 1) {
            DECLARE_MULTIPLIER(multiplier, outer_dim);
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    outer_dim, inner_dim, 1,
                        1.0, multiplier, X2data, 0.0, dX1data);
        } else if (type == 2) {
            DECLARE_MULTIPLIER(multiplier, inner_dim);
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    outer_dim, inner_dim, 1,
                        1.0, X2data, multiplier, 0.0, dX1data);
        }
        math::Div<T, Context>(Output(0)->count(), dYdata, dX1data, dX1data);
    }
}

template <class Context>
void DivGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));
    Output(1)->ReshapeLike(Input(1));

    if (XIsType(Input(0), float)) {
        if (Input(0).dims() == Input(1).dims())
            EltwiseRunWithType<float>();
        else if (Input(0).dim(0) == Input(1).dim(0) && Input(1).count(1) == 1)
            BroadcastRunWithType<float>(2);
        else if (Input(0).dim(-1) == Input(1).dim(-1) &&
            Input(1).count(0, Input(1).axis(-1)) == 1)
            BroadcastRunWithType<float>(1);
        else if (Input(1).ndim() == 1 && Input(1).dim(0) == 1)
            BroadcastRunWithType<float>(0);
        else LOG(FATAL) << "Could not be broadcast together with shapes "
                        << Input(-1).dim_string() << "  " << Input(0).dim_string();
    } else if (XIsType(Input(0), float16)) {
        if (Input(0).dims() == Input(1).dims())
            EltwiseRunWithType<float16>();
        else if (Input(0).dim(0) == Input(1).dim(0) && Input(1).count(1) == 1)
            BroadcastRunWithType<float16>(2);
        else if (Input(0).dim(-1) == Input(1).dim(-1) &&
            Input(1).count(0, Input(1).axis(-1)) == 1)
            BroadcastRunWithType<float16>(1);
        else if (Input(1).ndim() == 1 && Input(1).dim(0) == 1)
            BroadcastRunWithType<float16>(0);
        else LOG(FATAL) << "Could not be broadcast together with shapes "
                        << Input(-1).dim_string() << "  " << Input(0).dim_string();
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(DivGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(DivGradient);
#endif
OPERATOR_SCHEMA(DivGradient).NumInputs(3).NumOutputs(2);

class GetDivGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetDivGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0), GI(1)});
    }
};
REGISTER_GRADIENT(Div, GetDivGradient);

}    // namespace dragon