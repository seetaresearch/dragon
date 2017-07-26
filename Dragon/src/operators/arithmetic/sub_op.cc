#include "operators/arithmetic/sub_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h" 

namespace dragon {

template <class Context> template <typename T>
void SubOp<Context>::EltwiseRunWithType() {
    auto* X1data = input(0).template data<T, Context>();
    auto* X2data = input(1).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    math::Sub<T, Context>(input(0).count(), X1data, X2data, Ydata);
}

template <class Context> template <typename T>
void SubOp<Context>::BroadcastRunWithType(int type) {
    TIndex outer_dim, inner_dim;
    auto* X1data = input(0).template data<T, Context>();
    auto* X2data = input(1).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    ctx().template Copy<T, Context, Context>(input(0).count(), Ydata, X1data);

    if (type == 0 || type == 1) {
        if (type == 0) {
            outer_dim = input(0).count();
            inner_dim = 1;
        } else {
            outer_dim = input(0).count(0, input(0).axis(-1));
            inner_dim = input(0).dim(-1);
        }
        INIT_MULTIPLIER(bcast_multiplier, outer_dim);
        auto* BMul_data = bcast_multiplier->template data<T, Context>();
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, outer_dim, inner_dim, 1,
            -1.0, bcast_multiplier->template data<T, Context>(), X2data, 1.0, Ydata);
    } 
    else if (type == 2) {
        outer_dim = input(0).dim(0);
        inner_dim = input(0).count(1);
        INIT_MULTIPLIER(bcast_multiplier, inner_dim);
        auto* BMul_data = bcast_multiplier->template data<T, Context>();
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, outer_dim, inner_dim, 1,
            -1.0, X2data, bcast_multiplier->template data<T, Context>(), 1.0, Ydata);
    }
}

template <class Context>
void SubOp<Context>::RunOnDevice(){
    output(0)->ReshapeLike(input(0));

    if (input(0).dims() == input(1).dims()) {
        if (input(0).template IsType<float>()) EltwiseRunWithType<float>();
        else LOG(FATAL) << "unsupported input types.";
    }
    else if (input(0).dim(0) == input(1).dim(0) && input(1).count(1) == 1) {
        if (input(0).template IsType<float>()) BroadcastRunWithType<float>(2);
        else if (input(0).template IsType<float16>()) BroadcastRunWithType<float16>(2);
        else LOG(FATAL) << "unsupported input types.";
    }
    else if (input(0).dim(-1) == input(1).dim(-1) && 
             input(1).count(0, input(1).axis(-1)) == 1) {
        if (input(0).template IsType<float>()) BroadcastRunWithType<float>(1);
        else if (input(0).template IsType<float16>()) BroadcastRunWithType<float16>(1);
        else LOG(FATAL) << "unsupported input types.";
    } 
    else if (input(1).ndim() == 1 && input(1).dim(0) == 1) {
        if (input(0).template IsType<float>()) BroadcastRunWithType<float>(0);
        else if (input(0).template IsType<float16>()) BroadcastRunWithType<float16>(0);
        else LOG(FATAL) << "unsupported input types.";
    } 
    else {
        LOG(FATAL) << "could not be broadcast together with shapes "
                   << input(0).dim_string() << "  " << input(1).dim_string();
    }
}

DEPLOY_CPU(Sub);
#ifdef WITH_CUDA
DEPLOY_CUDA(Sub);
#endif
OPERATOR_SCHEMA(Sub).NumInputs(2).NumOutputs(1).Inplace({ { 0, 0 }, { 1, 0 } });

template <class Context> template <typename T>
void SubGradientOp<Context>::EltwiseRunWithType() {
    auto* dYdata = input(-1).template data<T, Context>();
    if (output(1)->name() != "ignore") {
        auto* dX2data = output(1)->template mutable_data<T, Context>();
        math::Scale<T, Context>(output(1)->count(), -1.0, dYdata, dX2data);
    }
    if (output(0)->name() != "ignore") {
        auto* dX1data = output(0)->template mutable_data<T, Context>();
        ctx().template Copy<T, Context, Context>(output(0)->count(), dX1data, dYdata);
    }
}

template <class Context> template <typename T>
void SubGradientOp<Context>::BroadcastRunWithType(int type) {
    TIndex outer_dim, inner_dim;
    auto* dYdata = input(-1).template data<T, Context>();

    if (output(1)->name() != "ignore") {
        auto* dX2data = output(1)->template mutable_data<T, Context>();
        if (type == 0 || type == 1) {
            if (type == 0) {
                outer_dim = input(-1).count();
                inner_dim = 1;
            } else {
                outer_dim = input(-1).count(0, input(-1).axis(-1));
                inner_dim = input(-1).dim(-1);
            }
            INIT_MULTIPLIER(bcast_multiplier, outer_dim);
            auto* BMul_data = bcast_multiplier->template data<T, Context>();
            math::Gemv<T, Context>(CblasTrans, outer_dim, inner_dim,
                                   -1.0, dYdata, BMul_data, 0.0, dX2data);
        }
        else if (type == 2) {
            outer_dim = input(-1).dim(0);
            inner_dim = input(-1).count(1);
            INIT_MULTIPLIER(bcast_multiplier, inner_dim);
            auto* BMul_data = bcast_multiplier->template data<T, Context>();
            math::Gemv<T, Context>(CblasNoTrans, outer_dim, inner_dim,
                                   -1.0, dYdata, BMul_data, 0.0, dX2data);
        }
    }

    if (output(0)->name() != "ignore") {
        auto* dX1data = output(0)->template mutable_data<T, Context>();
        ctx().template Copy<T, Context, Context>(output(0)->count(), dX1data, dYdata);
    }
}

template <class Context>
void SubGradientOp<Context>::RunOnDevice() {
    output(0)->ReshapeLike(input(-1));
    output(1)->ReshapeLike(input(0));

    if (input(-1).dims() == input(0).dims()) {
        if (input(0).template IsType<float>()) EltwiseRunWithType<float>();
        else LOG(FATAL) << "unsupported input types.";
    } 
    else if (input(-1).dim(0) == input(0).dim(0) && input(0).count(1) == 1) {
        if (input(0).template IsType<float>()) BroadcastRunWithType<float>(2);
        else if (input(0).template IsType<float16>()) BroadcastRunWithType<float16>(2);
        else LOG(FATAL) << "unsupported input types.";
    }
    else if (input(-1).dim(-1) == input(0).dim(-1) && 
             input(0).count(0, input(0).axis(-1)) == 1) {
        if (input(0).template IsType<float>()) BroadcastRunWithType<float>(1);
        else if (input(0).template IsType<float16>()) BroadcastRunWithType<float16>(1);
        else LOG(FATAL) << "unsupported input types.";
    } 
    else if (input(0).ndim() == 1 && input(0).dim(0) == 1) {
        if (input(0).template IsType<float>()) BroadcastRunWithType<float>(0);
        else if (input(0).template IsType<float16>()) BroadcastRunWithType<float16>(0);
        else LOG(FATAL) << "unsupported input types.";
    }
    else {
        LOG(FATAL) << "could not be broadcast together with shapes "
                   << input(-1).dim_string() << "  " << input(0).dim_string();
    }
}

template <class Context>
void SubGradientOp<Context>::ShareBeforeRun() {
    for (int i = 0; i < OutputSize(); i++) {
        if (output(i)->name() != "ignore") {
            Tensor* dX = ws()->GetBuffer();
            if (dX != nullptr) output(i)->Replace(*dX);
            break;
        }
    }
}

template <class Context>
void SubGradientOp<Context>::ClearAfterRun() {
    Tensor* dY = &input(-1);
    ws()->ReleaseBuffer(dY);
}

DEPLOY_CPU(SubGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SubGradient);
#endif
OPERATOR_SCHEMA(SubGradient).NumInputs(3).NumOutputs(2);

class GetSubGradient : public GradientMakerBase {
public:
    GRADIENT_MAKER_CTOR(GetSubGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(1), GO(0)},
            vector<string> {GI(0), GI(1)});
    }
};
REGISTER_GRADIENT(Sub, GetSubGradient);

}    // namespace dragon