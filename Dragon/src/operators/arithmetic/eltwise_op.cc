#include "operators/arithmetic/eltwise_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"

namespace dragon {

template <class Context> template <typename T>
void EltwiseOp<Context>::SumRunWithType() {
    TIndex count = output(0)->count();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    math::Set<T, Context>(count, dragon_cast<T, float>(0), Ydata);
    for (int i = 0; i < InputSize(); ++i) {
        math::Axpy<T, Context>(count,
                               coeffs[i], 
                               input(i).template data<T, Context>(), 
                               Ydata);
    }
}

template <class Context> template <typename T>
void EltwiseOp<Context>::ProdRunWithType() {
    TIndex count = output(0)->count();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    math::Mul<T, Context>(count,
                          input(0).template data<T, Context>(),
                          input(1).template data<T, Context>(), 
                          Ydata);
    for (int i = 2; i < InputSize(); i++) {
        math::Mul<T, Context>(count, 
                              Ydata, 
                              input(i).template data<T, Context>(), 
                              Ydata);
    }
}

template <class Context>
void EltwiseOp<Context>::RunOnDevice() {
    for (int i = 1; i < InputSize(); i++) 
        CHECK(input(i).dims() == input(0).dims());
    output(0)->ReshapeLike(input(0));

    if (operation == "SUM") {
        if (input(0).template IsType<float>()) SumRunWithType<float>();
        else if (input(0).template IsType<float16>()) SumRunWithType<float16>();
        else LOG(FATAL) << "unsupported input types.";
    } 
    else if (operation == "PROD") {
        if (input(0).template IsType<float>()) ProdRunWithType<float>();
        else if (input(0).template IsType<float16>()) ProdRunWithType<float16>();
        else LOG(FATAL) << "unsupported input types.";
    } 
    else {
        LOG(FATAL) << "unknwon operation: " << operation;
    }
}

DEPLOY_CPU(Eltwise);
#ifdef WITH_CUDA
DEPLOY_CUDA(Eltwise);
#endif
OPERATOR_SCHEMA(Eltwise).NumInputs(2, INT_MAX).NumOutputs(1);

template <class Context> template <typename T>
void EltwiseGradientOp<Context>::SumRunWithType() {
    auto* dYdata = input(-1).template data<T, Context>();
    TIndex count = input(-1).count();

    for (int i = 0; i < OutputSize(); i++){
        if (output(i)->name() == "ignore") continue;
        auto* dXdata = output(i)->template mutable_data<T, Context>();
        if (coeffs[i] == float(1)) {
            ctx().template Copy<T, Context, Context>(count, dXdata, dYdata);
        } else {
            math::Scale<T, Context>(count, coeffs[i], dYdata, dXdata);
        }
    }
}

template <class Context> template <typename T>
void EltwiseGradientOp<Context>::ProdRunWithType() {
    auto* dYdata = input(-1).template data<T, Context>();
    TIndex count = input(-1).count();

    for (int i = 0; i < OutputSize(); i++) {
        if (output(i)->name() == "ignore") continue;
        auto* dXdata = output(i)->template mutable_data<T, Context>();
        bool initialized = false;
        for (int j = 0; j < OutputSize(); j++) {
            if (i == j) continue;
            auto* Xdata = input(j).template data<T, Context>();
            if (!initialized) {
                ctx().template Copy<T, Context, Context>(count, dXdata, Xdata);
                initialized = true;
            } else math::Mul<T, Context>(count, Xdata, dXdata, dXdata);
        }
        math::Mul<T, Context>(count, dYdata, dXdata, dXdata);
    }
}

template <class Context>
void EltwiseGradientOp<Context>::RunOnDevice() {
    for (int i = 0; i < OutputSize(); i++)
        output(i)->ReshapeLike(input(i));

    if (operation == "SUM") {
        if (input(0).template IsType<float>()) SumRunWithType<float>();
        else if (input(0).template IsType<float16>()) SumRunWithType<float16>();
        else LOG(FATAL) << "unsupported input types.";
    } 
    else if (operation == "PROD") {
        if (input(0).template IsType<float>()) ProdRunWithType<float>();
        else if (input(0).template IsType<float16>()) ProdRunWithType<float16>();
        else LOG(FATAL) << "unsupported input types.";
    } 
    else {
        LOG(FATAL) << "unknwon operation: " << operation;
    }
}

template <class Context>
void EltwiseGradientOp<Context>::ShareBeforeRun() {
    for (int i = 0; i < OutputSize(); i++) {
        if (output(i)->name() != "ignore") {
            Tensor* dX = ws()->GetBuffer();
            if (dX != nullptr) output(i)->Replace(*dX);
            break;
        }
    }
}

template <class Context>
void EltwiseGradientOp<Context>::ClearAfterRun() {
    Tensor* dY = &input(-1);
    ws()->ReleaseBuffer(dY);
}

DEPLOY_CPU(EltwiseGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(EltwiseGradient);
#endif
OPERATOR_SCHEMA(EltwiseGradient).NumInputs(3, INT_MAX).NumOutputs(2, INT_MAX);

class GetEltwiseGradient final : public GradientMakerBase {
public:
    GRADIENT_MAKER_CTOR(GetEltwiseGradient);
    vector<OperatorDef> MakeDefs() override {
        vector<string> inputs, outputs;
        for (auto input : def.input()) inputs.push_back(input);
        for (int i = 0; i < def.input_size(); i++) outputs.push_back(GI(i));
        inputs.push_back(GO(0));
        return SingleDef(def.type() + "Gradient", "", inputs, outputs);
    }
};
REGISTER_GRADIENT(Eltwise, GetEltwiseGradient);

}    // namespace dragon