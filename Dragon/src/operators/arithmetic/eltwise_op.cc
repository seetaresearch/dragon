#include "utils/math_functions.h"
#include "operators/arithmetic/eltwise_op.h"

namespace dragon {

template <class Context> template <typename T>
void EltwiseOp<Context>::SumRunWithType() {
    TIndex count = Output(0)->count();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    math::Set<T, Context>(count,
        dragon_cast<T, float>(0), Ydata, ctx());
    for (int i = 0; i < InputSize(); ++i) {
        math::Axpy<T, Context>(count, coeffs[i],
            Input(i).template data<T, Context>(), Ydata, ctx());
    }
}

template <class Context> template <typename T>
void EltwiseOp<Context>::ProdRunWithType() {
    TIndex count = Output(0)->count();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    math::Mul<T, Context>(count,
        Input(0).template data<T, Context>(),
            Input(1).template data<T, Context>(),
                Ydata, ctx());
    for (int i = 2; i < InputSize(); i++) {
        math::Mul<T, Context>(count,
            Ydata,
                Input(i).template data<T, Context>(),
                    Ydata, ctx());
    }
}

template <class Context>
void EltwiseOp<Context>::RunOnDevice() {
    for (int i = 1; i < InputSize(); i++) {
        CHECK(Input(i).dims() == Input(0).dims())
            << "\nExcepted Input(" << i << ")'s dims as "
            << Input(0).DimString() << ",\n but got "
            << Input(1).DimString() << ".";
    }

    Output(0)->ReshapeLike(Input(0));

    if (operation == "SUM") {
        if (XIsType(Input(0), float)) SumRunWithType<float>();
        else if (XIsType(Input(0), float16)) SumRunWithType<float16>();
        else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
    }
    else if (operation == "PROD") {
        if (XIsType(Input(0), float)) ProdRunWithType<float>();
        else if (XIsType(Input(0), float16)) ProdRunWithType<float16>();
        else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
    }
    else {
        LOG(FATAL) << "Unknwon operation: " << operation;
    }
}

DEPLOY_CPU(Eltwise);
#ifdef WITH_CUDA
DEPLOY_CUDA(Eltwise);
#endif
OPERATOR_SCHEMA(Eltwise).NumInputs(2, INT_MAX).NumOutputs(1);

template <class Context> template <typename T>
void EltwiseGradientOp<Context>::SumRunWithType() {
    auto* dYdata = Input(-1).template data<T, Context>();
    TIndex count = Input(-1).count();

    for (int i = 0; i < OutputSize(); i++) {
        if (Output(i)->name() == "ignore") continue;
        auto* dXdata = Output(i)->template mutable_data<T, Context>();
        if (coeffs[i] == 1.f) {
            ctx()->template Copy<T, Context, Context>(
                count, dXdata, dYdata);
        } else {
            math::Scale<T, Context>(count,
                coeffs[i], dYdata, dXdata, ctx());
        }
    }
}

template <class Context> template <typename T>
void EltwiseGradientOp<Context>::ProdRunWithType() {
    auto* dYdata = Input(-1).template data<T, Context>();
    TIndex count = Input(-1).count();

    for (int i = 0; i < OutputSize(); i++) {
        if (Output(i)->name() == "ignore") continue;
        auto* dXdata = Output(i)->template mutable_data<T, Context>();
        bool initialized = false;
        for (int j = 0; j < OutputSize(); j++) {
            if (i == j) continue;
            auto* Xdata = Input(j).template data<T, Context>();
            if (!initialized) {
                ctx()->template Copy<T, Context, Context>(count, dXdata, Xdata);
                initialized = true;
            } else math::Mul<T, Context>(count, Xdata, dXdata, dXdata, ctx());
        }
        math::Mul<T, Context>(count, dYdata, dXdata, dXdata, ctx());
    }
}

template <class Context>
void EltwiseGradientOp<Context>::RunOnDevice() {
    for (int i = 0; i < OutputSize(); i++)
        Output(i)->ReshapeLike(Input(i));

    if (operation == "SUM") {
        if (XIsType(Input(0), float)) SumRunWithType<float>();
        else if (XIsType(Input(0), float16)) SumRunWithType<float16>();
        else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
    }
    else if (operation == "PROD") {
        if (XIsType(Input(0), float)) ProdRunWithType<float>();
        else if (XIsType(Input(0), float16)) ProdRunWithType<float16>();
        else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
    }
    else {
        LOG(FATAL) << "Unknwon operation: " << operation;
    }
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

}  // namespace dragon