#include "operators/arithmetic/pow_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"

namespace dragon {

template <class Context> template <typename T>
void PowOp<Context>::RunWithType() {
    TIndex count = Input(0).count();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    if (power_scale == float(0)) {
        float value = (power == float(0)) ? float(1) : pow(shift, power);
        math::Set<T, Context>(count, dragon_cast<T, float>(value), Ydata);
        return;
    }
    auto* Xdata = Input(0).template data<T, Context>();
    ctx().template Copy<T, Context, Context>(count, Ydata, Xdata);
    if (scale != float(1)) math::Scal<T, Context>(count, scale, Ydata);
    if (shift != float(0)) math::AddScalar<T, Context>(count, shift, Ydata);
    if (power != float(1)) math::Pow<T, Context>(count, power, Ydata, Ydata);
}

template <class Context>
void PowOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));
    
    if (Input(0).template IsType<float>()) RunWithType<float>();
#ifdef WITH_CUDA_FP16
    else if (Input(0).template IsType<float16>()) RunWithType<float16>();
#endif
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(Pow);
#ifdef WITH_CUDA
DEPLOY_CUDA(Pow);
#endif
OPERATOR_SCHEMA(Pow).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void PowGradientOp<Context>::RunWithType() {
    TIndex count = Input(0).count();
    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();

    if (power_scale == float(0) || power == float(1)) {
        const T value = dragon_cast<T, float>(power_scale);
        math::Set<T, Context>(count, value, dXdata);
    } else {
        auto* Xdata = Input(0).template data<T, Context>();
        if (power == float(2)) {
            math::Axpby<T, Context>(count, power_scale * scale, Xdata, 0, dXdata);
            if (shift != float(0)) 
                math::AddScalar<T, Context>(count, power_scale * shift, dXdata);
        } else if (shift == float(0)) {
            auto* Ydata = Input(1).template data<T, Context>();
            math::Div<T, Context>(count, Ydata, Xdata, dXdata);
            math::Scal<T, Context>(count, power, dXdata);
        } else {
            auto* Ydata = Input(1).template data<T, Context>();
            ctx().template Copy<T, Context, Context>(count, dXdata, Xdata);
            if (scale != float(1))
                math::Scal<T, Context>(count, scale, dXdata);
            if (shift != float(0))
                math::AddScalar<T, Context>(count, shift, dXdata);
            math::Div<T, Context>(count, Ydata, dXdata, dXdata);
            if (power_scale != float(1))
                math::Scal<T, Context>(count, power_scale, dXdata);
        }
    }
    if (power_scale != float(0))
        math::Mul<T, Context>(count, dYdata, dXdata, dXdata);
}

template <class Context>
void PowGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (Input(0).template IsType<float>()) RunWithType<float>();
#ifdef WITH_CUDA_FP16
    else if (Input(0).template IsType<float16>()) RunWithType<float16>();
#endif
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(PowGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(PowGradient);
#endif
OPERATOR_SCHEMA(PowGradient).NumInputs(3).NumOutputs(1);

class GetPowGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetPowGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), O(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(Pow, GetPowGradient);

}    // namespace dragon