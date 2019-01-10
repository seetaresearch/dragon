#include "core/workspace.h"
#include "utils/math_functions.h"
#include "operators/arithmetic/pow_op.h"

namespace dragon {

template <class Context> template <typename T>
void PowOp<Context>::RunWithType() {
    int64_t count = Input(0).count();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    if (power_scale == 0.f) {
        // Y is a constant Vector
        float value = (power == 0.f) ? 1.f : pow(shift, power);
        math::Set(count, cast::to<T>(value), Ydata, ctx()); return;
    }

    // Compute Y = (Ax + b) ** p
    auto* Xdata = Input(0).template data<T, Context>();
    math::Scale(count, scale, Xdata, Ydata, ctx());
    math::AddScalar(count, shift, Ydata, ctx());
    if (power != 1.f) math::Pow(count, power, Ydata, Ydata, ctx());
}

template <class Context>
void PowOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float16)) RunWithType<float16>();
    else if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), double)) RunWithType<double>();
    else LOG(FATAL) << DTypeHelper(Input(0), {
        "float16", "float32", "float64" });
}

DEPLOY_CPU(Pow);
#ifdef WITH_CUDA
DEPLOY_CUDA(Pow);
#endif
OPERATOR_SCHEMA(Pow).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void PowGradientOp<Context>::RunWithType() {
    int64_t count = Input(0).count();
    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();

    if (power_scale == 0.f || power == 1.f) {
        const T value = cast::to<T>(power_scale);
        math::Set(count, value, dXdata, ctx());
    } else {
        auto* Xdata = Input(0).template data<T, Context>();
        if (power == 2.f) {
            math::Axpby(count,
                power_scale * scale, Xdata,
                    0.f, dXdata, ctx());
            if (shift != 0.f) 
                math::AddScalar(count, power_scale * shift, dXdata, ctx());
        } else if (shift == 0.f) {
            auto* Ydata = Input(1).template data<T, Context>();
            math::Div(count, Ydata, Xdata, dXdata, ctx());
            math::Scale(count, power, dXdata, dXdata, ctx());
        } else {
            auto* Ydata = Input(1).template data<T, Context>();
            math::Scale(count, scale, Xdata, dXdata, ctx());
            math::AddScalar(count, shift, dXdata, ctx());
            math::Div(count, Ydata, dXdata, dXdata, ctx());
            math::Scale(count, power_scale, dXdata, dXdata, ctx());
        }
    }
    math::Mul(count, dYdata, dXdata, dXdata, ctx());
}

template <class Context>
void PowGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float16)) RunWithType<float16>();
    else if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), double)) RunWithType<double>();
    else LOG(FATAL) << DTypeHelper(Input(0), {
        "float16", "float32", "float64" });
}

DEPLOY_CPU(PowGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(PowGradient);
#endif

OPERATOR_SCHEMA(PowGradient)
    .NumInputs(3).NumOutputs(1);

class GetPowGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetPowGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), O(0), GO(0)}),
            vector<string>({ GI(0) }));
    }
};

REGISTER_GRADIENT(Pow, GetPowGradient);

}  // namespace dragon