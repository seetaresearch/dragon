#include "operators/loss/smooth_l1_loss_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void SmoothL1LossOp<Context>::RunWithType() {
    auto* X0data = input(0).template data<T, Context>();
    auto* X1data = input(1).template data<T, Context>();
    auto* diff_data = diff->template mutable_data<T, Context>();
    auto* error_data = error->template mutable_data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, CPUContext>();

    math::Sub<T, Context>(diff->count(), X0data, X1data, diff_data);
    if (InputSize() > 2){
        auto* inside_w_data = input(2).template data<T, Context>();
        math::Mul<T, Context>(diff->count(), inside_w_data, diff_data, diff_data);
    }
    kernel::SmoothL1<T, Context>(diff->count(), sigma2, diff_data, error_data);
    if (InputSize() > 3){
        auto* outside_w_data = input(3).template data<T, Context>();
        math::Mul<T, Context>(diff->count(), outside_w_data, error_data, error_data);
    }

    T loss = math::ASum<T, Context>(error->count(), error_data);
    Ydata[0] = loss / input(0).dim(0);
}

template <class Context>
void SmoothL1LossOp<Context>::RunOnDevice() {
    CHECK(input(0).dims() == input(1).dims());
    if (InputSize() > 2) CHECK(input(0).dims() == input(2).dims());
    if (InputSize() > 3) CHECK(input(0).dims() == input(3).dims());
    output(0)->Reshape(vector<TIndex>(1, 1));

    diff = ws()->CreateTensor("_t_" + anchor() + "_smoothl1_loss_diff");
    error = ws()->CreateTensor("_t_smoothl1_loss_error");
    diff->ReshapeLike(input(0));
    error->ReshapeLike(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(SmoothL1Loss);
#ifdef WITH_CUDA
DEPLOY_CUDA(SmoothL1Loss);
#endif
OPERATOR_SCHEMA(SmoothL1Loss).NumInputs(2, 4).NumOutputs(1);

template <class Context> template <typename T>
void SmoothL1LossGradientOp<Context>::RunWithType() {
    auto* dYdata = diff->template mutable_data<T, Context>();

    kernel::SmoothL1Grad<T, Context>(diff->count(), sigma2, dYdata, dYdata);

    for (int i = 0; i < 2; i++) {
        if (output(i)->name() == "ignore") continue;
        output(i)->ReshapeLike(input(i));
        auto* dXdata = output(i)->template mutable_data<T, Context>();
        const T sign = (i == 0) ? 1 : -1;
        const T coeff = sign / input(i).dim(0);
        math::Axpby<T, Context>(output(i)->count(), coeff, dYdata, 0, dXdata);
        if (InputSize() > 3) {
            auto* inside_w_data = input(2).template data<T, Context>();
            math::Mul<T, Context>(output(i)->count(), inside_w_data, dXdata, dXdata);
        }
        if (InputSize() > 4) {
            auto* outside_w_data = input(3).template data<T, Context>();
            math::Mul<T, Context>(output(i)->count(), outside_w_data, dXdata, dXdata);
        }
    }
}

template <class Context>
void SmoothL1LossGradientOp<Context>::RunOnDevice() {
    diff = ws()->GetTensor("_t_" + anchor() + "_smoothl1_loss_diff");
    
    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(SmoothL1LossGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SmoothL1LossGradient);
#endif
OPERATOR_SCHEMA(SmoothL1LossGradient).NumInputs(3, 5).NumOutputs(2);

class GetSmoothL1LossGradient final : public GradientMakerBase {
public:
    GRADIENT_MAKER_CTOR(GetSmoothL1LossGradient);
    vector<OperatorDef> MakeDefs() override {
        vector<string> inputs;
        for (auto input : def.input()) inputs.push_back(input);
        inputs.push_back(GO(0));
        return SingleDef(def.type() + "Gradient", "", 
                                              inputs, 
                      vector<string> {GI(0), GI(1)});
    }
};
REGISTER_GRADIENT(SmoothL1Loss, GetSmoothL1LossGradient);

}    // namespace dragon