#include "operators/loss/l2_loss_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"

namespace dragon {

template <class Context> template <typename T>
void L2LossOp<Context>::RunWithType() {
    auto* X0data = input(0).template data<T, Context>();
    auto* X1data = input(1).template data<T, Context>();
    auto* diff_data = diff->template mutable_data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, CPUContext>();
    math::Sub<T, Context>(diff->count(), X0data, X1data, diff_data);
    if (InputSize() > 2) {
        CHECK_EQ(input(0).count(), input(2).count());
        auto* Wdata = input(2).template data<T, Context>();
        math::Mul<T, Context>(diff->count(), Wdata, diff_data, diff_data);
    }
    T dot = math::Dot<T, Context>(diff->count(), diff_data, diff_data);
    Ydata[0] = T(0.5) * coeff * dot;
    T normalizer;
    if (normalization == "BATCH_SIZE") normalizer = input(0).dim(0);
    else if (normalization == "FULL") normalizer = input(0).count();
    else if (normalization == "NONE") normalizer = 1;
    Ydata[0] = Ydata[0] / normalizer;
}

template <class Context>
void L2LossOp<Context>::RunOnDevice() {
    CHECK_EQ(input(0).count(), input(1).count());
    output(0)->Reshape(vector<TIndex>(1, 1));
    diff = ws()->CreateTensor("_t_" + anchor() + "_l2_loss_diff");
    diff->ReshapeLike(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(L2Loss);
#ifdef WITH_CUDA
DEPLOY_CUDA(L2Loss);
#endif
OPERATOR_SCHEMA(L2Loss).NumInputs(2, 3).NumOutputs(1);

template <class Context> template <typename T>
void L2LossGradientOp<Context>::RunWithType() {
    auto* dYdata = diff->template mutable_data<T, Context>();
    T alpha = coeff, normalizer;
    if (normalization == "BATCH_SIZE") normalizer = input(0).dim(0);
    else if (normalization == "FULL") normalizer = input(0).count();
    else if (normalization == "NONE") normalizer = 1;
    alpha = alpha / normalizer;
    for (int i = 0; i < 2; i++) {
        if (output(i)->name() == "ignore") continue;
        output(i)->ReshapeLike(input(i));
        auto* dXdata = output(i)->template mutable_data<T, Context>();
        const T sign = (i == 0) ? 1 : -1;
        alpha *= sign;
        math::Axpby<T, Context>(output(i)->count(), alpha, dYdata, 0, dXdata);
    }
}

template <class Context>
void L2LossGradientOp<Context>::RunOnDevice() {
    diff = ws()->GetTensor("_t_" + anchor() + "_l2_loss_diff");

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(L2LossGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(L2LossGradient);
#endif
OPERATOR_SCHEMA(L2LossGradient).NumInputs(3).NumOutputs(2);

class GetL2LossGradient final : public GradientMakerBase {
public:
    GRADIENT_MAKER_CTOR(GetL2LossGradient);
    vector<OperatorDef> MakeDefs() override{
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0), GI(1)});
    }
};
REGISTER_GRADIENT(L2Loss, GetL2LossGradient);

}    // namespace dragon


