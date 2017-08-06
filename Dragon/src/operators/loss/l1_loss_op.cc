#include "operators/loss/l1_loss_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void L1LossOp<Context>::RunWithType() {
    auto* X0data = input(0).template data<T, Context>();
    auto* X1data = input(1).template data<T, Context>();
    auto* diff_data = diff->template mutable_data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, CPUContext>();

    math::Sub<T, Context>(input(0).count(), X0data, X1data, diff_data);
    if (InputSize() > 2) {
        CHECK_EQ(input(0).count(), input(2).count());
        auto* Wdata = input(2).template data<T, Context>();
        math::Mul<T, Context>(diff->count(), Wdata, diff_data, diff_data);
    }
    T abs_val = math::ASum<T, Context>(diff->count(), diff_data);
    Ydata[0] = coeff * abs_val;
    T normalizer;
    if (normalization == "BATCH_SIZE") normalizer = input(0).dim(0);
    else if (normalization == "FULL") normalizer = input(0).count();
    else if (normalization == "NONE") normalizer = 1;
    Ydata[0] = Ydata[0] / normalizer;
}

template <class Context>
void L1LossOp<Context>::RunOnDevice() {
    CHECK_EQ(input(0).count(), input(1).count());
    output(0)->Reshape(vector<TIndex>(1, 1));
    diff = ws()->CreateTensor("_t_" + anchor() + "_l1_loss_diff");
    diff->ReshapeLike(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(L1Loss);
#ifdef WITH_CUDA
DEPLOY_CUDA(L1Loss);
#endif
OPERATOR_SCHEMA(L1Loss).NumInputs(2, 3).NumOutputs(1);

template <class Context> template <typename T>
void L1LossGradientOp<Context>::RunWithType() {
    auto* dYdata = diff->template mutable_data<T, Context>();
    kernel::AbsGrad<T, Context>(diff->count(), dYdata, dYdata);
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
void L1LossGradientOp<Context>::RunOnDevice() {
    diff = ws()->GetTensor("_t_" + anchor() + "_l1_loss_diff");

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(L1LossGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(L1LossGradient);
#endif
OPERATOR_SCHEMA(L1LossGradient).NumInputs(3).NumOutputs(2);

class GetL1LossGradient final : public GradientMakerBase {
public:
    GRADIENT_MAKER_CTOR(GetL1LossGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0), GI(1)});
    }
};
REGISTER_GRADIENT(L1Loss, GetL1LossGradient);

}    // namespace dragon


