#include "core/workspace.h"
#include "utils/math_functions.h"
#include "operators/loss/l2_loss_op.h"

namespace dragon {

template <class Context> template <typename T>
void L2LossOp<Context>::RunWithType() {
    auto* X0data = Input(0).template data<T, Context>();
    auto* X1data = Input(1).template data<T, Context>();
    auto* diff_data = diff->template mutable_data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    math::Sub<T, Context>(diff->count(), X0data, X1data, diff_data);
    if (InputSize() > 2) {
        CHECK_EQ(Input(0).count(), Input(2).count());
        auto* Wdata = Input(2).template data<T, Context>();
        math::Mul<T, Context>(diff->count(), Wdata, diff_data, diff_data);
    }

    T normalizer = 1;
    if (normalization == "BATCH_SIZE") {
        normalizer = Input(0).dim(0);
    } else if (normalization == "FULL") {
        normalizer = Input(0).count();
    }

    T loss = T(0.5) * math::Dot<T, Context>(diff->count(),
        diff_data, diff_data, &ctx());
    math::Set<T, Context>(1, loss / normalizer, Ydata);
}

template <class Context>
void L2LossOp<Context>::RunOnDevice() {
    CHECK_EQ(Input(0).count(), Input(1).count());
    Output(0)->Reshape({ 1 });
    diff = ws()->CreateTensor("/mnt/" + anchor() + "/l2_loss/diff");
    diff->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(L2Loss);
#ifdef WITH_CUDA
DEPLOY_CUDA(L2Loss);
#endif
OPERATOR_SCHEMA(L2Loss).NumInputs(2, 3).NumOutputs(1);

template <class Context> template <typename T>
void L2LossGradientOp<Context>::RunWithType() {
    auto* diff_data = diff->template mutable_data<T, Context>();
    auto* dYdata = Input(-1).template data<T, Context>();
    T dYdata_host; ctx().template Copy<T, CPUContext, Context>(
        1, &dYdata_host, dYdata);

    T alpha = dYdata_host, normalizer = 1;
    if (normalization == "BATCH_SIZE") {
        normalizer = Input(0).dim(0);
    } else if (normalization == "FULL") {
        normalizer = Input(0).count();
    } alpha = alpha / normalizer;

    for (int i = 0; i < 2; i++) {
        if (Output(i)->name() == "ignore") continue;
        Output(i)->ReshapeLike(Input(i));
        auto* dXdata = Output(i)->template mutable_data<T, Context>();
        const T sign = (i == 0) ? 1 : -1;
        alpha *= sign;
        math::Axpby<T, Context>(Output(i)->count(),
            alpha, diff_data, 0, dXdata, &ctx());
    }
}

template <class Context>
void L2LossGradientOp<Context>::RunOnDevice() {
    diff = ws()->GetTensor("/mnt/" + anchor() + "/l2_loss/diff");

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(L2LossGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(L2LossGradient);
#endif
OPERATOR_SCHEMA(L2LossGradient).NumInputs(3).NumOutputs(2);

class GetL2LossGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetL2LossGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0), GI(1)});
    }
};
REGISTER_GRADIENT(L2Loss, GetL2LossGradient);

}    // namespace dragon