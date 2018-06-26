#include "operators/loss/l1_loss_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void L1LossOp<Context>::RunWithType() {
    auto* X0data = Input(0).template data<T, Context>();
    auto* X1data = Input(1).template data<T, Context>();
    auto* diff_data = diff->template mutable_data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    math::Sub<T, Context>(Input(0).count(), X0data, X1data, diff_data);
    if (InputSize() > 2) {
        CHECK_EQ(Input(0).count(), Input(2).count());
        auto* Wdata = Input(2).template data<T, Context>();
        math::Mul<T, Context>(diff->count(), Wdata, diff_data, diff_data);
    }

    T normalizer;
    if (normalization == "BATCH_SIZE") normalizer = Input(0).dim(0);
    else if (normalization == "FULL") normalizer = Input(0).count();
    else if (normalization == "NONE") normalizer = 1;
    T loss = math::ASum<T, Context>(diff->count(), diff_data);
    math::Set<T, Context>(1, loss / normalizer, Ydata);
}

template <class Context>
void L1LossOp<Context>::RunOnDevice() {
    CHECK_EQ(Input(0).count(), Input(1).count());
    Output(0)->Reshape(vector<TIndex>(1, 1));
    diff = ws()->CreateTensor("/mnt/" + anchor() + "/l1_loss/diff");
    diff->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(L1Loss);
#ifdef WITH_CUDA
DEPLOY_CUDA(L1Loss);
#endif
OPERATOR_SCHEMA(L1Loss).NumInputs(2, 3).NumOutputs(1);

template <class Context> template <typename T>
void L1LossGradientOp<Context>::RunWithType() {
    auto* diff_data = diff->template mutable_data<T, Context>();
    auto* dYdata = Input(-1).template data<T, Context>();
    T dYdata_host; Context::template Copy<T, CPUContext, Context>(
        1, &dYdata_host, dYdata);
    kernel::AbsGrad<T, Context>(diff->count(), diff_data, diff_data);

    T alpha = dYdata_host, normalizer;
    if (normalization == "BATCH_SIZE") normalizer = Input(0).dim(0);
    else if (normalization == "FULL") normalizer = Input(0).count();
    else if (normalization == "NONE") normalizer = 1;
    alpha = alpha / normalizer;
    for (int i = 0; i < 2; i++) {
        if (Output(i)->name() == "ignore") continue;
        Output(i)->ReshapeLike(Input(i));
        auto* dXdata = Output(i)->template mutable_data<T, Context>();
        const T sign = (i == 0) ? 1 : -1;
        alpha *= sign;
        math::Axpby<T, Context>(Output(i)->count(),
            alpha, diff_data, 0, dXdata);
    }
}

template <class Context>
void L1LossGradientOp<Context>::RunOnDevice() {
    diff = ws()->GetTensor("/mnt/" + anchor() + "/l1_loss/diff");

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
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