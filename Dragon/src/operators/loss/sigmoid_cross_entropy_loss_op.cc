#include "operators/loss/sigmoid_cross_entropy_loss_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void SigmoidCrossEntropyLossOp<Context>::RunWithType() {
    auto* Xdata = input(0).template data<T, Context>();
    auto* Pdata = prob->template mutable_data<T, Context>();
    kernel::Sigmoid<T, Context>(prob->count(), Xdata, Pdata);

    auto* Tdata = input(1).template data<T, Context>();
    auto* Ldata = losses.template mutable_data<T, Context>();
    kernel::SigmoidCrossEntropy<T, Context>(input(0).count(), Xdata, Tdata, Ldata);

    if (normalization == "UNIT") {
        output(0)->ReshapeLike(losses);
        output(0)->Share(losses);
        return;
    }

    T normalizer;
    if (normalization == "BATCH_SIZE") normalizer = input(0).dim(0);
    else if (normalization == "FULL") normalizer = input(0).count();
    else if (normalization == "NONE") normalizer = 1;
    T loss = math::ASum<T, Context>(losses.count(), Ldata);
    output(0)->Reshape(vector<TIndex>(1, 1));
    auto* Ydata = output(0)->template mutable_data<T, CPUContext>();
    Ydata[0] = loss / normalizer;
}

template <class Context>
void SigmoidCrossEntropyLossOp<Context>::RunOnDevice() {
    CHECK_EQ(input(0).count(), input(1).count())
        << "\nnumber of predictions must match the number of labels.";
    prob = ws()->CreateTensor("_t_" + anchor() + "_sigmoid_prob");
    prob->ReshapeLike(input(0));
    losses.ReshapeLike(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(SigmoidCrossEntropyLoss);
#ifdef WITH_CUDA
DEPLOY_CUDA(SigmoidCrossEntropyLoss);
#endif
OPERATOR_SCHEMA(SigmoidCrossEntropyLoss).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void SigmoidCrossEntropyLossGradientOp<Context>::RunWithType() {
    auto* Pdata = prob->template data<T, Context>();
    auto* Tdata = input(1).template data<T, Context>();
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    ctx().template Copy<T, Context, Context>(prob->count(), dXdata, Pdata);
    math::Axpy<T, Context>(output(0)->count(), -1.0, Tdata, dXdata);

    if (normalization == "UNIT") {
        auto* dYdata = input(-1).template data<T, Context>();
        math::Mul<T, Context>(output(0)->count(), dYdata, dXdata, dXdata);
        return;
    }

    T normalizer;
    if (normalization == "BATCH_SIZE") normalizer = input(0).dim(0);
    else if (normalization == "FULL") normalizer = input(0).count();
    else if (normalization == "NONE") normalizer = 1;
    auto* dYdata = input(-1).template data<T, CPUContext>();
    math::Scal<T, Context>(output(0)->count(), dYdata[0] / normalizer, dXdata);
}

template <class Context>
void SigmoidCrossEntropyLossGradientOp<Context>::RunOnDevice() {
    prob = ws()->GetTensor("_t_" + anchor() + "_sigmoid_prob");
    output(0)->ReshapeLike(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(SigmoidCrossEntropyLossGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SigmoidCrossEntropyLossGradient);
#endif
OPERATOR_SCHEMA(SigmoidCrossEntropyLossGradient).NumInputs(3).NumOutputs(1);

class GetSigmoidCrossEntropyLossGradient final : public GradientMakerBase {
public:
    GRADIENT_MAKER_CTOR(GetSigmoidCrossEntropyLossGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(SigmoidCrossEntropyLoss, GetSigmoidCrossEntropyLossGradient);

}    // namespace dragon