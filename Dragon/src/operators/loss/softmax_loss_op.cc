#include "operators/activation/softmax_op.h"
#include "operators/loss/softmax_loss_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"
#include "utils/proto_utils.h"

namespace dragon {

template <class Context> template <typename T>
void SoftmaxLossOp<Context>::RunWithType() {
    auto* prob_data = prob->template data<T, Context>();
    auto* label_data = input(1).template data<T, Context>();
    auto* loss_data = losses.template mutable_data<T, Context>();
    auto* valid_data = valid.template mutable_data<T, Context>();
    kernel::SparseSoftmaxCrossEntropy<T, Context>(input(0).count(), input(0).dim(axis),
        outer_dim, inner_dim, prob_data, label_data, loss_data, valid_data, &ignore);

    if (normalization == "UNIT") {
        output(0)->ReshapeLike(losses);
        output(0)->Share(losses);
        return;
    }

    T normalizer;
    if (normalization == "VALID")
        normalizer = math::ASum<T, Context>(valid.count(), valid_data);
    else if (normalization == "BATCH_SIZE") normalizer = outer_dim;
    else if (normalization == "FULL") normalizer = outer_dim * inner_dim;
    else if (normalization == "NONE") normalizer = 1;
    T loss = math::ASum<T, Context>(losses.count(), loss_data);
    output(0)->Reshape(vector<TIndex>(1, 1));
    auto* Ydata = output(0)->template mutable_data<T, CPUContext>();
    Ydata[0] = loss / normalizer;
}

template <class Context>
void SoftmaxLossOp<Context>::RunOnDevice() {
    outer_dim = input(0).count(0, axis);
    inner_dim = input(0).count(axis + 1);
    CHECK_EQ(outer_dim * inner_dim, input(1).count())
        << "\nnumber of predictions must match the number of labels.";
    valid.Reshape(vector<TIndex>(1, outer_dim * inner_dim));
    losses.Reshape(vector<TIndex>(1, outer_dim * inner_dim));
    softmax_op->Run();
    prob = ws()->GetTensor("_t_" + anchor() + "_softmax_prob");
    
    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(SoftmaxLoss);
#ifdef WITH_CUDA
DEPLOY_CUDA(SoftmaxLoss);
#endif
OPERATOR_SCHEMA(SoftmaxLoss).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void SoftmaxLossGradientOp<Context>::RunWithType() {
    auto* label_data = input(1).template data<T, Context>();
    auto* prob_data = prob->template mutable_data<T, Context>();
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    auto* valid_data = valid.template mutable_data<T, Context>();
    ctx().template Copy<T, Context, Context>(prob->count(), dXdata, prob_data);
    kernel::SoftmaxLossGrad<T, Context>(output(0)->count(), output(0)->dim(axis),
        outer_dim, inner_dim, label_data, prob_data, valid_data, &ignore, dXdata);

    if (normalization == "UNIT") {
        auto* dYdata = input(-1).template data<T, Context>();
        kernel::SumGrad<T, Context>(input(0).count() / input(0).dim(axis),
            input(0).dim(axis), inner_dim, 1.0, dYdata, prob_data);
        math::Mul<T, Context>(output(0)->count(), prob_data, dXdata, dXdata);
        return;
    }

    T normalizer;
    if (normalization == "VALID") normalizer = math::ASum<T, Context>(valid.count(), valid_data);
    else if (normalization == "BATCH_SIZE") normalizer = outer_dim;
    else if (normalization == "FULL") normalizer = outer_dim * inner_dim;
    else if (normalization == "NONE") normalizer = 1;
    auto* dYdata = input(-1).template data<T, CPUContext>();
    math::Scal<T, Context>(output(0)->count(), dYdata[0] / normalizer, dXdata);
}

template <class Context>
void SoftmaxLossGradientOp<Context>::RunOnDevice() {
    prob = ws()->GetTensor("_t_" + anchor() + "_softmax_prob");
    outer_dim = prob->count(0, axis);
    inner_dim = prob->count(axis + 1);
    output(0)->ReshapeLike(input(0));
    valid.Reshape(vector<TIndex>(1, outer_dim * inner_dim));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(SoftmaxLossGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SoftmaxLossGradient);
#endif
OPERATOR_SCHEMA(SoftmaxLossGradient).NumInputs(3).NumOutputs(1);

class GetSoftmaxLossGradient final : public GradientMakerBase {
public:
    GRADIENT_MAKER_CTOR(GetSoftmaxLossGradient);
    vector<OperatorDef> MakeDefs() override{
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(SoftmaxLoss, GetSoftmaxLossGradient);

}    // namespace dragon