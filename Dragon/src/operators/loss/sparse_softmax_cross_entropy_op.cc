#include "operators/activation/softmax_op.h"
#include "operators/loss/sparse_softmax_cross_entropy_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"
#include "utils/proto_utils.h"

namespace dragon {

template <class Context> template <typename T>
void SparseSoftmaxCrossEntropyOp<Context>::RunWithType() {
    auto* prob_data = prob->template data<T, Context>();
    auto* label_data = Input(1).template data<T, Context>();
    auto* loss_data = losses.template mutable_data<T, Context>();
    auto* valid_data = valid.template mutable_data<T, Context>();

    kernel::SparseSoftmaxCrossEntropy<T, Context>(Input(0).count(),
                                                Input(0).dim(axis),
                                                         outer_dim,
                                                         inner_dim,
                                                         prob_data,
                                                        label_data,
                                                         loss_data,
                                                        valid_data,
                                                          &ignore);

    if (normalization == "UNIT") {
        Output(0)->ReshapeLike(losses);
        Output(0)->Share(losses);
        return;
    }

    T normalizer;
    if (normalization == "VALID")
        normalizer = math::ASum<T, Context>(valid.count(), valid_data);
    else if (normalization == "BATCH_SIZE") normalizer = Input(0).dim(0);
    else if (normalization == "FULL") normalizer = outer_dim * inner_dim;
    else if (normalization == "NONE") normalizer = 1;
    T loss = math::ASum<T, Context>(losses.count(), loss_data);
    Output(0)->Reshape(vector<TIndex>(1, 1));
    auto* Ydata = Output(0)->template mutable_data<T, CPUContext>();
    Ydata[0] = loss / normalizer;
}

template <class Context>
void SparseSoftmaxCrossEntropyOp<Context>::RunOnDevice() {
    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
    CHECK_EQ(outer_dim * inner_dim, Input(1).count())
        << "\nNumber of predictions must match the number of labels.";
    valid.Reshape(vector<TIndex>(1, outer_dim * inner_dim));
    losses.Reshape(vector<TIndex>(1, outer_dim * inner_dim));
    softmax_op->Run();
    prob = ws()->GetTensor("/mnt/" + Anchor() + "/softmax/prob");
    
    if (Input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(SparseSoftmaxCrossEntropy);
#ifdef WITH_CUDA
DEPLOY_CUDA(SparseSoftmaxCrossEntropy);
#endif
OPERATOR_SCHEMA(SparseSoftmaxCrossEntropy).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void SparseSoftmaxCrossEntropyGradientOp<Context>::RunWithType() {
    auto* label_data = Input(1).template data<T, Context>();
    auto* prob_data = prob->template mutable_data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* valid_data = valid.template mutable_data<T, Context>();
    ctx().template Copy<T, Context, Context>(prob->count(), dXdata, prob_data);

    kernel::SparseSoftmaxCrossEntropyGrad<T, Context>(Output(0)->count(),
                                                    Output(0)->dim(axis),
                                                               outer_dim,
                                                               inner_dim,
                                                               prob_data,
                                                              label_data,
                                                              valid_data,
                                                                 &ignore,
                                                                 dXdata);

    if (normalization == "UNIT") {
        auto* dYdata = Input(-1).template data<T, Context>();
        kernel::SumGrad<T, Context>(Input(0).count() / Input(0).dim(axis),
            Input(0).dim(axis), inner_dim, 1.0, dYdata, prob_data);
        math::Mul<T, Context>(Output(0)->count(), prob_data, dXdata, dXdata);
        return;
    }

    T normalizer;
    if (normalization == "VALID") normalizer = math::ASum<T, Context>(valid.count(), valid_data);
    else if (normalization == "BATCH_SIZE") normalizer = Input(0).dim(0);
    else if (normalization == "FULL") normalizer = outer_dim * inner_dim;
    else if (normalization == "NONE") normalizer = 1;
    auto* dYdata = Input(-1).template data<T, CPUContext>();
    math::Scal<T, Context>(Output(0)->count(), dYdata[0] / normalizer, dXdata);
}

template <class Context>
void SparseSoftmaxCrossEntropyGradientOp<Context>::RunOnDevice() {
    prob = ws()->GetTensor("/mnt/" + Anchor() + "/softmax/prob");
    outer_dim = prob->count(0, axis);
    inner_dim = prob->count(axis + 1);
    Output(0)->ReshapeLike(Input(0));
    valid.Reshape(vector<TIndex>(1, outer_dim * inner_dim));

    if (Input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(SparseSoftmaxCrossEntropyGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SparseSoftmaxCrossEntropyGradient);
#endif
OPERATOR_SCHEMA(SparseSoftmaxCrossEntropyGradient).NumInputs(3).NumOutputs(1);

class GetSparseSoftmaxCrossEntropyGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetSparseSoftmaxCrossEntropyGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(SparseSoftmaxCrossEntropy, GetSparseSoftmaxCrossEntropyGradient);

}    // namespace dragon