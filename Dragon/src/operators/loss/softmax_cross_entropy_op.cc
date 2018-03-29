#include "operators/activation/softmax_op.h"
#include "operators/loss/softmax_cross_entropy_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"
#include "utils/proto_utils.h"

namespace dragon {

template <class Context> template <typename T>
void SoftmaxCrossEntropyOp<Context>::RunWithType() {
    auto* Pdata = prob->template data<T, Context>();
    auto* Tdata = Input(1).template data<T, Context>();
    auto* Ldata = losses.template mutable_data<T, Context>();
    kernel::SoftmaxCrossEntropy<T, Context>(Input(0).count(), Pdata, Tdata, Ldata);

    if (normalization == "UNIT") {
        Output(0)->Reshape(vector<TIndex>(1, outer_dim * inner_dim));
        auto* Ydata = Output(0)->template mutable_data<T, Context>();
        kernel::Sum<T, Context>(outer_dim * inner_dim,
                                   Input(0).dim(axis),
                                            inner_dim,
                                                Ldata,
                                               Ydata);
        return;
    }

    T normalizer;
    if (normalization == "BATCH_SIZE") normalizer = Input(0).dim(0);
    else if (normalization == "FULL") normalizer = outer_dim * inner_dim;
    else if (normalization == "NONE") normalizer = 1;
    T loss = math::ASum<T, Context>(losses.count(), Ldata);
    Output(0)->Reshape(vector<TIndex>(1, 1));
    auto* Ydata = Output(0)->template mutable_data<T, CPUContext>();
    Ydata[0] = loss / normalizer;
}

template <class Context>
void SoftmaxCrossEntropyOp<Context>::RunOnDevice() {
    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
    CHECK_EQ(Input(0).count(), Input(1).count())
        << "\nNumber of predictions must match the number of labels.";
    losses.ReshapeLike(Input(0));
    softmax_op->Run();
    prob = ws()->GetTensor("/mnt/" + anchor() + "/softmax_prob");

    if (Input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(SoftmaxCrossEntropy);
#ifdef WITH_CUDA
DEPLOY_CUDA(SoftmaxCrossEntropy);
#endif
OPERATOR_SCHEMA(SoftmaxCrossEntropy).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void SoftmaxCrossEntropyGradientOp<Context>::RunWithType() {
    auto* Tdata = Input(1).template data<T, Context>();
    auto* Pdata = prob->template mutable_data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    ctx().template Copy<T, Context, Context>(prob->count(), dXdata, Pdata);
    math::Axpy<T, Context>(Output(0)->count(), -1.0, Tdata, dXdata);

    if (normalization == "UNIT") {
        auto* dYdata = Input(-1).template data<T, Context>();
        kernel::SumGrad<T, Context>(outer_dim * inner_dim,
                                       Input(0).dim(axis),
                                                inner_dim,
                                                      1.0,
                                                   dYdata,
                                                   Pdata);
        math::Mul<T, Context>(Output(0)->count(), Pdata, dXdata, dXdata);
        return;
    }

    T normalizer;
    if (normalization == "BATCH_SIZE") normalizer = Input(0).dim(0);
    else if (normalization == "FULL") normalizer = outer_dim * inner_dim;
    else if (normalization == "NONE") normalizer = 1;
    auto* dYdata = Input(-1).template data<T, CPUContext>();
    math::Scal<T, Context>(Output(0)->count(), dYdata[0] / normalizer, dXdata);
}

template <class Context>
void SoftmaxCrossEntropyGradientOp<Context>::RunOnDevice() {
    prob = ws()->GetTensor("/mnt/" + anchor() + "/softmax_prob");
    outer_dim = prob->count(0, axis);
    inner_dim = prob->count(axis + 1);
    Output(0)->ReshapeLike(Input(0));

    if (Input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(SoftmaxCrossEntropyGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SoftmaxCrossEntropyGradient);
#endif
OPERATOR_SCHEMA(SoftmaxCrossEntropyGradient).NumInputs(3).NumOutputs(1);

class GetSoftmaxCrossEntropyGradient final : public GradientMakerBase { 
 public:
    GRADIENT_MAKER_CTOR(GetSoftmaxCrossEntropyGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(SoftmaxCrossEntropy, GetSoftmaxCrossEntropyGradient);

}    // namespace dragon