#include "operators/loss/sigmoid_cross_entropy_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void SigmoidCrossEntropyOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Tdata = Input(1).template data<T, Context>();
    auto* Ldata = losses.template mutable_data<T, Context>();
    auto* Vdata = valid.template mutable_data<T, Context>();

    kernel::SigmoidCrossEntropy<T, Context>(Input(0).count(),
                                                       Xdata,
                                                       Tdata,
                                                       Ldata,
                                                       Vdata);

    if (normalization == "UNIT") {
        Output(0)->ReshapeLike(losses);
        Output(0)->Share(losses);
        return;
    }

    T normalizer;
    if (normalization == "VALID")
        normalizer = math::ASum<T, Context>(valid.count(), Vdata);
    else if (normalization == "BATCH_SIZE") normalizer = Input(0).dim(0);
    else if (normalization == "FULL") normalizer = Input(0).count();
    else if (normalization == "NONE") normalizer = 1;
    T loss = math::ASum<T, Context>(losses.count(), Ldata);
    Output(0)->Reshape(vector<TIndex>(1, 1));
    auto* Ydata = Output(0)->template mutable_data<T, CPUContext>();
    Ydata[0] = loss / normalizer;
}

template <class Context>
void SigmoidCrossEntropyOp<Context>::RunOnDevice() {
    CHECK_EQ(Input(0).count(), Input(1).count())
        << "\nNumber of predictions must match the number of labels.";
    losses.ReshapeLike(Input(0));
    valid.ReshapeLike(Input(0));

    if (Input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(SigmoidCrossEntropy);
#ifdef WITH_CUDA
DEPLOY_CUDA(SigmoidCrossEntropy);
#endif
OPERATOR_SCHEMA(SigmoidCrossEntropy).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void SigmoidCrossEntropyGradientOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Tdata = Input(1).template data<T, Context>();
    auto* Vdata = valid.template mutable_data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();

    kernel::SigmoidCrossEntropyGrad<T, Context>(Input(0).count(),
                                                           Xdata,
                                                           Tdata,
                                                          dXdata,
                                                           Vdata);

    if (normalization == "UNIT") {
        auto* dYdata = Input(-1).template data<T, Context>();
        math::Mul<T, Context>(Output(0)->count(), dYdata, dXdata, dXdata);
        return;
    }

    T normalizer;
    if (normalization == "VALID") normalizer = math::ASum<T, Context>(valid.count(), Vdata);
    else if (normalization == "BATCH_SIZE") normalizer = Input(0).dim(0);
    else if (normalization == "FULL") normalizer = Input(0).count();
    else if (normalization == "NONE") normalizer = 1;
    auto* dYdata = Input(-1).template data<T, CPUContext>();
    math::Scal<T, Context>(Output(0)->count(), dYdata[0] / normalizer, dXdata);
}

template <class Context>
void SigmoidCrossEntropyGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));
    valid.ReshapeLike(Input(0));

    if (Input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(SigmoidCrossEntropyGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SigmoidCrossEntropyGradient);
#endif
OPERATOR_SCHEMA(SigmoidCrossEntropyGradient).NumInputs(3).NumOutputs(1);

class GetSigmoidCrossEntropyGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetSigmoidCrossEntropyGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(SigmoidCrossEntropy, GetSigmoidCrossEntropyGradient);

}    // namespace dragon