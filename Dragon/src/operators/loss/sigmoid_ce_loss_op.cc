#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/loss/sigmoid_ce_loss_op.h"

namespace dragon {

template <class Context> template <typename T>
void SigmoidCrossEntropyOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Tdata = Input(1).template data<T, Context>();
    auto* Ldata = losses.template mutable_data<T, Context>();
    auto* Fdata = flags.template mutable_data<int, Context>();

    kernel::SigmoidCrossEntropy(Input(0).count(),
        Xdata, Tdata, Ldata, Fdata, ctx());

    if (normalization == "UNIT") {
        Output(0)->ReshapeLike(losses);
        Output(0)->template CopyFrom<Context>(
            losses, ctx()); return;
    }

    double normalizer = 1.;
    if (normalization == "VALID") {
        normalizer = std::max(
            math::Sum(flags.count(),
                1.f, Fdata, ctx()), 1);
    } else if (normalization == "BATCH_SIZE") {
        normalizer = Input(0).dim(0);
    } else if (normalization == "FULL") {
        normalizer = Input(0).count();
    }

    Output(0)->Reshape(vector<int64_t>());
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    math::Sum(losses.count(), 1. / normalizer, Ldata, Ydata, ctx());
}

template <class Context>
void SigmoidCrossEntropyOp<Context>::RunOnDevice() {
    CHECK_EQ(Input(0).count(), Input(1).count())
        << "\nNumber of predictions must match the number of labels.";
    losses.ReshapeLike(Input(0));
    flags.ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
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
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* Fdata = flags.template mutable_data<int, Context>();

    kernel::SigmoidCrossEntropyGrad(
        Input(0).count(), Xdata, Tdata, dXdata, Fdata, ctx());

    if (normalization == "UNIT") {
        auto* dYdata = Input(-1).template data<T, Context>();
        math::Mul(Output(0)->count(),
            dYdata, dXdata, dXdata, ctx()); return;
    }

    double normalizer = 1.;
    if (normalization == "VALID") {
        normalizer = std::max(
            math::Sum(flags.count(),
                1.f, Fdata, ctx()), 1);
    } else if (normalization == "BATCH_SIZE") {
        normalizer = Input(0).dim(0);
    } else if (normalization == "FULL") {
        normalizer = Input(0).count();
    }

    auto* dYdata = Input(-1).template data<T, Context>();
    T dYHost; ctx()->template Copy
        <T, CPUContext, Context>(
            1, &dYHost, dYdata);
    ctx()->FinishDeviceCompution();

    math::Scale(
        Output(0)->count(),
            dYHost / normalizer,
                dXdata, dXdata, ctx());
}

template <class Context>
void SigmoidCrossEntropyGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));
    flags.ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(SigmoidCrossEntropyGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SigmoidCrossEntropyGradient);
#endif

OPERATOR_SCHEMA(SigmoidCrossEntropyGradient)
    .NumInputs(3).NumOutputs(1);

class GetSigmoidCrossEntropyGradient
    final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetSigmoidCrossEntropyGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), GO(0) }),
            vector<string>({ GI(0) }));
    }
};

REGISTER_GRADIENT(
    SigmoidCrossEntropy,
    GetSigmoidCrossEntropyGradient
);

}  // namespace dragon