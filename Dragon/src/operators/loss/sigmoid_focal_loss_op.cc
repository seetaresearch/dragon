#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/loss/sigmoid_focal_loss_op.h"

namespace dragon {

template <class Context> template <typename T>
void SigmoidFocalLossOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Tdata = Input(1).template data<T, Context>();
    auto* Ldata = losses.template mutable_data<T, Context>();
    auto* Fdata = flags.template mutable_data<T, Context>();

    kernel::SigmoidFocalLoss<T, Context>(
        outer_dim, axis_dim, inner_dim,
            pos_alpha, neg_alpha, gamma, neg_id,
                Xdata, Tdata, Ldata, Fdata, ctx());

    if (normalization == "UNIT") {
        vector<TIndex> output_dims = Input(0).dims();
        output_dims.erase(output_dims.begin() + axis);
        Output(0)->Reshape(output_dims);
        Output(0)->template CopyFrom<Context>(losses, ctx());
        return;
    }

    T normalizer = 1;
    if (normalization == "VALID") {
        normalizer = std::max(
            math::ASum<T, Context>(
                flags.count(), Fdata), 1.f);
    } else if (normalization == "BATCH_SIZE") {
        normalizer = (float)Input(0).dim(0);
    } else if (normalization == "FULL") {
        normalizer = (float)(outer_dim * inner_dim);
    }

    T loss = math::ASum<T, Context>(losses.count(), Ldata);
    Output(0)->Reshape({ 1 });
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    math::Set<T, Context>(1, loss / normalizer, Ydata, ctx());
}

template <class Context>
void SigmoidFocalLossOp<Context>::RunOnDevice() {
    ctx()->set_stream_id(0);  // Enforce SyncStream

    outer_dim = Input(0).count(0, axis);
    axis_dim = Input(0).dim(axis);
    inner_dim = Input(0).count(axis + 1);
    CHECK_EQ(outer_dim * inner_dim, Input(1).count())
        << "\nNumber of predictions must match the number of labels.";

    losses.ReshapeLike(Input(0));
    flags.ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(SigmoidFocalLoss);
#ifdef WITH_CUDA
DEPLOY_CUDA(SigmoidFocalLoss);
#endif
OPERATOR_SCHEMA(SigmoidFocalLoss).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void SigmoidFocalLossGradientOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Tdata = Input(1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* Fdata = flags.template mutable_data<T, Context>();

    kernel::SigmoidFocalLossGrad<T, Context>(
        outer_dim, axis_dim, inner_dim,
            pos_alpha, neg_alpha, gamma, neg_id,
                Xdata, Tdata, dXdata, Fdata, ctx());

    if (normalization == "UNIT") {
        auto* dYdata = Input(-1).template data<T, Context>();
        math::Mul<T, Context>(Output(0)->count(),
            dYdata, dXdata, dXdata, ctx()); return;
    }

    T normalizer = 1;
    if (normalization == "VALID") {
        normalizer = std::max(
            math::ASum<T, Context>(
                flags.count(), Fdata), 1.f);
    } else if (normalization == "BATCH_SIZE") {
        normalizer = Input(0).dim(0);
    } else if (normalization == "FULL") {
        normalizer = Input(0).count();
    }

    auto* dYdata = Input(-1).template data<T, Context>();
    T dYdata_host; ctx()->template Copy
        <T, CPUContext, Context>(
            1, &dYdata_host, dYdata);
    math::Scal<T, Context>(Output(0)->count(),
        dYdata_host / normalizer, dXdata, ctx());
}

template <class Context>
void SigmoidFocalLossGradientOp<Context>::RunOnDevice() {
    ctx()->set_stream_id(0);  // Enforce SyncStream

    outer_dim = Input(0).count(0, axis);
    axis_dim = Input(0).dim(axis);
    inner_dim = Input(0).count(axis + 1);

    Output(0)->ReshapeLike(Input(0));
    flags.ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(SigmoidFocalLossGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SigmoidFocalLossGradient);
#endif
OPERATOR_SCHEMA(SigmoidFocalLossGradient).NumInputs(3).NumOutputs(1);

class GetSigmoidFocalLossGradient
    final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetSigmoidFocalLossGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(
    SigmoidFocalLoss,
    GetSigmoidFocalLossGradient
);

}  // namespace dragon