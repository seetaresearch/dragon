#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "utils/proto_utils.h"
#include "operators/activation/softmax_op.h"
#include "operators/loss/softmax_focal_loss_op.h"

namespace dragon {

template <class Context> template <typename T>
void SoftmaxFocalLossOp<Context>::RunWithType() {
    auto* Pdata = this->prob->template data<T, Context>();
    auto* Tdata = Input(1).template data<T, Context>();
    auto* Idata = !this->ignores.count() ? nullptr :
        this->ignores.template data<int, Context>();
    auto* Ldata = losses.template mutable_data<T, Context>();
    auto* Fdata = flags.template mutable_data<T, Context>();

    kernel::SoftmaxFocalLoss<T, Context>(
        outer_dim, Input(0).dim(axis), inner_dim,
            pos_alpha, neg_alpha, gamma, neg_id,
                Pdata, Tdata, Idata, this->ignores.count(),
                    Ldata, Fdata, &ctx());

    if (normalization == "UNIT") {
        Output(0)->ReshapeLike(losses);
        Output(0)->template Copy<Context, Context>(losses);
        return;
    }

    T normalizer = 1;
    if (normalization == "VALID") {
        normalizer = std::max(
            math::ASum<T, Context>(
                flags.count(), Fdata), 1.f);
    } else if (normalization == "BATCH_SIZE") {
        normalizer = Input(0).dim(0);
    } else if (normalization == "FULL"){
        normalizer = outer_dim * inner_dim;
    }

    T loss = math::ASum<T, Context>(losses.count(), Ldata);
    Output(0)->Reshape({ 1 });
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    math::Set<T, Context>(1, loss / normalizer, Ydata);
}

template <class Context>
void SoftmaxFocalLossOp<Context>::RunOnDevice() {
    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
    CHECK_EQ(outer_dim * inner_dim, Input(1).count())
        << "\nNumber of predictions must match the number of labels.";
    flags.Reshape({ outer_dim * inner_dim });
    losses.Reshape({ outer_dim * inner_dim });
    ws()->CreateTensor("/mnt/" + anchor() + "/softmax/prob");
    this->SoftmaxRun();
    this->prob = ws()->GetTensor("/mnt/" + anchor() + "/softmax/prob");

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(SoftmaxFocalLoss);
#ifdef WITH_CUDA
DEPLOY_CUDA(SoftmaxFocalLoss);
#endif
OPERATOR_SCHEMA(SoftmaxFocalLoss).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void SoftmaxFocalLossGradientOp<Context>::RunWithType() {
    auto* Pdata = this->prob->template mutable_data<T, Context>();
    auto* Tdata = Input(1).template data<T, Context>();
    auto* Idata = !this->ignores.count() ? nullptr :
        this->ignores.template data<int, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* Fdata = flags.template mutable_data<T, Context>();

    kernel::SoftmaxFocalLossGrad<T, Context>(
        outer_dim, Output(0)->dim(axis), inner_dim,
            pos_alpha, neg_alpha, gamma, neg_id,
                Pdata, Tdata, Idata, this->ignores.count(),
                    dXdata, Fdata, &ctx());

    if (normalization == "UNIT") {
        auto* dYdata = Input(-1).template data<T, Context>();
        kernel::SumGrad<T, Context>(
            Input(0).count() / Input(0).dim(axis),
                Input(0).dim(axis), inner_dim,
                    1.0, dYdata, Pdata);
        math::Mul<T, Context>(Output(0)->count(),
            Pdata, dXdata, dXdata); return;
    }

    T normalizer = 1;
    if (normalization == "VALID") {
        normalizer = std::max(
            math::ASum<T, Context>(
                flags.count(), Fdata), 1.f);
    } else if (normalization == "BATCH_SIZE") {
        normalizer = Input(0).dim(0);
    } else if (normalization == "FULL") {
        normalizer = outer_dim * inner_dim;
    }

    auto* dYdata = Input(-1).template data<T, Context>();
    T dYdata_host; ctx().template Copy<T, CPUContext, Context>(
        1, &dYdata_host, dYdata);
    math::Scal<T, Context>(Output(0)->count(),
        dYdata_host / normalizer, dXdata, &ctx());
}

template <class Context>
void SoftmaxFocalLossGradientOp<Context>::RunOnDevice() {
    this->prob = ws()->GetTensor("/mnt/" + anchor() + "/softmax/prob");
    outer_dim = this->prob->count(0, axis);
    inner_dim = this->prob->count(axis + 1);
    Output(0)->ReshapeLike(Input(0));
    flags.Reshape({ outer_dim * inner_dim });

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(SoftmaxFocalLossGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SoftmaxFocalLossGradient);
#endif
OPERATOR_SCHEMA(SoftmaxFocalLossGradient).NumInputs(3).NumOutputs(1);

class GetSoftmaxFocalLossGradient
    final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetSoftmaxFocalLossGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(
    SoftmaxFocalLoss,
    GetSoftmaxFocalLossGradient
);

}    // namespace dragon