#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "utils/proto_utils.h"
#include "operators/loss/softmax_focal_loss_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGUMENTS(X) \
    axis = OperatorBase::Arg<int64_t>("axis", 1); \
    axis = axis < 0 ? axis + X.ndim() : axis; \
    CHECK(axis >= 0 && axis < X.ndim()) \
       << "\nExcepted the axis in [-" << X.ndim() << ", " << X.ndim() \
       << "), got " << OperatorBase::Arg<int64_t>("axis", 1) << ".";

template <class Context> template <typename Tx, typename Ty>
void SoftmaxFocalLossOp<Context>::RunWithType() {
    auto* Pdata = this->prob->template data<Tx, Context>();
    auto* Tdata = Input(1).template data<Ty, Context>();
    auto* Idata = !this->ignores.count() ? nullptr :
        this->ignores.template data<int, Context>();
    auto* Ldata = losses.template mutable_data<Tx, Context>();
    auto* Fdata = flags.template mutable_data<int, Context>();

    kernel::SoftmaxFocalLoss(
        outer_dim, Input(0).dim(axis), inner_dim, this->ignores.count(),
            pos_alpha, neg_alpha, gamma, neg_id,
                Pdata, Tdata, Idata, Ldata, Fdata, ctx());

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
    } else if (normalization == "FULL"){
        normalizer = outer_dim * inner_dim;
    }

    Output(0)->Reshape(vector<int64_t>());
    auto* Ydata = Output(0)->template mutable_data<Tx, Context>();
    math::Sum(losses.count(), 1. / normalizer, Ldata, Ydata, ctx());
}

template <class Context>
void SoftmaxFocalLossOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGUMENTS(Input(0));

    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
    CHECK_EQ(outer_dim * inner_dim, Input(1).count())
        << "\nNumber of predictions must match the number of labels.";
    flags.Reshape({ outer_dim * inner_dim });
    losses.Reshape({ outer_dim * inner_dim });

    this->prob = ws()->CreateTensor(
        mount_name("softmax/prob"));
    this->SoftmaxRun();

    if (XIsType(Input(0), float)) {
        if (XIsType(Input(1), float)) RunWithType<float, float>();
        else if (XIsType(Input(1), int64_t)) RunWithType<float, int64_t>();
        else LOG(FATAL) << DTypeHelper(Input(1), { "float32", "int64" });
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(SoftmaxFocalLoss);
#ifdef WITH_CUDA
DEPLOY_CUDA(SoftmaxFocalLoss);
#endif
OPERATOR_SCHEMA(SoftmaxFocalLoss).NumInputs(2).NumOutputs(1);

template <class Context> template <typename Tx, typename Ty>
void SoftmaxFocalLossGradientOp<Context>::RunWithType() {
    auto* Pdata = this->prob->template mutable_data<Tx, Context>();
    auto* Tdata = Input(1).template data<Ty, Context>();
    auto* Idata = !this->ignores.count() ? nullptr :
        this->ignores.template data<int, Context>();
    auto* dXdata = Output(0)->template mutable_data<Tx, Context>();
    auto* Fdata = flags.template mutable_data<int, Context>();

    kernel::SoftmaxFocalLossGrad(
        outer_dim, Output(0)->dim(axis), inner_dim, this->ignores.count(),
            pos_alpha, neg_alpha, gamma, neg_id,
                Pdata, Tdata, Idata, dXdata, Fdata, ctx());

    if (normalization == "UNIT") {
        auto* dYdata = Input(-1).template data<Tx, Context>();
        kernel::Repeat(outer_dim, 1, inner_dim,
            Input(0).dim(axis), dYdata, Pdata, ctx());
        math::Mul(Output(0)->count(),
            Pdata, dXdata, dXdata, ctx()); return;
    }

    double normalizer = 1.;
    if (normalization == "VALID") {
        normalizer = std::max(
            math::Sum(flags.count(),
                1.f, Fdata, ctx()), 1);
    } else if (normalization == "BATCH_SIZE") {
        normalizer = Input(0).dim(0);
    } else if (normalization == "FULL") {
        normalizer = outer_dim * inner_dim;
    }

    auto* dYdata = Input(-1).template data<Tx, Context>();
    Tx dYHost; ctx()->template Copy
        <Tx, CPUContext, Context>(
            1, &dYHost, dYdata);
    ctx()->FinishDeviceCompution();

    math::Scale(
        Output(0)->count(),
            dYHost / normalizer,
                dXdata, dXdata, ctx());
}

template <class Context>
void SoftmaxFocalLossGradientOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGUMENTS(Input(0));

    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
    Output(0)->ReshapeLike(Input(0));
    flags.Reshape({ outer_dim * inner_dim });

    this->prob = ws()->GetTensor(
        mount_name("softmax/prob"));

    if (XIsType(Input(0), float)) {
        if (XIsType(Input(1), float)) RunWithType<float, float>();
        else if (XIsType(Input(1), int64_t)) RunWithType<float, int64_t>();
        else LOG(FATAL) << DTypeHelper(Input(1), { "float32", "int64" });
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(SoftmaxFocalLossGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SoftmaxFocalLossGradient);
#endif

OPERATOR_SCHEMA(SoftmaxFocalLossGradient)
    .NumInputs(3).NumOutputs(1);

class GetSoftmaxFocalLossGradient
    final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetSoftmaxFocalLossGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), GO(0) }),
            vector<string>({ GI(0) }));
    }
};

REGISTER_GRADIENT(
    SoftmaxFocalLoss,
    GetSoftmaxFocalLossGradient
);

#undef DETERMINE_RUNTIME_ARGUMENTS

}  // namespace dragon