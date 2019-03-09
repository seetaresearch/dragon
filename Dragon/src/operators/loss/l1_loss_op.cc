#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/loss/l1_loss_op.h"

namespace dragon {

template <class Context> template <typename T>
void L1LossOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    auto* Ddata = diff->template mutable_data<T, Context>();

    if (InputSize() > 1) {
        // Compute Diff = X1 - X2
        auto* Tdata = Input(1).template data<T, Context>();
        math::Sub(Input(0).count(), Xdata, Tdata, Ddata, ctx());
    } else {
        // Let Diff = X1
        ctx()->template Copy<T, Context, Context>(
            Input(0).count(), Ddata, Xdata);
    }

    if (InputSize() > 2) {
        // Compute Diff *= Mask
        auto* mask = Input(2).template data<T, Context>();
        math::Mul(Input(0).count(), mask, Ddata, Ddata, ctx());
    }

    double normalizer = 1. / scale;
    if (normalization == "BATCH_SIZE") {
        normalizer *= Input(0).dim(0);
    } else if (normalization == "FULL") {
        normalizer *= Input(0).count();
    }

    T loss = math::ASum(Input(0).count(), Ddata, ctx());
    math::Set(1, T(loss / normalizer), Ydata, ctx());
}

template <class Context>
void L1LossOp<Context>::RunOnDevice() {
    ctx()->set_stream_id(0);  // Enforce SyncStream

    for (int i = 1; i < InputSize(); i++) {
        CHECK_EQ(Input(0).count(), Input(i).count())
            << "\nTensor(" << Input(i).name() << ") takes the "
            << "dimensions of " << Input(i).DimString() << ", "
            << "while " << Input(0).DimString() << " is required.";
    }

    Output(0)->Reshape(vector<int64_t>());
    diff = ws()->CreateTensor("/mnt/" + anchor() +
        "/l1_loss/diff")->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(L1Loss);
#ifdef WITH_CUDA
DEPLOY_CUDA(L1Loss);
#endif
OPERATOR_SCHEMA(L1Loss).NumInputs(1, 3).NumOutputs(1);

template <class Context> template <typename T>
void L1LossGradientOp<Context>::RunWithType() {
    auto* dYdata = Input(-1).template data<T, Context>();

    T dYHost; ctx()->template Copy
        <T, CPUContext, Context>(
            1, &dYHost, dYdata);
    ctx()->FinishDeviceCompution();

    auto* Ddata = diff->template mutable_data<T, Context>();
    kernel::AbsGrad(diff->count(), Ddata, Ddata, ctx());

    if (normalization == "BATCH_SIZE") {
        dYHost /= (Input(0).dim(0) / scale);
    } else if (normalization == "FULL") {
        dYHost /= (Input(0).count() / scale);
    } else { dYHost *= scale; }

    for (int i = 0; i < 2; i++) {
        if (Output(i)->name() == "ignore") continue;
        Output(i)->ReshapeLike(Input(i));
        auto* dXdata = Output(i)->template mutable_data<T, Context>();
        math::Scale(Output(i)->count(),
            dYHost * (i == 0 ? 1.f : -1.f),
                Ddata, dXdata, ctx());
        if (Input(2).name() != "ignore") {
            auto* mask = Input(2).template data<T, Context>();
            math::Mul(Output(i)->count(), mask, dXdata, dXdata, ctx());
        }
    }
}

template <class Context>
void L1LossGradientOp<Context>::RunOnDevice() {
    diff = ws()->GetTensor(mount_name("l1_loss/diff"));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(L1LossGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(L1LossGradient);
#endif

OPERATOR_SCHEMA(L1LossGradient)
    .NumInputs(4).NumOutputs(2);

class GetL1LossGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetL1LossGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), I(2), GO(0) }),
            vector<string>({ GI(0), GI(1) }));
    }
};

REGISTER_GRADIENT(L1Loss, GetL1LossGradient);

}  // namespace dragon