#include "core/workspace.h"
#include "utils/math_functions.h"
#include "operators/loss/l2_loss_op.h"

namespace dragon {

template <class Context> template <typename T>
void L2LossOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<float, Context>();

    const T* Tdata = nullptr; T* Ddata = nullptr;

    if (InputSize() > 1) {
        // Compute Diff = X1 - X2
        diff->ReshapeLike(Input(0));
        Tdata = Input(1).template data<T, Context>();
        Ddata = diff->template mutable_data<T, Context>();
        math::Sub(Input(0).count(), Xdata, Tdata, Ddata, ctx());
    } else {
        // Let Diff = X1
        Ddata = const_cast<T*>(Xdata);
    }

    if (InputSize() > 2) {
        // Compute Diff *= Mask
        auto* mask = Input(2).template data<T, Context>();
        math::Mul(Input(0).count(), mask, Ddata, Ddata, ctx());
    }

    double normalizer = 2. / scale;
    if (normalization == "BATCH_SIZE") {
        normalizer *= Input(0).dim(0);
    } else if (normalization == "FULL") {
        normalizer *= Input(0).count();
    }

    T lossT;
    math::Dot(Input(0).count(), Ddata, Ddata, &lossT, ctx());
    math::Set(1, cast::to<float>(
        cast::to<float>(lossT) /
            normalizer), Ydata, ctx());
}

template <class Context>
void L2LossOp<Context>::RunOnDevice() {
    for (int i = 1; i < InputSize(); i++) {
        CHECK_EQ(Input(0).count(), Input(i).count())
            << "\nTensor(" << Input(i).name() << ") takes the "
            << "dimensions of " << Input(i).DimString() << ", "
            << "while " << Input(0).DimString() << " is required.";
    }

    Output(0)->Reshape(vector<int64_t>());
    diff = ws()->CreateTensor(mount_name("l2_loss/diff"));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(L2Loss);
#ifdef WITH_CUDA
DEPLOY_CUDA(L2Loss);
#endif
OPERATOR_SCHEMA(L2Loss).NumInputs(1, 3).NumOutputs(1);

template <class Context> template <typename T>
void L2LossGradientOp<Context>::RunWithType() {
    const T* Ddata = nullptr;

    if (diff->count() == Input(0).count()) {
        Ddata = diff->template data<T, Context>();
    } else {
        Ddata = Input(0).template data<T, Context>();
    }

    auto* dYdata = Input(-1).template data<float, Context>();
    float dYHost; ctx()->template Copy
        <float, CPUContext, Context>(
            1, &dYHost, dYdata);
    ctx()->FinishDeviceCompution();

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
void L2LossGradientOp<Context>::RunOnDevice() {
    diff = ws()->GetTensor(mount_name("l2_loss/diff"));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(L2LossGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(L2LossGradient);
#endif

OPERATOR_SCHEMA(L2LossGradient)
    .NumInputs(4).NumOutputs(2);

class GetL2LossGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetL2LossGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), I(2), GO(0) }),
            vector<string>({ GI(0), GI(1) }));
    }
};

REGISTER_GRADIENT(L2Loss, GetL2LossGradient);

}  // namespace dragon