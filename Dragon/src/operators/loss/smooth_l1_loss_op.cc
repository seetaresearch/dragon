#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/loss/smooth_l1_loss_op.h"

namespace dragon {

template <class Context> template <typename T>
void SmoothL1LossOp<Context>::RunWithType() {
    auto* X0data = Input(0).template data<T, Context>();
    auto* X1data = Input(1).template data<T, Context>();
    auto* diff_data = diff->template mutable_data<T, Context>();
    auto* error_data = error->template mutable_data<T, Context>();

    math::Sub<T, Context>(diff->count(),
        X0data, X1data, diff_data, ctx());
    if (InputSize() > 2) {
        auto* inside_w_data = Input(2).template data<T, Context>();
        math::Mul<T, Context>(diff->count(),
            inside_w_data, diff_data, diff_data, ctx());
    }
    kernel::SmoothL1<T, Context>(diff->count(),
        beta, diff_data, error_data, ctx());
    if (InputSize() > 3) {
        auto* outside_w_data = Input(3).template data<T, Context>();
        math::Mul<T, Context>(diff->count(),
            outside_w_data, error_data, error_data, ctx());
    }

    T normalizer = 1;
    if (normalization == "BATCH_SIZE") {
        normalizer = Input(0).dim(0);
    } else if (normalization == "FULL") {
        normalizer = Input(0).count();
    }

    Output(0)->Reshape(vector<int64_t>());
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    math::Sum<T, Context>(error->count(),
        1.f / normalizer, error_data, Ydata, ctx());
}

template <class Context>
void SmoothL1LossOp<Context>::RunOnDevice() {
    CHECK(Input(0).count() == Input(1).count());
    if (InputSize() > 2) CHECK(Input(0).count() == Input(2).count());
    if (InputSize() > 3) CHECK(Input(0).count() == Input(3).count());

    diff = ws()->CreateTensor(mount_name(
        "smoothl1_loss/diff"))->ReshapeLike(Input(0));
    error = ws()->CreateTensor(
        "/share/smoothl1_loss_error")->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(SmoothL1Loss);
#ifdef WITH_CUDA
DEPLOY_CUDA(SmoothL1Loss);
#endif
OPERATOR_SCHEMA(SmoothL1Loss).NumInputs(2, 4).NumOutputs(1);

template <class Context> template <typename T>
void SmoothL1LossGradientOp<Context>::RunWithType() {
    auto* Ddata = diff->template mutable_data<T, Context>();
    auto* dYdata = Input(-1).template data<T, Context>();
    float dYdata_host; ctx()->template Copy
        <float, CPUContext, Context>(
            1, &dYdata_host, dYdata);
    ctx()->FinishDeviceCompution();

    kernel::SmoothL1Grad<T, Context>(diff->count(),
        beta, Ddata, Ddata, ctx());

    if (normalization == "BATCH_SIZE") {
        dYdata_host /= Input(0).dim(0);
    } else if (normalization == "FULL") {
        dYdata_host /= Input(0).count();
    }

    for (int i = 0; i < 2; i++) {
        if (Output(i)->name() == "ignore") continue;
        Output(i)->ReshapeLike(Input(i));
        auto* dXdata = Output(i)->template mutable_data<T, Context>();
        math::Scale<T, Context>(Output(i)->count(),
            dYdata_host * (i == 0 ? 1.f : -1.f),
                Ddata, dXdata, ctx());
        if (InputSize() > 3) {
            auto* inside_w_data = Input(2).template data<T, Context>();
            math::Mul<T, Context>(Output(i)->count(),
                inside_w_data, dXdata, dXdata, ctx());
        }
        if (InputSize() > 4) {
            auto* outside_w_data = Input(3).template data<T, Context>();
            math::Mul<T, Context>(Output(i)->count(),
                outside_w_data, dXdata, dXdata, ctx());
        }
    }
}

template <class Context>
void SmoothL1LossGradientOp<Context>::RunOnDevice() {
    diff = ws()->GetTensor(mount_name("smoothl1_loss/diff"));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(SmoothL1LossGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SmoothL1LossGradient);
#endif

OPERATOR_SCHEMA(SmoothL1LossGradient)
    .NumInputs(3, 5).NumOutputs(2);

class GetSmoothL1LossGradient
    final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetSmoothL1LossGradient);
    vector<OperatorDef> MakeDefs() override {
        vector<string> inputs;
        for (auto input : def.input()) inputs.push_back(input);
        inputs.push_back(GO(0));
        return SingleDef(def.type() + "Gradient", "",
            inputs, vector<string>({ GI(0), GI(1) }));
    }
};

REGISTER_GRADIENT(SmoothL1Loss, GetSmoothL1LossGradient);

}  // namespace dragon