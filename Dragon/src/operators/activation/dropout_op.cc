#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "operators/activation/dropout_op.h"

namespace dragon {

template <class Context> template <typename T>
void DropoutOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    float scale = use_scale ? 1.f / (1.f - prob()) : 1.f;
    if (phase() == "TEST") {
        if (Output(0) != &Input(0)) {
            ctx()->template Copy<T, Context, Context>(
                Output(0)->count(), Ydata, Xdata);
        }
        if (!use_scale) {
            math::Scal<T, Context>(Output(0)->count(),
                1.0 - prob(), Ydata, ctx());
        }
    } else if (phase() == "TRAIN") {
        Tensor* mask = ws()->CreateTensor(
            "/mnt/" + anchor() + "/dropout/mask");
        mask->ReshapeLike(Input(0));

        auto WSdata = ws()->template caches<Context>({
            mask->count() * sizeof(uint32_t) });

        auto* Mdata = mask->template mutable_data<uint8_t, Context>();

        kernel::Dropout<T, Context>(
            Output(0)->count(), prob(), scale,
                Xdata, (uint32_t*)WSdata[0],
                    Mdata, Ydata, ctx());

    } else LOG(FATAL) << "Incorrect Op phase: " << phase();
}

template <class Context>
void DropoutOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(Dropout);
#ifdef WITH_CUDA
DEPLOY_CUDA(Dropout);
#endif
OPERATOR_SCHEMA(Dropout)
    .NumInputs(1).NumOutputs(1)
    .Inplace({ { 0, 0 } });

template <class Context> template <typename T>
void DropoutGradientOp<Context>::RunWithType() {
    auto* mask = ws()->GetTensor(
        "/mnt/" + anchor() + "/dropout/mask");

    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* Mdata = mask->template data<uint8_t, Context>();

    float scale = use_scale ? 1.f / (1.f - prob()) : 1.f;

    if (phase() == "TEST") { NOT_IMPLEMENTED; }
    else if (phase() == "TRAIN") {
        kernel::ApplyMask<T, uint8_t, Context>(mask->count(),
            scale, dYdata, Mdata, dXdata, ctx());
    } else LOG(FATAL) << "Incorrect Op phase: " << phase();
}

template <class Context>
void DropoutGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(DropoutGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(DropoutGradient);
#endif
OPERATOR_SCHEMA(DropoutGradient)
    .NumInputs(2).NumOutputs(1)
    .Inplace({ { 1, 0 } });

class GetDropoutGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetDropoutGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {O(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(Dropout, GetDropoutGradient);

}    // namepsace dragon