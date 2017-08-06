#include "operators/activation/dropout_op.h"
#include "core/workspace.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void DropoutOp<Context>::RunWithType() {
    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    uint32_t* Mdata = mask->template mutable_data<uint32_t, Context>();

    if (this->phase() == "TRAIN") {
        kernel::Dropout<T, Context>(output(0)->count(), 
                                                  prob, 
                                                 scale, 
                                                 Xdata, 
                                                 Mdata,
                                                 Ydata, 
                                               &ctx());
    } else if (this->phase() == "TEST") {
        ctx().template Copy<T, Context, Context>(output(0)->count(), Ydata, Xdata);
        if (scale == 1.0) math::Scal<T, Context>(output(0)->count(), 1.0 - prob, Ydata);
    }
}

template <class Context>
void DropoutOp<Context>::RunOnDevice() {
    output(0)->ReshapeLike(input(0));
    mask = ws()->CreateTensor("_t_" + anchor() + "_dropout_mask");
    mask->ReshapeLike(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(Dropout);
#ifdef WITH_CUDA
DEPLOY_CUDA(Dropout);
#endif
OPERATOR_SCHEMA(Dropout).NumInputs(1).NumOutputs(1).Inplace({ { 0, 0 } });

template <class Context> template <typename T>
void DropoutGradientOp<Context>::RunWithType() {
    mask = ws()->GetTensor("_t_" + anchor() + "_dropout_mask");

    auto* dYdata = input(-1).template data<T, Context>();
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    auto* Mdata = mask->template data<uint32_t, Context>();

    if (this->phase() == "TRAIN") {
        kernel::DropoutGrad<T, Context>(output(0)->count(), 
                                                      prob, 
                                                     scale,
                                                    dYdata, 
                                                     Mdata,
                                                   dXdata);

    } else if (this->phase() == "TEST") {
        NOT_IMPLEMENTED;
    }
}

template <class Context>
void DropoutGradientOp<Context>::RunOnDevice() {
    output(0)->ReshapeLike(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

template <class Context>
void DropoutGradientOp<Context>::ClearAfterRun() {
    ws()->ReleaseBuffer(mask, true);
}

DEPLOY_CPU(DropoutGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(DropoutGradient);
#endif
OPERATOR_SCHEMA(DropoutGradient).NumInputs(2).NumOutputs(1).Inplace({ { 1, 0 } });

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

