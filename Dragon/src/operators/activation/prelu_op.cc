#include "operators/activation/prelu_op.h"
#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void PReluOp<Context>::RunWithType() {
    if (channel_shared) {
        TENSOR_FILL(Input(1), vector<TIndex>(1, 1));
    } else {
        TENSOR_FILL(Input(1), vector<TIndex>(1, Input(0).dim(1)));
    }

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Wdata = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    kernel::PRelu<T, Context>(Output(0)->count(),
                                        channels,
                                             dim,
                                  channel_shared,
                                     data_format,
                                           Xdata,
                                           Wdata,
                                          Ydata);
}

template <class Context>
void PReluOp<Context>::RunOnDevice() {
    if (data_format == "NCHW") {
        channels = Input(0).dim(1);
        dim = Input(0).count(2);
    } else {
        channels = Input(0).dim(-1);
        dim = Input(0).count(1) / channels;
    }
    Output(0)->ReshapeLike(Input(0));

    if (Input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(PRelu);
#ifdef WITH_CUDA
DEPLOY_CUDA(PRelu);
#endif
OPERATOR_SCHEMA(PRelu).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void PReluGradientOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* dYdata = Input(-1).template data<T, Context>();

    if (Output(1)->name() != "ignore") {
        INIT_MULTIPLIER(multiplier, channels * dim);
        bcast_dw = ws()->GetBuffer();
        bcast_dw->Reshape(vector<TIndex>(1, channels * dim));
        auto* dWdata = Output(1)->template mutable_data<T, Context>();
        auto* dWBdata = bcast_dw->template mutable_data<T, Context>();
        kernel::PReluWGrad<T, Context>(Input(0).dim(0),
                                     Input(0).count(1),
                                              channels,
                                                   dim,
                                        channel_shared,
                                           data_format,
                                                dYdata,
                                                 Xdata,
               multiplier->template data<T, Context>(),
                                               dWBdata,
                                               dWdata);
        ws()->ReleaseBuffer(bcast_dw);
    }

    if (Output(0)->name() != "ignore") {
        auto* Wdata = Input(1).template data<T, Context>();
        auto* dXdata = Output(0)->template mutable_data<T, Context>();
        kernel::PReluGrad<T, Context>(Output(0)->count(),
                                                channels,
                                                     dim,
                                          channel_shared,
                                             data_format,
                                                  dYdata,
                                                   Xdata,
                                                   Wdata,
                                                 dXdata);
    }
}

template <class Context>
void PReluGradientOp<Context>::RunOnDevice() {
    if (data_format == "NCHW") {
        channels = Input(0).dim(1);
        dim = Input(0).count(2);
    } else {
        channels = Input(0).dim(-1);
        dim = Input(0).count(1) / channels;
    }

    Output(0)->ReshapeLike(Input(0));
    Output(1)->ReshapeLike(Input(1));

    if (Input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(PReluGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(PReluGradient);
#endif
OPERATOR_SCHEMA(PReluGradient).NumInputs(3).NumOutputs(2);

class GetPReluGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetPReluGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0), GI(1)});
    }
};
REGISTER_GRADIENT(PRelu, GetPReluGradient);

}    // namespace dragon