#include "operators/activation/prelu_op.h"
#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void PReluOp<Context>::RunWithType() {
    if (channel_shared) {
        TENSOR_FILL(input(1), vector<TIndex>(1, 1));
    } else {
        TENSOR_FILL(input(1), vector<TIndex>(1, input(0).dim(1)));
    }

    auto* Xdata = input(0).template data<T, Context>();
    auto* Wdata = input(1).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    kernel::PRelu<T, Context>(output(0)->count(),
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
        channels = input(0).dim(1);
        dim = input(0).count(2);
    } else {
        channels = input(0).dim(-1);
        dim = input(0).count() / channels;
    }
    output(0)->ReshapeLike(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(PRelu);
#ifdef WITH_CUDA
DEPLOY_CUDA(PRelu);
#endif
OPERATOR_SCHEMA(PRelu).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void PReluGradientOp<Context>::RunWithType() {
    auto* Xdata = input(0).template data<T, Context>();
    auto* dYdata = input(-1).template data<T, Context>();

    if (output(1)->name() != "ignore") {
        INIT_MULTIPLIER(multiplier, channels * dim);
        bcast_dw = ws()->GetBuffer();
        bcast_dw->Reshape(vector<TIndex>(1, channels * dim));
        auto* dWdata = output(1)->template mutable_data<T, Context>();
        auto* dWBdata = bcast_dw->template mutable_data<T, Context>();
        kernel::PReluWGrad<T, Context>(input(0).dim(0),
                                     input(0).count(1),
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

    if (output(0)->name() != "ignore") {
        auto* Wdata = input(1).template data<T, Context>();
        auto* dXdata = output(0)->template mutable_data<T, Context>();
        kernel::PReluGrad<T, Context>(output(0)->count(),
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
        channels = input(0).dim(1);
        dim = input(0).count(2);
    } else {
        channels = input(0).dim(-1);
        dim = input(0).count() / channels;
    }

    output(0)->ReshapeLike(input(0));
    output(1)->ReshapeLike(input(1));

    if (input(0).template IsType<float>()) RunWithType<float>();
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