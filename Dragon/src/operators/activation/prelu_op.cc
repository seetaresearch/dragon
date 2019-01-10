#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"
#include "operators/activation/prelu_op.h"

namespace dragon {

template <class Context> template <typename T>
void PReluOp<Context>::RunWithType() {
    if (channel_shared) { TENSOR_FILL(Input(1), vector<int64_t>(1, 1)); }
    else { TENSOR_FILL(Input(1), vector<int64_t>(1, Input(0).dim(1))); }

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Wdata = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    kernel::PRelu(Output(0)->count(), channels, dim,
        channel_shared ? true : false, data_format,
            Xdata, Wdata, Ydata, ctx());
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

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
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
        DECLARE_MULTIPLIER(multiplier, channels * dim);
        auto* dWdata = Output(1)->template mutable_data<T, Context>();
        auto* dWBdata = ws()->template caches<T, Context>({ channels * dim })[0];
        kernel::PReluWGrad(Input(0).dim(0), Input(0).count(1),
            channels, dim, channel_shared ? true : false, data_format,
                dYdata, Xdata, multiplier, dWBdata, dWdata, ctx());
    }

    if (Output(0)->name() != "ignore") {
        auto* Wdata = Input(1).template data<T, Context>();
        auto* dXdata = Output(0)->template mutable_data<T, Context>();
        kernel::PReluGrad(Output(0)->count(), channels, dim,
            channel_shared ? true : false, data_format,
                dYdata, Xdata, Wdata, dXdata, ctx());
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

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(PReluGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(PReluGradient);
#endif

OPERATOR_SCHEMA(PReluGradient)
    .NumInputs(3).NumOutputs(2);

class GetPReluGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetPReluGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), GO(0) }),
            vector<string>({ GI(0), GI(1)} ));
    }
};

REGISTER_GRADIENT(PRelu, GetPReluGradient);

}  // namespace dragon