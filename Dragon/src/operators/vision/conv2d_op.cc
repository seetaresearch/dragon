#include "core/workspace.h"
#include "utils/filler.h"
#include "operators/vision/conv_op.h"

namespace dragon {

template <class Context> template <typename T>
void Conv2dOp<Context>::RunWithType() {
    TENSOR_FILL(Input(1), weight_shape);
    if (HasBias()) { TENSOR_FILL(Input(2), bias_shape); }

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Wdata = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    for (int n = 0; n < Input(0).dim(0); n++) {
        Wx(Xdata + n * x_offset, Wdata, Ydata + n * y_offset);
        if (HasBias()) {
            auto* Bdata = Input(2).template data<T, Context>();
            Pb(Bdata, Ydata + n * y_offset);
        }
    }
}

template <class Context>
void Conv2dOp<Context>::RunOnDevice() {
    Reshape();

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(Conv2d);
#ifdef WITH_CUDA
DEPLOY_CUDA(Conv2d);
#endif
OPERATOR_SCHEMA(Conv2d).NumInputs(2, 3).NumOutputs(1);

template <class Context> template <typename T>
void Conv2dGradientOp<Context>::RunWithType() {
    auto* dYdata = Input(-1).template data<T, Context>();

    if (HasBias()) {
        T* dBdata = Output(2)->template mutable_data<T, Context>();
        for (int n = 0; n < Input(2).dim(0); n++)
            Db(dYdata + n * y_offset, dBdata);
    }

    for (int n = 0; n < Input(2).dim(0); n++) {
        if (Output(1)->name() != "ignore") {
            auto* Xdata = Input(0).template data<T, Context>();
            auto* dWdata = Output(1)->template mutable_data<T, Context>();
            Dw(dYdata + n * y_offset, Xdata + n * x_offset, dWdata);
        }
        if (Output(0)->name() != "ignore") {
            auto* Wdata = Input(1).template data<T, Context>();
            auto* dXdata = Output(0)->template mutable_data<T, Context>();
            Dx(dYdata + n * y_offset, Wdata, dXdata + n * x_offset);
        }
    }
}

template <class Context>
void Conv2dGradientOp<Context>::RunOnDevice() {
    GradientReshape();

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(Conv2dGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(Conv2dGradient);
#endif
OPERATOR_SCHEMA(Conv2dGradient).NumInputs(3).NumOutputs(3);

class GetConv2dGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetConv2dGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0), GI(1), GI(2)});
    }
};
REGISTER_GRADIENT(Conv2d, GetConv2dGradient);

}    // namespace dragon