#include "core/workspace.h"
#include "utils/filler.h"
#include "operators/vision/conv_transpose_op.h"

namespace dragon {

template <class Context> template <typename T>
void Conv2dTransposeOp<Context>::RunWithType() {
    TENSOR_FILL(Input(1), weight_shape);
    if (InputSize() > 2) { TENSOR_FILL(Input(2), bias_shape); }

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Wdata = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    for (int n = 0; n < Input(0).dim(0); n++) {
        Dx(Xdata + n * x_offset, Wdata, Ydata + n * y_offset);
        if (InputSize() > 2) {
            auto* Bdata = Input(2).template data<T, Context>();
            Pb(Bdata, Ydata + n * y_offset);
        }
    }
}

template <class Context>
void Conv2dTransposeOp<Context>::RunOnDevice() {
    Reshape();
    //  fix the output shape for im2col/col2im
    for (int i = 0; i < this->num_spatial_axes; i++) 
        this->output_shape[i] = Input(0).dim(this->spatial_axis + i);

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(Conv2dTranspose);
#ifdef WITH_CUDA
DEPLOY_CUDA(Conv2dTranspose);
#endif
OPERATOR_SCHEMA(Conv2dTranspose).NumInputs(2, 3).NumOutputs(1);

template <class Context> template <typename T>
void Conv2dTransposeGradientOp<Context>::RunWithType() {
    auto* dYdata = Input(-1).template data<T, Context>();

    if (Output(2)->name() != "ignore") {
        auto* dBdata = Output(2)->template mutable_data<T, Context>(ctx());
        for (int n = 0; n < Input(2).dim(0); n++)
            Db(dYdata + n * y_offset, dBdata);
    }

    for (int n = 0; n < Input(2).dim(0); n++) {
        if (Output(1)->name() != "ignore") {
            auto* Xdata = Input(0).template data<T, Context>();
            auto* dWdata = Output(1)->template mutable_data<T, Context>(ctx());
            Dw(Xdata + n * x_offset, dYdata + n * y_offset, dWdata);
        }
        if (Output(0)->name() != "ignore") {
            auto* Wdata = Input(1).template data<T, Context>();
            auto* dXdata = Output(0)->template mutable_data<T, Context>();
            bool skip = Output(1)->name() != "ignore";
            Wx(dYdata + n * y_offset, Wdata, dXdata + n * x_offset, skip);
        }
    }
}

template <class Context>
void Conv2dTransposeGradientOp<Context>::RunOnDevice() {
    GradientReshape();
    //  fix the output shape for im2col/col2im
    for (int i = 0; i < this->num_spatial_axes; i++)
        this->output_shape[i] = Input(0).dim(this->spatial_axis + i);

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(Conv2dTransposeGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(Conv2dTransposeGradient);
#endif
OPERATOR_SCHEMA(Conv2dTransposeGradient).NumInputs(3).NumOutputs(3);

class GetConv2dTransposeGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetConv2dTransposeGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0), GI(1), GI(2)});
    }
};
REGISTER_GRADIENT(Conv2dTranspose, GetConv2dTransposeGradient);

}    // namespace dragon