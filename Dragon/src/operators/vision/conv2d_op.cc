#include "operators/vision/conv_op.h"
#include "core/workspace.h"
#include "utils/filler.h"

namespace dragon {

template <class Context> template <typename T>
void Conv2dOp<Context>::RunWithType() {
    //  get buffer
    this->col_buffer = ws()->GetBuffer();
    this->col_buffer->Reshape(this->col_shape);

    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    TENSOR_FILL(input(1), this->weight_shape);
    auto* Wdata = input(1).template data<T, Context>();
    if (HasBias()) {
        TENSOR_FILL(input(2), this->bias_shape);
        INIT_MULTIPLIER(this->bias_multiplier, this->out_spatial_dim);
    }

    for (int n = 0; n < input(0).dim(0); n++) {
        Wx(Xdata + n * this->x_offset, Wdata, Ydata + n * this->y_offset);
        if (HasBias()) {
            auto* Bdata = input(2).template data<T, Context>();
            Pb(Bdata, Ydata + n * this->y_offset);
        }
    }

    //  release buffer
    ws()->ReleaseBuffer(this->col_buffer);
}

template <class Context>
void Conv2dOp<Context>::RunOnDevice() {
    Reshape();

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(Conv2d);
#ifdef WITH_CUDA
DEPLOY_CUDA(Conv2d);
#endif
OPERATOR_SCHEMA(Conv2d).NumInputs(2, 3).NumOutputs(1);

template <class Context> template <typename T>
void Conv2dGradientOp<Context>::RunWithType() {
    //  get buffer
    this->col_buffer = ws()->GetBuffer();
    this->col_buffer->Reshape(this->col_shape);

    auto* dYdata = input(-1).template data<T, Context>();

    if (HasBias()) {
        INIT_MULTIPLIER(this->bias_multiplier, this->out_spatial_dim);
        T* dBdata = output(2)->template mutable_data<T, Context>();
        for (int n = 0; n < input(2).dim(0); n++)
            Db(dYdata + n * this->y_offset, dBdata);
    }

    for (int n = 0; n < input(2).dim(0); n++) {
        if (output(1)->name() != "ignore") {
            auto* Xdata = input(0).template data<T, Context>();
            auto* dWdata = output(1)->template mutable_data<T, Context>();
            Dw(dYdata + n * this->y_offset, Xdata + n * this->x_offset, dWdata);
        }
        if (output(0)->name() != "ignore") {
            auto* Wdata = input(1).template data<T, Context>();
            auto* dXdata = output(0)->template mutable_data<T, Context>();
            Dx(dYdata + n * this->y_offset, Wdata, dXdata + n * this->x_offset);
        }
    }

    //  release buffer
    ws()->ReleaseBuffer(this->col_buffer);
}

template <class Context>
void Conv2dGradientOp<Context>::RunOnDevice() {
    GradientReshape();

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types."; 
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