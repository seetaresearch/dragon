#include "operators/vision/conv_op.h"
#include "core/workspace.h"
#include "utils/filler.h"

namespace dragon {

template <class Context> template <typename T>
void Conv2dOp<Context>::RunWithType() {
    //  get buffer
    this->col_buffer = ws()->GetBuffer();
    this->col_buffer->Reshape(this->col_shape);

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    TENSOR_FILL(Input(1), this->weight_shape);
    auto* Wdata = Input(1).template data<T, Context>();
    if (HasBias()) {
        TENSOR_FILL(Input(2), this->bias_shape);
        INIT_MULTIPLIER(this->bias_multiplier, this->out_spatial_dim);
    }

    for (int n = 0; n < Input(0).dim(0); n++) {
        Wx(Xdata + n * this->x_offset, Wdata, Ydata + n * this->y_offset);
        if (HasBias()) {
            auto* Bdata = Input(2).template data<T, Context>();
            Pb(Bdata, Ydata + n * this->y_offset);
        }
    }

    //  release buffer
    ws()->ReleaseBuffer(this->col_buffer);
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
    //  get buffer
    this->col_buffer = ws()->GetBuffer();
    this->col_buffer->Reshape(this->col_shape);

    auto* dYdata = Input(-1).template data<T, Context>();

    if (HasBias()) {
        INIT_MULTIPLIER(this->bias_multiplier, this->out_spatial_dim);
        T* dBdata = Output(2)->template mutable_data<T, Context>();
        for (int n = 0; n < Input(2).dim(0); n++)
            Db(dYdata + n * this->y_offset, dBdata);
    }

    for (int n = 0; n < Input(2).dim(0); n++) {
        if (Output(1)->name() != "ignore") {
            auto* Xdata = Input(0).template data<T, Context>();
            auto* dWdata = Output(1)->template mutable_data<T, Context>();
            Dw(dYdata + n * this->y_offset, Xdata + n * this->x_offset, dWdata);
        }
        if (Output(0)->name() != "ignore") {
            auto* Wdata = Input(1).template data<T, Context>();
            auto* dXdata = Output(0)->template mutable_data<T, Context>();
            Dx(dYdata + n * this->y_offset, Wdata, dXdata + n * this->x_offset);
        }
    }

    //  release buffer
    ws()->ReleaseBuffer(this->col_buffer);
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