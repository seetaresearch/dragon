#include "operators/vision/deconv_op.h"
#include "core/workspace.h"
#include "utils/filler.h"

namespace dragon {

template <class Context>
void DeConvOp<Context>::ComputeOutputShape() {
    this->output_shape.clear();
    for (int i = 0; i < this->num_spatial_axes; i++) {
        const int input_dim = this->bottom_shape[this->channel_axis + i + 1];
        const int dilated_kernel = this->dilation[i] * (this->kernel_size[i] - 1) + 1;
        const int output_dim = this->stride[i] * (input_dim - 1) + dilated_kernel - 2 * this->pad[i];
        this->output_shape.push_back(output_dim);
    }
}

template <class Context> template <typename T>
void DeConvOp<Context>::RunWithType() {
    //  get buffer
    this->col_buffer = ws()->GetBuffer();
    this->col_buffer->Reshape(this->col_buffer_shape);

    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    TENSOR_FILL(input(1), this->weight_shape);
    auto* Wdata = input(1).template data<T, Context>();
    if (InputSize() > 2) {
        TENSOR_FILL(input(2), this->bias_shape);
        INIT_MULTIPLIER(this->bias_multiplier, this->out_spatial_dim);
    }

    for (int n = 0; n < input(0).dim(0); n++) {
        Dx(Xdata + n * this->x_offset, Wdata, Ydata + n * this->y_offset);
        if (InputSize() > 2) {
            auto* Bdata = input(2).template data<T, Context>();
            Pb(Bdata, Ydata + n * this->y_offset);
        }
    }

    //  release buffer
    ws()->ReleaseBuffer(this->col_buffer);
}

template <class Context>
void DeConvOp<Context>::RunOnDevice() {
    Reshape();

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(DeConv);
#ifdef WITH_CUDA
DEPLOY_CUDA(DeConv);
#endif
OPERATOR_SCHEMA(DeConv).NumInputs(2, 3).NumOutputs(1);

template <class Context> template <typename T>
void DeConvGradientOp<Context>::RunWithType() {
    //  get buffer
    this->col_buffer = ws()->GetBuffer();
    this->col_buffer->Reshape(this->col_buffer_shape);

    auto* dYdata = input(-1).template data<T, Context>();

    if (output(2)->name() != "ignore") {
        INIT_MULTIPLIER(this->bias_multiplier, this->out_spatial_dim);
        auto* dBdata = output(2)->template mutable_data<T, Context>();
        for (int n = 0; n < input(2).dim(0); n++)
            Db(dYdata + n * this->y_offset, dBdata);
    }

    for (int n = 0; n < input(2).dim(0); n++) {
        if (output(1)->name() != "ignore") {
            auto* Xdata = input(0).template data<T, Context>();
            auto* dWdata = output(1)->template mutable_data<T, Context>();
            Dw(Xdata + n * this->x_offset, dYdata + n * this->y_offset, dWdata);
        }
        if (output(0)->name() != "ignore") {
            auto* Wdata = input(1).template data<T, Context>();
            auto* dXdata = output(0)->template mutable_data<T, Context>();
            bool skip = output(1)->name() != "ignore";
            Wx(dYdata + n * this->y_offset, Wdata, dXdata + n * this->x_offset, skip);
        }
    }

    //  release buffer
    ws()->ReleaseBuffer(this->col_buffer);
}

template <class Context>
void DeConvGradientOp<Context>::RunOnDevice() {
    GradientReshape();

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

template <class Context>
void DeConvGradientOp<Context>::ShareBeforeRun() {
    if (output(0)->name() != "ignore") {
        Tensor* dX = ws()->GetBuffer();
        if (dX != nullptr) output(0)->Replace(*dX);
    }
}

template <class Context>
void DeConvGradientOp<Context>::ClearAfterRun() {
    Tensor* dY = &input(-1);
    ws()->ReleaseBuffer(dY);
}

DEPLOY_CPU(DeConvGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(DeConvGradient);
#endif
OPERATOR_SCHEMA(DeConvGradient).NumInputs(3).NumOutputs(3);

class GetDeConvGradient final : public GradientMakerBase {
public:
    GRADIENT_MAKER_CTOR(GetDeConvGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0), GI(1), GI(2)});
    }
};
REGISTER_GRADIENT(DeConv, GetDeConvGradient);

}    // namespace dragon