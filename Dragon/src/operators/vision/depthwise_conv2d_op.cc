#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/op_kernel.h"
#include "operators/vision/depthwise_conv_op.h"

namespace dragon {

template <class Context> template <typename T>
void DepthwiseConv2dOp<Context>::RunWithType() {
    TENSOR_FILL(Input(1), weight_shape);
    if (HasBias()) { TENSOR_FILL(Input(2), bias_shape); }

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Wdata = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    kernel::DepthwiseConv2d(
        Input(0).dim(0), channels,
        input_shape[0], input_shape[1],
        output_shape[0], output_shape[1],
        kernel_shape[0], kernel_shape[1],
        stride[0], pad_l[0], pad_l[1],
        data_format, Xdata, Wdata, Ydata, ctx());

    if (HasBias()) {
        auto* Bdata = Input(2).template data<T, Context>();
        Pb(Bdata, Ydata);
    }
}

template <class Context>
void DepthwiseConv2dOp<Context>::RunOnDevice() {
    group = channels = data_format == "NCHW" ?
        Input(0).dim(1) : Input(0).dim(-1);
    CHECK_EQ(channels, num_output)
        << "Excepted in/out channels unchanged.";
    Reshape();

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(DepthwiseConv2d);
#ifdef WITH_CUDA
DEPLOY_CUDA(DepthwiseConv2d);
#endif
OPERATOR_SCHEMA(DepthwiseConv2d).NumInputs(2, 3).NumOutputs(1);

template <class Context> template <typename T>
void DepthwiseConv2dGradientOp<Context>::RunWithType() {
    auto* dYdata = Input(-1).template data<T, Context>();

    if (HasBias()) {
        T* dBdata = Output(2)->template mutable_data<T, Context>();
        for (int n = 0; n < Input(2).dim(0); n++)
            Db(dYdata + n * y_offset, dBdata);
    }

    for (int n = 0; n < Input(2).dim(0); n++) {
        if (Output(1)->name() != "NULL") {
            auto* Xdata = Input(0).template data<T, Context>();
            auto* dWdata = Output(1)->template mutable_data<T, Context>();
            math::Set(
                Output(1)->count(),
                cast::to<T>(0.f),
                dWdata,
                ctx()
            );  // Zero the gradient of W
            kernel::DepthwiseConv2dWGrad(
                Input(0).dim(0), channels,
                input_shape[0], input_shape[1],
                output_shape[0], output_shape[1],
                kernel_shape[0], kernel_shape[1],
                stride[0], pad_l[0], pad_l[1],
                data_format, dYdata, Xdata, dWdata, ctx());
        }
        if (Output(0)->name() != "NULL") {
            auto* Wdata = Input(1).template data<T, Context>();
            auto* dXdata = Output(0)->template mutable_data<T, Context>();
            kernel::DepthwiseConv2dGrad(
                Input(0).dim(0), channels,
                input_shape[0], input_shape[1],
                output_shape[0], output_shape[1],
                kernel_shape[0], kernel_shape[1],
                stride[0], pad_l[0], pad_l[1],
                data_format, dYdata, Wdata, dXdata, ctx());
        }
    }
}

template <class Context>
void DepthwiseConv2dGradientOp<Context>::RunOnDevice() {
    group = channels = data_format == "NCHW" ?
        Input(0).dim(1) : Input(0).dim(-1);
    GradientReshape();

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(DepthwiseConv2dGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(DepthwiseConv2dGradient);
#endif

OPERATOR_SCHEMA(DepthwiseConv2dGradient)
    .NumInputs(3).NumOutputs(3);

class GetDepthwiseConv2dGradient
    final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetDepthwiseConv2dGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), GO(0) }),
            vector<string>({ GI(0), GI(1), GI(2) }));
    }
};

REGISTER_GRADIENT(DepthwiseConv2d, GetDepthwiseConv2dGradient);

}  // namespace dragon