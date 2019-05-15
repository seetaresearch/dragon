#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/op_kernel.h"
#include "operators/vision/depthwise_conv_op.h"

namespace dragon {

template <class Context> template <typename T>
void DepthwiseConv2dOp<Context>::RunImpl() {
    TENSOR_FILL(X(1), w_shape_);
    if (HasBias()) TENSOR_FILL(X(2), b_shape_);

    auto* x = X(0).template data<T, Context>();
    auto* w = X(1).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    kernel::DepthwiseConv2d(
        X(0).dim(0), channels_,
        in_shape_[0], in_shape_[1],
        out_shape_[0], out_shape_[1],
        kshape_[0], kshape_[1],
        stride_[0], stride_[1],
        pad_l_[0], pad_l_[1],
        dilation_[0], dilation_[1],
        data_format(),
        x, w,
        y, ctx()
    );

    if (HasBias()) {
        Pb(X(2).template data<T, Context>(), y);
    }
}

template <class Context>
void DepthwiseConv2dOp<Context>::RunOnDevice() {
    group_ = channels_ = data_format() == "NCHW" ?
        X(0).dim(1) : X(0).dim(-1);
    CHECK_EQ(channels_, num_output_)
        << "\nExcepted in/out channels unchanged.";
    ConvOpBase<Context>::Reshape();

    DispatchHelper<TensorTypes
        <float>>::Call(this, X(0));
}

template <class Context> template <typename T>
void DepthwiseConv2dGradientOp<Context>::RunImpl() {
    auto* dy = X(-1).template data<T, Context>();

    if (Y(2)->name() != "NULL") {
        Db(dy, Y(2)->template mutable_data<T, Context>());
    }

    if (Y(1)->name() != "NULL") {
        auto* x = X(0).template data<T, Context>();
        auto* dw = Y(1)->template mutable_data<T, Context>();
        kernel::DepthwiseConv2dWGrad(
            X(0).dim(0), channels_,
            in_shape_[0], in_shape_[1],
            out_shape_[0], out_shape_[1],
            kshape_[0], kshape_[1],
            stride_[0], stride_[1],
            pad_l_[0], pad_l_[1],
            dilation_[0], dilation_[1],
            data_format(),
            dy, x,
            dw, ctx()
        );
    }

    if (Y(0)->name() != "NULL") {
        auto* w = X(1).template data<T, Context>();
        auto* dx = Y(0)->template mutable_data<T, Context>();
        kernel::DepthwiseConv2dGrad(
            X(0).dim(0), channels_,
            in_shape_[0], in_shape_[1],
            out_shape_[0], out_shape_[1],
            kshape_[0], kshape_[1],
            stride_[0], stride_[1],
            pad_l_[0], pad_l_[1],
            dilation_[0], dilation_[1],
            data_format(),
            dy, w,
            dx, ctx()
        );
    }
}

template <class Context>
void DepthwiseConv2dGradientOp<Context>::RunOnDevice() {
    group_ = channels_ = data_format() == "NCHW" ?
        X(0).dim(1) : X(0).dim(-1);
    ConvOpBase<Context>::Reshape(true);

    DispatchHelper<TensorTypes
        <float>>::Call(this, X(0));
}

DEPLOY_CPU(DepthwiseConv2d);
#ifdef WITH_CUDA
DEPLOY_CUDA(DepthwiseConv2d);
#endif

DEPLOY_CPU(DepthwiseConv2dGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(DepthwiseConv2dGradient);
#endif

OPERATOR_SCHEMA(DepthwiseConv2d)
     /* X, W, B */
    .NumInputs(2, 3)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(DepthwiseConv2dGradient)
     /* X, W, dY */
    .NumInputs(3)
     /* dX, dW, dB */
    .NumOutputs(3);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GradientMaker);
    vector<OperatorDef> MakeDef() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), GO(0) }),
            vector<string>({ GI(0), GI(1), GI(2) })
        );
    }
};

}  // namespace

REGISTER_GRADIENT(DepthwiseConv2d, GradientMaker);

}  // namespace dragon