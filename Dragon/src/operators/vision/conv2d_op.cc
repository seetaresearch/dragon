#include "core/workspace.h"
#include "utils/filler.h"
#include "operators/vision/conv_op.h"

namespace dragon {

template <class Context> template <typename T>
void Conv2dOp<Context>::RunImpl() {
    TENSOR_FILL(X(1), w_shape_);
    if (HasBias()) TENSOR_FILL(X(2), b_shape_);

    auto* x = X(0).template data<T, Context>();
    auto* w = X(1).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    for (int n = 0; n < X(0).dim(0); n++)
        Wx(x + n * x_ofs_, w, y + n * y_ofs_, false);
  
    if (HasBias()) Pb(X(2).template data<T, Context>(), y);
}

template <class Context>
void Conv2dOp<Context>::RunOnDevice() {
    ConvOpBase<Context>::Reshape();

    // You really need the CuDNN to help you -:)
    if (data_format() == "NHWC" && group_ != 1)
        LOG(FATAL) << "GroupConv(NHWC) is not supported.";

    if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else {
        LOG(FATAL) << DTypeString(
            X(0), { "float32" }
        );
    }
}

template <class Context> template <typename T>
void Conv2dGradientOp<Context>::RunImpl() {
    auto* dy = X(-1).template data<T, Context>();

    if (Y(2)->name() != "NULL") {
        Db(dy, Y(2)->template mutable_data<T, Context>());
    }

    for (int n = 0; n < X(0).dim(0); n++) {
        if (Y(1)->name() != "NULL") {
            auto* x = X(0).template data<T, Context>();
            auto* dw = Y(1)->template mutable_data<T, Context>();
            Dw(dy + n * y_ofs_, x + n * x_ofs_, dw, n > 0);
        }
        if (Y(0)->name() != "NULL") {
            auto* w = X(1).template data<T, Context>();
            auto* dx = Y(0)->template mutable_data<T, Context>();
            Dx(dy + n * y_ofs_, w, dx + n * x_ofs_);
        }
    }
}

template <class Context>
void Conv2dGradientOp<Context>::RunOnDevice() {
    ConvOpBase<Context>::Reshape(true);

    // You really need the CuDNN to help you -:)
    if (data_format() == "NHWC" && group_ != 1)
        LOG(FATAL) << "GroupConv(NHWC) is not supported.";

    if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else {
        LOG(FATAL) << DTypeString(
            X(0), { "float32" }
        );
    }
}

DEPLOY_CPU(Conv2d);
#ifdef WITH_CUDA
DEPLOY_CUDA(Conv2d);
#endif

DEPLOY_CPU(Conv2dGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(Conv2dGradient);
#endif

OPERATOR_SCHEMA(Conv2d)
     /* X, W, B */
    .NumInputs(2, 3)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(Conv2dGradient)
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

REGISTER_GRADIENT(Conv2d, GradientMaker);

}  // namespace dragon