#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"
#include "operators/activation/prelu_op.h"

namespace dragon {

template <class Context> template <typename T>
void PReluOp<Context>::RunImpl() {
    if (channel_shared_) {
        TENSOR_FILL(X(1), vec64_t({ 1 }));
    } else {
        TENSOR_FILL(X(1), vec64_t({ X(0).dim(1) }));
    }

    auto* x = X(0).template data<T, Context>();
    auto* w = X(1).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    kernel::PRelu(
        X(0).count(),
        channels_, dim_,
        channel_shared_ ? true : false,
        data_format(),
        x, w,
        y, ctx()
    );
}

template <class Context>
void PReluOp<Context>::RunOnDevice() {
    if (data_format() == "NCHW") {
        channels_ = X(0).dim(1);
        dim_ = X(0).count(2);
    } else {
        channels_ = X(0).dim(-1);
        dim_ = X(0).count(1) / channels_;
    }

    Y(0)->ReshapeLike(X(0));

    if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else {
        LOG(FATAL) << DTypeString(X(0),
            { "float32" }
        );
    }
}

template <class Context> template <typename T>
void PReluGradientOp<Context>::RunImpl() {
    auto* x  = X(0).template data<T, Context>();
    auto* dy = X(2).template data<T, Context>();

    if (Y(1)->name() != "NULL") {
        DECLARE_MULTIPLIER(multiplier, channels_ * dim_);
        auto* dw = Y(1)->template mutable_data<T, Context>();
        auto* scratch = ws()
            ->template data<T, Context>
                ({ channels_ * dim_ })[0];
        kernel::PReluWGrad(
            X(0).dim(0),
            X(0).count(1),
            channels_, dim_,
            channel_shared_ ? true : false,
            data_format(),
            dy, x, multiplier,
            scratch, dw, ctx()
        );
    }

    if (Y(0)->name() != "NULL") {
        auto* w = X(1).template data<T, Context>();
        auto* dx = Y(0)->template mutable_data<T, Context>();
        kernel::PReluGrad(
            X(0).count(),
            channels_, dim_,
            channel_shared_ ? true : false,
            data_format(),
            dy, x, w,
            dx, ctx()
        );
    }
}

template <class Context>
void PReluGradientOp<Context>::RunOnDevice() {
    if (data_format() == "NCHW") {
        channels_ = X(0).dim(1);
        dim_ = X(0).count(2);
    } else {
        channels_ = X(0).dim(-1);
        dim_ = X(0).count(1) / channels_;
    }

    Y(0)->ReshapeLike(X(0));
    Y(1)->ReshapeLike(X(1));

    if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else {
        LOG(FATAL) << DTypeString(
            X(0), { "float32" }
        );
    }
}

DEPLOY_CPU(PRelu);
#ifdef WITH_CUDA
DEPLOY_CUDA(PRelu);
#endif

DEPLOY_CPU(PReluGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(PReluGradient);
#endif

OPERATOR_SCHEMA(PRelu)
     /* X, W */
    .NumInputs(2)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(PReluGradient)
     /* X, W, dY */
    .NumInputs(3)
     /* dX, dW */
    .NumOutputs(2);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GradientMaker);
    vector<OperatorDef> MakeDef() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), GO(0) }),
            vector<string>({ GI(0), GI(1)} )
        );
    }
};

}  // namespace

REGISTER_GRADIENT(PRelu, GradientMaker);

}  // namespace dragon