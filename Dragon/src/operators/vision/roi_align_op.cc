#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/vision/roi_align_op.h"

namespace dragon {

template <class Context> template <typename T>
void ROIAlignOp<Context>::RunImpl() {
    auto* x   = X(0).template data<T, Context>();
    auto* roi = X(1).template data<float, Context>();
    auto* y   = Y(0)->template mutable_data<T, Context>();

    kernel::ROIAlign(
        X(0).dim(1),
        X(0).dim(2),
        X(0).dim(3),
        pool_h_, pool_w_,
        X(1).dim(0),
        spatial_scale_,
        sampling_ratio_,
        x, roi,
        y, ctx()
    );
}

template <class Context>
void ROIAlignOp<Context>::RunOnDevice() {
    Y(0)->Reshape({
        X(1).dim(0),  /*   nRoIs   */
        X(0).dim(1),  /* nchannels */
        pool_h_,      /* feature_h */
        pool_w_       /* feature_w */
    });

    if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else if (XIsType(X(0), float16)) {
        RunImpl<float16>();
    } else {
        LOG(FATAL) << DTypeString(X(0),
            { "float32", "float16" }
        );
    };
}

template <class Context> template <typename T>
void ROIAlignGradientOp<Context>::RunImpl() {
    auto* roi = X(1).template data<float, Context>();
    auto* dy  = X(2).template data<T, Context>();
    auto* dx  = Y(0)->template mutable_data<T, Context>();

    math::Set(
        Y(0)->count(),
        cast::to<T>(0.f),
        dx, ctx()
    );

    kernel::ROIAlignGrad(
        Y(0)->dim(1),
        Y(0)->dim(2),
        Y(0)->dim(3),
        pool_h_, pool_w_,
        X(1).dim(0),
        spatial_scale_,
        sampling_ratio_,
        dy, roi,
        dx, ctx()
    );
}

template <class Context>
void ROIAlignGradientOp<Context>::RunImplFloat16() {
    auto* roi = X(1).template data<float, Context>();
    auto* dy  = X(2).template data<float16, Context>();
    auto* dx  = Y(0)->template mutable_data<float16, Context>();

    auto buf = ws()
        ->template data<float, Context>
            ({ X(2).count(), Y(0)->count() });

    math::Set(
        Y(0)->count(),
        0.f,
        buf[1], ctx()
    );

    kernel::TypeA2B(
        X(2).count(),
        dy, buf[0], ctx()
    );

    kernel::ROIAlignGrad(
        Y(0)->dim(1),
        Y(0)->dim(2),
        Y(0)->dim(3),
        pool_h_, pool_w_,
        X(1).dim(0),
        spatial_scale_,
        sampling_ratio_,
        buf[0], roi,
        buf[1], ctx()
    );

    kernel::TypeA2B(
        Y(0)->count(),
        buf[1], dx, ctx()
    );
}

template <class Context>
void ROIAlignGradientOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else if (XIsType(X(0), float16)) {
        RunImplFloat16();
    } else {
        LOG(FATAL) << DTypeString(X(0),
            { "float32", "float16" }
        );
    };
}

DEPLOY_CPU(ROIAlign);
#ifdef WITH_CUDA
DEPLOY_CUDA(ROIAlign);
#endif

DEPLOY_CPU(ROIAlignGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ROIAlignGradient);
#endif

OPERATOR_SCHEMA(ROIAlign)
     /* X, RoI */
    .NumInputs(2)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(ROIAlignGradient)
     /* X, RoI, dY */
    .NumInputs(3)
     /* dX */
    .NumOutputs(1);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GradientMaker);
    vector<OperatorDef> MakeDef() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), GO(0) }),
            vector<string>({ GI(0) })
        );
    }
};

}  // namespace

REGISTER_GRADIENT(ROIAlign, GradientMaker);

}  // namespace dragon