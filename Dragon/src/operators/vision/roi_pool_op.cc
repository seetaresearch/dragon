#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/vision/roi_pool_op.h"

namespace dragon {

template <class Context> template <typename T>
void ROIPoolOp<Context>::RunImpl() {
    auto* mask = ws()
        ->CreateTensor(unique_name("mask"))
        ->ReshapeLike(*Y(0))
        ->template mutable_data<int, Context>();

    auto* x = X(0).template data<T, Context>();
    auto* roi = X(1).template data<float, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    kernel::ROIPool(
        X(0).dim(1),
        X(0).dim(2),
        X(0).dim(3),
        pool_h_, pool_w_,
        X(1).dim(0),
        spatial_scale_,
        x, roi, mask,
        y, ctx()
    );
}

template <class Context>
void ROIPoolOp<Context>::RunOnDevice() {
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
void ROIPoolGradientOp<Context>::RunImpl() {
    auto* mask = ws()
        ->GetTensor(unique_name("mask"))
        ->template data<int, Context>();

    auto* roi = X(1).template data<float, Context>();
    auto* dy  = X(2).template data<T, Context>();
    auto* dx  = Y(0)->template mutable_data<T, Context>();

    kernel::ROIPoolGrad(
        Y(0)->dim(0),
        Y(0)->dim(1),
        Y(0)->dim(2),
        Y(0)->dim(3),
        pool_h_, pool_w_,
        X(1).dim(0),
        spatial_scale_,
        dy, roi, mask,
        dx, ctx()
    );
}

template <class Context>
void ROIPoolGradientOp<Context>::RunImplFloat16() {
    auto* mask = ws()
        ->GetTensor(unique_name("mask"))
        ->template data<int, Context>();

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

    kernel::ROIPoolGrad(
        Y(0)->dim(0),
        Y(0)->dim(1),
        Y(0)->dim(2),
        Y(0)->dim(3),
        pool_h_, pool_w_,
        X(1).dim(0),
        spatial_scale_,
        buf[0], roi, mask,
        buf[1], ctx()
    );

    kernel::TypeA2B(
        Y(0)->count(),
        buf[1], dx, ctx()
    );
}

template <class Context>
void ROIPoolGradientOp<Context>::RunOnDevice() {
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

DEPLOY_CPU(ROIPool);
#ifdef WITH_CUDA
DEPLOY_CUDA(ROIPool);
#endif

DEPLOY_CPU(ROIPoolGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ROIPoolGradient);
#endif

OPERATOR_SCHEMA(ROIPool)
     /* X, RoI */
    .NumInputs(2)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(ROIPoolGradient)
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

REGISTER_GRADIENT(ROIPool, GradientMaker);

}  // namespace dragon