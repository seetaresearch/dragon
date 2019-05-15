#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/vision/nn_resize_op.h"

namespace dragon {

template <class Context> template <typename T>
void NNResizeOp<Context>::RunImpl() {
    if (data_format() == "NCHW") {
        n_ = X(0).dim(0), c_ = X(0).dim(1);
        h_ = X(0).dim(2), w_ = X(0).dim(3);
        out_h_ = Y(0)->dim(2);
        out_w_ = Y(0)->dim(3);
    } else if (data_format() == "NHWC") {
        n_ = X(0).dim(0), h_ = X(0).dim(1);
        w_ = X(0).dim(2), c_ = X(0).dim(3);
        out_h_ = Y(0)->dim(1);
        out_w_ = Y(0)->dim(2);
    }

    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    kernel::NNResize(
        n_, c_, h_, w_,
        out_h_, out_w_,
        data_format(),
        x, y, ctx()
    );
}

template <class Context>
void NNResizeOp<Context>::RunOnDevice() {
    auto out_shape = X(0).dims();
    if (GET_ARGS_SIZE(dsize) > 0) {
        for (int i = 0; i < 2; i++)
            out_shape[axis_ + i] = dsize(i);
    } else if (!shape_desc_.empty()) {
        auto* sl = ws()->GetTensor(shape_desc_);
        for (int i = 0; i < 2; i++)
            out_shape[axis_ + i] = sl->dim(axis_ + i);
    } else {
        CHECK(fy_ != -1.f && fx_ != -1.f)
                << "\nThe fx and fy should be set.";
        out_shape[axis_] = int(out_shape[axis_] * fy_);
        out_shape[axis_ + 1] = int(out_shape[axis_ + 1] * fx_);
    }

    Y(0)->Reshape(out_shape);

    DispatchHelper<TensorTypes
        <float, float16>>::Call(this, X(0));
}

template <class Context> template <typename T>
void NNResizeGradientOp<Context>::RunImpl() {
    if (data_format() == "NCHW") {
        n_ = X(0).dim(0), c_ = X(0).dim(1);
        h_ = X(0).dim(2), w_ = X(0).dim(3);
        out_h_ = X(-1).dim(2);
        out_w_ = X(-1).dim(3);
    } else if (data_format() == "NHWC") {
        n_ = X(0).dim(0), h_ = X(0).dim(1);
        w_ = X(0).dim(2), c_ = X(0).dim(3);
        out_h_ = X(-1).dim(1);
        out_w_ = X(-1).dim(2);
    }

    auto* dy = X(-1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

    math::Set(
        Y(0)->count(),
        cast::to<T>(0.f),
        dx, ctx()
    );

    kernel::NNResizeGrad(
        n_, c_, h_, w_,
        out_h_, out_w_,
        data_format(),
        dy, dx, ctx()
    );
}

template <class Context>
void NNResizeGradientOp<Context>::RunImplFloat16() {
    if (data_format() == "NCHW") {
        n_ = X(0).dim(0), c_ = X(0).dim(1);
        h_ = X(0).dim(2), w_ = X(0).dim(3);
        out_h_ = X(-1).dim(2);
        out_w_ = X(-1).dim(3);
    } else if (data_format() == "NHWC") {
        n_ = X(0).dim(0), h_ = X(0).dim(1);
        w_ = X(0).dim(2), c_ = X(0).dim(3);
        out_h_ = X(-1).dim(1);
        out_w_ = X(-1).dim(2);
    }

    auto* dy = X(-1).template data<float16, Context>();
    auto* dx = Y(0)->template mutable_data<float16, Context>();

    auto buf = ws()->template data<float, Context>({
        X(-1).count(), Y(0)->count() });

    math::Set(
        Y(0)->count(),
        0.f,
        buf[1], ctx()
    );

    kernel::TypeA2B(
        X(-1).count(),
        dy, buf[0], ctx()
    );

    kernel::NNResizeGrad(
        n_, c_, h_, w_,
        out_h_, out_w_,
        data_format(),
        buf[0], buf[1], ctx()
    );

    kernel::TypeA2B(
        Y(0)->count(),
        buf[1], dx, ctx()
    );
}

template <class Context>
void NNResizeGradientOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));
    
    if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else if (XIsType(X(0), float16)) {
        RunImplFloat16();
    } else {
        LOG(FATAL) << DTypeString(X(0),
            { "float32", "float16" }
        );
    }
}

DEPLOY_CPU(NNResize);
#ifdef WITH_CUDA
DEPLOY_CUDA(NNResize);
#endif

DEPLOY_CPU(NNResizeGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(NNResizeGradient);
#endif

OPERATOR_SCHEMA(NNResize)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(NNResizeGradient)
     /* X, dY */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(NNResize, SimpleGradientMaker);

}  // namespace dragon