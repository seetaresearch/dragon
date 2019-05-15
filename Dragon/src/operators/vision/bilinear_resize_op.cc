#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/vision/bilinear_resize_op.h"

namespace dragon {

template <class Context> template <typename T>
void BilinearResizeOp<Context>::RunImpl() {
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

    kernel::BilinearResize(
        n_, c_, h_, w_, out_h_, out_w_,
        data_format(), x, y, ctx()
    );
}

template <class Context>
void BilinearResizeOp<Context>::RunOnDevice() {
    auto out_shape = X(0).dims();
    if (GET_ARGS_SIZE(dsize) > 0) {
        for (int i = 0; i < 2; i++)
            out_shape[axis_ + i] = dsize(i);
    } else if (!shape_desc_.empty()) {
        auto* sl = ws()->GetTensor(shape_desc_);
        for (int i = 0; i < 2; i++)
            out_shape[axis_ + i] =
                sl->dim(axis_ + i);
    } else {
        CHECK(fy_ != -1.f && fx_ != -1.f)
            << "\nThe fx and fy should be set.";
        out_shape[axis_] = (int64_t)(
            out_shape[axis_] * fy_);
        out_shape[axis_ + 1] = (int64_t)(
            out_shape[axis_ + 1] * fx_);
    }

    Y(0)->Reshape(out_shape);

    DispatchHelper<TensorTypes
        <float>>::Call(this, X(0));
}

template <class Context> template <typename T>
void BilinearResizeGradientOp<Context>::RunImpl() {
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

    kernel::BilinearResizeGrad(
        n_, c_, h_, w_, out_h_, out_w_,
        data_format(), dy, dx, ctx()
    );
}

template <class Context>
void BilinearResizeGradientOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <float>>::Call(this, X(0));
}

DEPLOY_CPU(BilinearResize);
#ifdef WITH_CUDA
DEPLOY_CUDA(BilinearResize);
#endif

DEPLOY_CPU(BilinearResizeGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(BilinearResizeGradient);
#endif

OPERATOR_SCHEMA(BilinearResize)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(BilinearResizeGradient)
     /* X, dY */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(BilinearResize, SimpleGradientMaker);

}  // namespace dragon