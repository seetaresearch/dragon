#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "operators/activation/droppath_op.h"

namespace dragon {

template <class Context> template <typename T>
void DropPathOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    if (phase() == "TEST") {
        math::Copy(
            Y(0)->count(),
            x, y, ctx()
        );
    } else if (phase() == "TRAIN") {
        if (inc_ > 0.f && drop_prob_ < prob()) {
            drop_prob_ += inc_;
        } else {
            drop_prob_ = prob();
        }

        auto* mask = ws()
            ->CreateTensor(unique_name("mask"))
            ->Reshape({ rows_ })
            ->template mutable_data<float, Context>();

        auto* scale = ws()
            ->CreateTensor(unique_name("scale"))
            ->Reshape({})
            ->template mutable_data<float, CPUContext>();

        math::RandomUniform(rows_, 0.f, 1.f, mask, ctx());
        scale[0] = 1.f / (1.f - drop_prob_);

        kernel::DropPath(
            rows_, cols_,
            scale[0],
            x, mask,
            y, ctx()
        );
    } else {
        LOG(FATAL) << "Unknown Phase: " << phase();
    }
}

template <class Context>
void DropPathOp<Context>::RunOnDevice() {
    rows_ = X(0).dim(0);
    cols_  = X(0).stride(0);

    Y(0)->ReshapeLike(X(0));

    if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else if (XIsType(X(0), float16)) {
        RunImpl<float16>();
    } else {
        LOG(FATAL) << DTypeString(X(0),
            { "float32", "float16" }
        );
    }
}

template <class Context> template <typename T>
void DropPathGradientOp<Context>::RunImpl() {
    auto* mask = ws()
        ->GetTensor(unique_name("mask"))
        ->template data<float, Context>();

    auto* scale = ws()
        ->GetTensor(unique_name("scale"))
        ->template data<float, CPUContext>();

    auto* dy = X(1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

    if (phase() == "TEST") {
        NOT_IMPLEMENTED;
    } else if (phase() == "TRAIN") {
        kernel::DropPath(
            rows_, cols_,
            scale[0],
            dy, mask,
            dx, ctx()
        );
    } else {
        LOG(FATAL) << "Unknown Phase: " << phase();
    }
}

template <class Context>
void DropPathGradientOp<Context>::RunOnDevice() {
    rows_ = X(0).dim(0);
    cols_ = X(0).stride(0);

    Y(0)->ReshapeLike(X(0));

    if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else if (XIsType(X(0), float16)) {
        RunImpl<float16>();
    } else {
        LOG(FATAL) << DTypeString(X(0),
            { "float32", "float16" }
        );
    }
}

DEPLOY_CPU(DropPath);
#ifdef WITH_CUDA
DEPLOY_CUDA(DropPath);
#endif

DEPLOY_CPU(DropPathGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(DropPathGradient);
#endif

OPERATOR_SCHEMA(DropPath)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1)
     /* X => Y */
    .Inplace({ { 0, 0 } });

OPERATOR_SCHEMA(DropPathGradient)
     /* Y, dY */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1)
     /* dY => dX */
    .Inplace({ { 1, 0 } });

REGISTER_GRADIENT(DropPath, InplaceGradientMaker);

}  // namepsace dragon