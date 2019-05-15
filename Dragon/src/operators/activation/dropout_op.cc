#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "operators/activation/dropout_op.h"

namespace dragon {

template <class Context> template <typename T>
void DropoutOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();
    auto scale = use_scale_ ? 1.f / (1.f - prob()) : 1.f;

    if (phase() == "TEST") {
        Y(0)->CopyFrom(X(0), ctx());
        if (!use_scale_) {
            math::Scale(
                Y(0)->count(),
                1.f - prob(), y,
                y, ctx()
            );
        }
    } else if (phase() == "TRAIN") {
        auto* mask = ws()
            ->CreateTensor(unique_name("mask"))
            ->ReshapeLike(X(0))
            ->template mutable_data<uint8_t, Context>();

        auto* scratch = ws()
            ->template data<uint32_t, Context>
                ({ Y(0)->count() })[0];

        kernel::Dropout(
            Y(0)->count(),
            prob(), scale,
            x, scratch,
            mask, y, ctx()
        );
    } else {
        LOG(FATAL) << "Unknown Phase: " << phase();
    }
}

template <class Context>
void DropoutOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <float, float16>>::Call(this, X(0));
}

template <class Context> template <typename T>
void DropoutGradientOp<Context>::RunImpl() {
    auto* mask = ws()
        ->GetTensor(unique_name("mask"))
        ->template data<uint8_t, Context>();

    auto* dy = X(1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();
    auto scale = use_scale_ ? 1.f / (1.f - prob()) : 1.f;

    if (phase() == "TEST") {
        NOT_IMPLEMENTED;
    } else if (phase() == "TRAIN") {
        kernel::ApplyMask(
            Y(0)->count(),
            scale,
            dy, mask,
            dx, ctx()
        );
    } else {
        LOG(FATAL) << "Incorrect Op phase: " << phase();
    }
}

template <class Context>
void DropoutGradientOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <float, float16>>::Call(this, X(0));
}

DEPLOY_CPU(Dropout);
#ifdef WITH_CUDA
DEPLOY_CUDA(Dropout);
#endif

DEPLOY_CPU(DropoutGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(DropoutGradient);
#endif

OPERATOR_SCHEMA(Dropout)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1)
     /* X => Y */
    .Inplace({ { 0, 0 } });

OPERATOR_SCHEMA(DropoutGradient)
     /* Y, dY */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1)
     /* dY => dX */
    .Inplace({ { 1, 0 } });

REGISTER_GRADIENT(Dropout, InplaceGradientMaker);

}  // namepsace dragon