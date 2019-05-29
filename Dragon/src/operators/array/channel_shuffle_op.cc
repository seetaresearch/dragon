#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "operators/array/channel_shuffle_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGS(X) \
    axis_ = OpArg<int64_t>("axis", 0); \
    axis_ = axis_ < 0 ? axis_ + X.ndim() : axis_; \
    CHECK(axis_ >= 0 && axis_ < X.ndim()) \
        << "\nExcepted the axis in [-" << X.ndim() \
        << ", " << X.ndim() << "), got " \
        << OpArg<int64_t>("axis", 0) << ".";

template <class Context> template <typename T>
void ChannelShuffleOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    kernel::ChannelShuffle(
        outer_dim_,
        inner_dim_,
        axis_dim_,
        group_,
        x, y, ctx()
    );
}

template <class Context>
void ChannelShuffleOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));
    axis_dim_ = X(0).dim(axis_);
    outer_dim_ = X(0).count(0, axis_);
    inner_dim_ = X(0).count(axis_ + 1);

    CHECK_EQ(axis_dim_ % group_, 0)
        << "\nThe " << axis_dim_ << " channels "
        << "can not be split into " << group_ << " groups.";

    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <bool, int8_t, uint8_t, int, int64_t,
               float16, float, double>
    >::Call(this, X(0));
}

template <class Context> template <typename T>
void ChannelShuffleGradientOp<Context>::RunImpl() {
    auto* dy = X(1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

    kernel::ChannelShuffle(
        outer_dim_,
        inner_dim_,
        axis_dim_,
        axis_dim_ / group_,
        dy, dx, ctx()
    );
}

template <class Context>
void ChannelShuffleGradientOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));
    axis_dim_ = X(0).dim(axis_);
    outer_dim_ = X(0).count(0, axis_);
    inner_dim_ = X(0).count(axis_ + 1);

    CHECK_EQ(axis_dim_ % group_, 0)
        << "\nThe " << axis_dim_ << " channels "
        << "can not be split into " << group_ << " groups.";

    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <bool, int8_t, uint8_t, int, int64_t,
               float16, float, double>
    >::Call(this, X(0));
}

DEPLOY_CPU(ChannelShuffle);
#ifdef WITH_CUDA
DEPLOY_CUDA(ChannelShuffle);
#endif

DEPLOY_CPU(ChannelShuffleGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ChannelShuffleGradient);
#endif

OPERATOR_SCHEMA(ChannelShuffle)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(ChannelShuffleGradient)
     /* X, dY */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(ChannelShuffle, SimpleGradientMaker);

#undef DETERMINE_RUNTIME_ARGS

}  // namespace dragon