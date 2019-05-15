#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "operators/array/stack_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGS(X) \
    axis_ = OpArg<int64_t>("axis", 0); \
    axis_ = axis_ < 0 ? axis_ + X.ndim() + 1 : axis_; \
    CHECK(axis_ >= 0 && axis_ <= X.ndim()) \
        << "\nExcepted the axis in [-" << X.ndim() + 1 \
        << ", " << X.ndim() << "], got " \
        << OpArg<int64_t>("axis", 0) << ".";

template <class Context> template <typename T>
void StackOp<Context>::RunImpl() {
    auto* y = Y(0)->template mutable_data<T, Context>();

    for (int i = 0; i < XSize(); i++) {
        auto* x = X(i).template data<T, Context>();
        kernel::Concat(
            outer_dim_,
            inner_dim_,
            1, XSize(), i,
            x, y, ctx()
        );
    }
}

template <class Context>
void StackOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));

    for (int i = 1; i < XSize(); i++) {
        CHECK_EQ(X(0).ndim(), X(i).ndim())
            << "\nAll inputs should have the same ndim.";
        for (int j = 0; j < X(0).ndim(); j++)
            CHECK_EQ(X(0).dim(j), X(i).dim(j))
                << "\nAll inputs should have the same dims.";
    }

    outer_dim_ = X(0).count(0, axis_);
    inner_dim_ = X(0).count(axis_);

    auto out_shape = X(0).dims();

    out_shape.insert(
        out_shape.begin() +
        axis_, XSize()
    );

    Y(0)->Reshape(out_shape);

    DispatchHelper<TensorTypes
        <bool, int8_t, uint8_t, int, int64_t,
               float16, float, double>
    >::Call(this, X(0));
}

template <class Context> template <typename T>
void StackGradientOp<Context>::RunImpl() {
    auto* dy = X(-1).template data<T, Context>();

    for (int i = 0; i < YSize(); i++) {
        if (Y(i)->name() != "NULL") {
            auto* dx = Y(i)->template
                mutable_data<T, Context>();
            kernel::Slice(
                outer_dim_,
                inner_dim_,
                YSize(), 1, i,
                dy, dx, ctx()
            );
        }
    }
}

template <class Context>
void StackGradientOp<Context>::RunOnDevice() {
    if (X(-1).name() == "NULL") return;
    DETERMINE_RUNTIME_ARGS(X(-1));

    outer_dim_ = X(0).count(0, axis_);
    inner_dim_ = X(0).count(axis_);

    for (int i = 0; i < YSize(); i++)
        Y(i)->ReshapeLike(X(i));

    DispatchHelper<TensorTypes
        <bool, int8_t, uint8_t, int, int64_t,
               float16, float, double>
    >::Call(this, X(0));
}

DEPLOY_CPU(Stack);
#ifdef WITH_CUDA
DEPLOY_CUDA(Stack);
#endif

DEPLOY_CPU(StackGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(StackGradient);
#endif

OPERATOR_SCHEMA(Stack)
     /* X(0), ... */
    .NumInputs(1, INT_MAX)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(StackGradient)
     /* X(0), ..., dY */
    .NumInputs(2, INT_MAX)
     /* dX(0), ... */
    .NumOutputs(1, INT_MAX);

REGISTER_GRADIENT(Stack, SimpleGradientMaker);

#undef DETERMINE_RUNTIME_ARGS

}  // namespace dragon