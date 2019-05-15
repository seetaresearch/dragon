#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/array/argreduce_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGS(X) \
    axis_ = OpArg<int64_t>("axis", INT_MAX); \
    if (axis_ != INT_MAX) { \
        axis_ = axis_ < 0 ? axis_ + X.ndim() : axis_; \
        CHECK(axis_ >= 0 && axis_ < X.ndim()) \
            << "\nExcepted the axis in [-" << X.ndim() \
            << ", " << X.ndim() << "), got " \
            << OpArg<int64_t>("axis", INT_MAX) << "."; \
    }

template <class Context> template <typename T>
void ArgReduceOp<Context>::RunImpl() {
    if (top_k_ != 1) {
        auto* x = X(0).template data<T, CPUContext>();
        auto* i = Y(0)->template mutable_data<int64_t, CPUContext>();
        auto* v = Y(1)->name() == "NULL" ? nullptr :
                  Y(1)->template mutable_data<T, CPUContext>();
        if (operation_ == "ARGMAX") {
            kernel::ArgMax(
                outer_dim_,
                inner_dim_,
                axis_dim_,
                top_k_, x,
                i, v, &cctx_
            );
        } else if (operation_ == "ARGMIN") {
            kernel::ArgMin(
                outer_dim_,
                inner_dim_,
                axis_dim_,
                top_k_, x,
                i, v, &cctx_
            );
        } else {
            LOG(FATAL) << "Unknown Operation: " << operation_;
        }
    } else {
        auto* x = X(0).template data<T, Context>();
        auto* i = Y(0)->template mutable_data<int64_t, Context>();
        auto* v = Y(1)->name() == "NULL" ? nullptr :
                  Y(1)->template mutable_data<T, Context>();
        if (operation_ == "ARGMAX") {
            kernel::ArgMax(
                outer_dim_,
                inner_dim_,
                axis_dim_,
                top_k_, x,
                i, v, ctx()
            );
        } else if (operation_ == "ARGMIN") {
            kernel::ArgMin(
                outer_dim_,
                inner_dim_,
                axis_dim_,
                top_k_, x,
                i, v, ctx()
            );
        } else {
            LOG(FATAL) << "Unknown operation: " << operation_;
        }
    }
}

template <class Context>
void ArgReduceOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));

    if (axis_ != INT_MAX) {
        axis_dim_ = X(0).dim(axis_);
        outer_dim_ = X(0).count(0, axis_);
        inner_dim_ = X(0).count(axis_ + 1);
    } else {
        axis_dim_ = X(0).count();
        outer_dim_ = inner_dim_ = 1;
    }

    auto out_shape = X(0).dims();

    if (!keep_dims_) {
        if (axis_ != INT_MAX) {
            if (top_k_ > 1) {
                out_shape[axis_] = top_k_;
            } else {
                out_shape.erase(out_shape.begin() + axis_);
            }
        } else {
            if (top_k_ > 1) out_shape = { top_k_ };
            else out_shape = {};
        }
    } else {
        if (axis_ != INT_MAX) out_shape[axis_] = top_k_;
        else out_shape = { top_k_ };
    }

    Y(0)->Reshape(out_shape);
    Y(1)->Reshape(out_shape);

    DispatchHelper<TensorTypes
        <bool, int8_t, uint8_t, int, int64_t,
               float16, float, double>
    >::Call(this, X(0));
}

DEPLOY_CPU(ArgReduce);
#ifdef WITH_CUDA
DEPLOY_CUDA(ArgReduce);
#endif

OPERATOR_SCHEMA(ArgReduce)
     /* X */
    .NumInputs(1)
     /* Value, Index */
    .NumOutputs(2);

NO_GRADIENT(ArgReduce);

#undef DETERMINE_RUNTIME_ARGS

}  // namespace dragon