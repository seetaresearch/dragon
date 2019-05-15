#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/array/reduce_op.h"

namespace dragon {

#define TENSOR_FROM_VEC(tensor, vec, T) \
    { \
        tensor.Reshape({ (int64_t)vec.size() }); \
        auto* data = tensor.template mutable_data<T, CPUContext>(); \
        for (int i = 0; i < vec.size(); i++) data[i] = (T)vec[i]; \
    }

template <class Context> template <typename T>
void ReduceOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    auto scale = operation_ == "SUM" ? 1.f :
        1.f / (X(0).count() / Y(0)->count());

    if (X(0).count() == 1) {
        // Just copy the contents
        math::Copy(Y(0)->count(), x, y, ctx());
    } else {
        kernel::ReduceSum(
            (int)dims32_.size(),
            dims32_.data(),
            (int)axes32_.size(),
            axes32_.data(),
            scale, x,
            y, ctx()
        );
    }
}

template <class Context>
void ReduceOp<Context>::RunOnDevice() {
    dims_ = X(0).dims();
    dims32_.assign(dims_.begin(), dims_.end());
    axes32_.assign(axes_.begin(), axes_.end());

    if (axes32_.empty()) {
        // Reduce to a Scalar if missing axes
        for (int i = 0; i < X(0).ndim(); ++i)
            axes32_.push_back(i);
    }

    for (int i = 0; i < axes32_.size(); i++) {
        int axis = axes32_[i];
        axes32_[i] = axis < 0 ? axis + X(0).ndim() : axis;
        CHECK(axes32_[i] >= 0 && axes32_[i] < X(0).ndim()) \
            << "\nExcepted the axis in [-" << X(0).ndim()
            << ", " << X(0).ndim() << "), got " << axis << ".";
        dims_[axes32_[i]] = 1;
    }

    vec64_t out_shape;
    for (const auto& dim : dims_) {
        if (dim != 1 || keep_dims_)
            out_shape.emplace_back(dim);
    }

    Y(0)->Reshape(out_shape);

    DispatchHelper<TensorTypes
        <int8_t, uint8_t, int, int64_t,
            float16, float, double>
    >::Call(this, X(0));
}

template <class Context> template <typename T>
void ReduceGradientOp<Context>::RunImpl() {
    auto* dy = X(1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

    auto scale = operation_ == "SUM" ? 1.f :
        1.f / (X(0).count() / X(1).count());

    if (X(0).count() == 1) {
        // Just copy the contents
        math::Copy(
            Y(0)->count(),
            dy, dx, ctx()
        );
    } else if (X(-1).count() == 1) {
        // Directly set the dX from a constant Scalar
        T dyHost = X(1).template data<T, CPUContext>()[0];
        dyHost = cast::to<T>(cast::to<float>(dyHost) * scale);
        math::Set(Y(0)->count(), dyHost, dx, ctx());
    } else {
        // We need a unsqueezed strides
        int64_t stride = 1;
        y_strides_.resize(y_dims_.size(), 1);
        for (int i = (int)y_dims_.size() - 1; i >= 0; i--) {
            y_strides_[i] = stride;
            stride *= y_dims_[i];
        }

        TENSOR_FROM_VEC(X_dims_, X(0).dims(), int);
        TENSOR_FROM_VEC(Y_dims_, y_dims_, int);
        TENSOR_FROM_VEC(Y_strides_, y_strides_, int);

        // Apply a simple Nd-Broadcast solution
        kernel::ReduceSumGrad(
            Y(0)->count(),
            Y(0)->ndim(),
            X_dims_.template data<int, Context>(),
            Y_dims_.template data<int, Context>(),
            Y_strides_.template data<int, Context>(),
            scale, dy,
            dx, ctx()
        );
    }
}

template <class Context>
void ReduceGradientOp<Context>::RunOnDevice() {
    y_dims_ = X(0).dims();
    axes32_.assign(axes_.begin(), axes_.end());

    if (axes32_.empty()) {
        // Reduce to a Scalar if missing axes
        for (int i = 0; i < X(0).ndim(); ++i)
            axes32_.push_back(i);
    }

    for (int i = 0; i < axes32_.size(); i++) {
        int axis = axes32_[i];
        axes32_[i] = axis < 0 ? axis + X(0).ndim() : axis;
        CHECK(axes32_[i] >= 0 && axes32_[i] < X(0).ndim()) \
            << "\nExcepted the axis in [-" << X(0).ndim()
            << ", " << X(0).ndim() << "), got " << axis << ".";
        y_dims_[axes32_[i]] = 1;
    }

    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <int8_t, uint8_t, int, int64_t,
            float16, float, double>
    >::Call(this, X(0));
}

DEPLOY_CPU(Reduce);
#ifdef WITH_CUDA
DEPLOY_CUDA(Reduce);
#endif

DEPLOY_CPU(ReduceGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ReduceGradient);
#endif

OPERATOR_SCHEMA(Reduce)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(ReduceGradient)
     /* X, dY */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(Reduce, SimpleGradientMaker);

}  // namespace dragon