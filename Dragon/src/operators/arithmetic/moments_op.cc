#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/arithmetic/moments_op.h"

namespace dragon {

template <class Context>
template <typename Tx, typename Ty>
void MomentsOp<Context>::RunImpl() {
    dims_ = X(0).dims(); axes32_.clear();
    dims32_.assign(dims_.begin(), dims_.end());
    axes32_.assign(axes_.begin(), axes_.end());

    if (axes32_.empty()) {
        // Reduce to a Scalar if missing axes
        for (int i = 0; i < X(0).ndim(); ++i)
            axes32_.push_back(i);
    }

    for (int i = 0; i < axes32_.size(); i++) {
        auto axis = axes32_[i];
        axes32_[i] = axis < 0 ? axis + X(0).ndim() : axis;
        CHECK(axes32_[i] >= 0 && axes32_[i] < X(0).ndim()) \
            << "\nExcepted the axis in [-" << X(0).ndim()
            << ", " << X(0).ndim() << "), got " << axis << ".";
        dims_[axes32_[i]] = 1;
    }

    vec64_t out_shape;
    for (const auto& dim : dims_) {
        if (dim != 1 || keep_dims_)
            out_shape.push_back(dim);
    }

    Y(0)->Reshape(out_shape);
    Y(1)->Reshape(out_shape);

    auto* x = X(0).template data<Tx, Context>();
    auto* mean = Y(0)->template mutable_data<Ty, Context>();
    auto* var = Y(1)->template mutable_data<Ty, Context>();

    if (X(0).count() == 1) {
        kernel::TypeA2B(
            Y(0)->count(),
            x, mean, ctx()
        );
        math::Set(
            Y(0)->count(),
            cast::to<Ty>(0.f),
            var, ctx()
        );
    } else {
        kernel::Moments(
            (int)dims32_.size(), dims32_.data(),
            (int)axes32_.size(), axes32_.data(),
            x, mean, var, ctx()
        );
    }
}

template <class Context>
void MomentsOp<Context>::RunOnDevice() {
    if (XIsType(X(0), int8_t)) {
        RunImpl<int8_t, float>();
    } else if (XIsType(X(0), uint8_t)) {
        RunImpl<uint8_t, float>();
    } else if (XIsType(X(0), int)) {
        RunImpl<int, float>();
    } else if (XIsType(X(0), int64_t)) {
        RunImpl<int64_t, float>();
    } else if (XIsType(X(0), float16)) {
        RunImpl<float16, float>();
    } else if (XIsType(X(0), float)) {
        RunImpl<float, float>();
    } else if (XIsType(X(0), double)) {
        RunImpl<double, double>();
    } else {
        LOG(FATAL) << DTypeString(X(0), {
            "int8", "uint8", "int32", "int64",
            "float16", "float32", "float64",
        });
    }
}

DEPLOY_CPU(Moments);
#ifdef WITH_CUDA
DEPLOY_CUDA(Moments);
#endif

OPERATOR_SCHEMA(Moments)
     /* X */
    .NumInputs(1)
     /* Mean, Var */
    .NumOutputs(2);

NO_GRADIENT(Moments);

}  // namespace dragon