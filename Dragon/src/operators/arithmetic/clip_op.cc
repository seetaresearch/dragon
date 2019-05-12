#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/arithmetic/clip_op.h"

namespace dragon {

#define DEFINE_TYPED_RUN_IMPL \
    lowT_ = low_, highT_ = high_; \
    if (XIsType(X(0), int8_t)) { \
        lowT_ = std::max(low_, -128.f); \
        highT_ = std::min(high_, 127.f); \
        RunImpl<int8_t>(); \
    } else if (XIsType(X(0), uint8_t)) { \
        lowT_ = std::max(low_, 0.f); \
        highT_ = std::min(high_, 255.f); \
        RunImpl<uint8_t>(); \
    } else if (XIsType(X(0), int)) { \
        /* Careful bounds for float32 -> int32 */ \
        lowT_ = std::max(low_, -214748e4f); \
        highT_ = std::min(high_, 214748e4f); \
        RunImpl<int>(); \
    } else if (XIsType(X(0), int64_t)) { \
        /* Careful bounds for float32 -> int64 */ \
        lowT_ = std::max(low_, -922337e13f); \
        highT_ = std::min(high_, 922337e13f); \
        RunImpl<int64_t>(); \
    } else if (XIsType(X(0), float16)) { \
        lowT_ = std::max(low_, -65505.f); \
        highT_ = std::min(high_, 65504.f); \
        RunImpl<float16>(); \
    } else if (XIsType(X(0), float)) { \
        RunImpl<float>(); \
    } else if (XIsType(X(0), double)) { \
        RunImpl<double>(); \
    } else LOG(FATAL) << DTypeString(X(0), { \
        "int8", "uint8", "int32", "int64", \
        "float16", "float32", "float64", \
    });

template <class Context> template <typename T>
void ClipOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    kernel::Clip(
        X(0).count(),
        lowT_, highT_,
        x, y, ctx()
    );
}

template <class Context>
void ClipOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));
    DEFINE_TYPED_RUN_IMPL;
}

template <class Context> template <typename T>
void ClipGradientOp<Context>::RunImpl() {
    auto* x  = X(0).template data<T, Context>();
    auto* dy = X(1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

    kernel::ClipGrad(
        X(0).count(),
        lowT_, highT_,
        x, dy,
        dx, ctx()
    );
}

template <class Context>
void ClipGradientOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));
    DEFINE_TYPED_RUN_IMPL;
}

DEPLOY_CPU(Clip);
#ifdef WITH_CUDA
DEPLOY_CUDA(Clip);
#endif

DEPLOY_CPU(ClipGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ClipGradient);
#endif

OPERATOR_SCHEMA(Clip)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(ClipGradient)
     /* X, dY */
    .NumInputs(2)
     /* X, dX */
    .NumOutputs(1);

REGISTER_GRADIENT(Clip, SimpleGradientMaker);

#undef DEFINE_TYPED_RUN_IMPL

}  // namespace dragon