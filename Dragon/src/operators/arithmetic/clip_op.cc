#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/arithmetic/clip_op.h"

namespace dragon {

#define DEFINE_TYPED_CALLER \
    lowT = low, highT = high; \
    if (XIsType(Input(0), int8_t)) { \
        lowT = std::max(low, -128.f); \
        highT = std::min(high, 127.f); \
        RunWithType<int8_t>(); \
    } else if (XIsType(Input(0), uint8_t)) { \
        lowT = std::max(low, 0.f); \
        highT = std::min(high, 255.f); \
        RunWithType<uint8_t>(); \
    } else if (XIsType(Input(0), int)) { \
        /* Careful bounds for float32 -> int32 */ \
        lowT = std::max(low, -214748e4f); \
        highT = std::min(high, 214748e4f); \
        RunWithType<int>(); \
    } else if (XIsType(Input(0), int64_t)) { \
        /* Careful bounds for float32 -> int64 */ \
        lowT = std::max(low, -922337e13f); \
        highT = std::min(high, 922337e13f); \
        RunWithType<int64_t>(); \
    } else if (XIsType(Input(0), float16)) { \
        lowT = std::max(low, -65505.f); \
        highT = std::min(high, 65504.f); \
        RunWithType<float16>(); \
    } else if (XIsType(Input(0), float)) { \
        RunWithType<float>(); \
    } else if (XIsType(Input(0), double)) { \
        RunWithType<double>(); \
    } else LOG(FATAL) << DTypeHelper(Input(0), { \
        "int8", "uint8", "int32", "int64", \
            "float16", "float32", "float64", \
    });

template <class Context> template <typename T>
void ClipOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    kernel::Clip(Output(0)->count(),
        lowT, highT, Xdata, Ydata, ctx());
}

template <class Context>
void ClipOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));
    DEFINE_TYPED_CALLER;
}

DEPLOY_CPU(Clip);
#ifdef WITH_CUDA
DEPLOY_CUDA(Clip);
#endif
OPERATOR_SCHEMA(Clip).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void ClipGradientOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();

    kernel::ClipGrad(Output(0)->count(),
        lowT, highT, Xdata, dYdata, dXdata, ctx());
}

template <class Context>
void ClipGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));
    DEFINE_TYPED_CALLER;
}

DEPLOY_CPU(ClipGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ClipGradient);
#endif

OPERATOR_SCHEMA(ClipGradient)
    .NumInputs(2).NumOutputs(1);

REGISTER_GRADIENT(Clip, SimpleGradientMaker);

#undef DEFINE_TYPED_CALLER

}  // namespace dragon