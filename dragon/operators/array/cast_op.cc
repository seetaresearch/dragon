#include "dragon/operators/array/cast_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

#define ELIGIBLE_TENSOR_TYPES \
  { "bool", "int8", "uint8", "int32", "int64", "float16", "float32", "float64" }

#define DISPATCH_TYPE_TO(InputT, OutputT)                                    \
  if (dtype() == types::to_string<OutputT>()) {                              \
    if (InputSize() != 0) {                                                  \
      Output(0)->ReshapeLike(Input(0));                                      \
      auto* x = Input(0).template data<InputT, Context>();                   \
      auto* y = Output(0)->template mutable_data<OutputT, Context>();        \
      math::Cast(Input(0).count(), x, y, ctx());                             \
    } else {                                                                 \
      auto n = Output(0)->count();                                           \
      auto* x = Output(0)->template data<InputT, Context>();                 \
      auto* y = ctx()->workspace()->template data<OutputT, Context>({n})[0]; \
      math::Cast(n, x, y, ctx());                                            \
      ctx()->FinishDeviceComputation();                                      \
      auto* z = Output(0)->template mutable_data<OutputT, Context>();        \
      math::Copy(n, y, z, ctx());                                            \
    }                                                                        \
    return;                                                                  \
  }

#define DISPATCH_TYPE_TO_ALL(InputT) \
  DISPATCH_TYPE_TO(InputT, bool);    \
  DISPATCH_TYPE_TO(InputT, int8_t);  \
  DISPATCH_TYPE_TO(InputT, uint8_t); \
  DISPATCH_TYPE_TO(InputT, int);     \
  DISPATCH_TYPE_TO(InputT, int64_t); \
  DISPATCH_TYPE_TO(InputT, float16); \
  DISPATCH_TYPE_TO(InputT, float);   \
  DISPATCH_TYPE_TO(InputT, double);  \
  LOG(FATAL) << MessageForUnsupported(dtype(), ELIGIBLE_TENSOR_TYPES);

#define DISPATCH_WITH_TENSOR(X)                             \
  if (X.template IsType<bool>()) {                          \
    DISPATCH_TYPE_TO_ALL(bool);                             \
  } else if (X.template IsType<int8_t>()) {                 \
    DISPATCH_TYPE_TO_ALL(int8_t);                           \
  } else if (X.template IsType<uint8_t>()) {                \
    DISPATCH_TYPE_TO_ALL(uint8_t);                          \
  } else if (X.template IsType<int>()) {                    \
    DISPATCH_TYPE_TO_ALL(int);                              \
  } else if (X.template IsType<int64_t>()) {                \
    DISPATCH_TYPE_TO_ALL(int64_t);                          \
  } else if (X.template IsType<float16>()) {                \
    DISPATCH_TYPE_TO_ALL(float16);                          \
  } else if (X.template IsType<float>()) {                  \
    DISPATCH_TYPE_TO_ALL(float);                            \
  } else if (X.template IsType<double>()) {                 \
    DISPATCH_TYPE_TO_ALL(double);                           \
  } else {                                                  \
    LOG(FATAL) << MessageForUnsupported(                    \
        types::to_string(X.meta()), ELIGIBLE_TENSOR_TYPES); \
  }

template <class Context>
void CastOp<Context>::RunOnDevice() {
  if (InputSize() > 0) {
    STORE_INPUT_SPEC(0);
    DISPATCH_WITH_TENSOR(Input(0));
  } else {
    Buffer("X_spec:0")->ReshapeLike(*Output(0))->set_meta(Output(0)->meta());
    DISPATCH_WITH_TENSOR((*Output(0)));
  };
}

template <class Context>
void CastGradientOp<Context>::RunOnDevice() {
  auto& X = RESTORE_INPUT_SPEC(0);
  this->dtype_ = types::to_string(X.meta());
  DISPATCH_WITH_TENSOR(Input(-1));
}

DEPLOY_CPU_OPERATOR(Cast);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Cast);
#endif

DEPLOY_CPU_OPERATOR(CastGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(CastGradient);
#endif

OPERATOR_SCHEMA(Cast)
    /* X */
    .NumInputs(0, 1)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(CastGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(Cast, SimpleGradientMaker);

#undef ELIGIBLE_TENSOR_TYPES
#undef DISPATCH_TYPE_TO
#undef DISPATCH_TYPE_TO_ALL
#undef DISPATCH_WITH_TENSOR

} // namespace dragon
