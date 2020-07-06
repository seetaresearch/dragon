#include "dragon/operators/array/cast_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

#define ELIGIBLE_TENSOR_TYPES \
  { "bool", "int8", "uint8", "int32", "int64", "float16", "float32", "float64" }

#define DEFINE_TYPE_A_TO_B(Ta, type_str, Tb)                     \
  if (dtype() == type_str) {                                     \
    if (InputSize() != 0) {                                      \
      Output(0)->ReshapeLike(Input(0));                          \
      auto* x = Input(0).template data<Ta, Context>();           \
      auto* y = Output(0)->template mutable_data<Tb, Context>(); \
      kernel::Cast(Input(0).count(), x, y, ctx());               \
    } else {                                                     \
      auto n = Output(0)->count();                               \
      auto* x = Output(0)->template data<Ta, Context>();         \
      auto* scratch = ws()->template data<Tb, Context>({n})[0];  \
      kernel::Cast(n, x, scratch, ctx());                        \
      ctx()->FinishDeviceComputation();                          \
      auto* y = Output(0)->template mutable_data<Tb, Context>(); \
      math::Copy(n, scratch, y, ctx());                          \
    }                                                            \
    return;                                                      \
  }

#define DEFINE_TYPE_A_TO_ALL(Ta)              \
  DEFINE_TYPE_A_TO_B(Ta, "bool", bool);       \
  DEFINE_TYPE_A_TO_B(Ta, "int8", int8_t);     \
  DEFINE_TYPE_A_TO_B(Ta, "uint8", uint8_t);   \
  DEFINE_TYPE_A_TO_B(Ta, "int32", int);       \
  DEFINE_TYPE_A_TO_B(Ta, "int64", int64_t);   \
  DEFINE_TYPE_A_TO_B(Ta, "float16", float16); \
  DEFINE_TYPE_A_TO_B(Ta, "float32", float);   \
  DEFINE_TYPE_A_TO_B(Ta, "float64", double)

#define DISPATCH_WITH_TENSOR(X)                         \
  if (XIsType(X, bool)) {                               \
    DEFINE_TYPE_A_TO_ALL(bool);                         \
  } else if (XIsType(X, int8_t)) {                      \
    DEFINE_TYPE_A_TO_ALL(int8_t);                       \
  } else if (XIsType(X, uint8_t)) {                     \
    DEFINE_TYPE_A_TO_ALL(uint8_t);                      \
  } else if (XIsType(X, int)) {                         \
    DEFINE_TYPE_A_TO_ALL(int);                          \
  } else if (XIsType(X, int64_t)) {                     \
    DEFINE_TYPE_A_TO_ALL(int64_t);                      \
  } else if (XIsType(X, float16)) {                     \
    DEFINE_TYPE_A_TO_ALL(float16);                      \
  } else if (XIsType(X, float)) {                       \
    DEFINE_TYPE_A_TO_ALL(float);                        \
  } else if (XIsType(X, double)) {                      \
    DEFINE_TYPE_A_TO_ALL(double);                       \
  } else {                                              \
    LOG(FATAL) << TypeString(X, ELIGIBLE_TENSOR_TYPES); \
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

DEPLOY_CPU(Cast);
#ifdef USE_CUDA
DEPLOY_CUDA(Cast);
#endif

DEPLOY_CPU(CastGradient);
#ifdef USE_CUDA
DEPLOY_CUDA(CastGradient);
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
#undef DEFINE_TYPE_A_TO_B
#undef DEFINE_TYPE_A_TO_ALL
#undef DISPATCH_WITH_TENSOR

} // namespace dragon
