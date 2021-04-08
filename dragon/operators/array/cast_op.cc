#include "dragon/operators/array/cast_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename InputT, typename OutputT>
bool CastOp<Context>::MaybeConvert() {
  auto &X = Input(0), *Y = Output(0);
  if (dtypes::to_string(X.meta()) == dtype()) {
    Y->ReshapeLike(X)->CopyFrom(X, ctx());
    return true;
  }
  if (dtypes::to_string<OutputT>() == dtype()) {
    const auto N = X.count();
    if ((void*)&X != (void*)Y) {
      math::Cast(
          N,
          X.template data<InputT, Context>(),
          Y->ReshapeLike(X)->template mutable_data<OutputT, Context>(),
          ctx());
    } else {
      auto* data = ctx()->workspace()->template data<OutputT, Context>({N})[0];
      math::Cast(N, X.template data<InputT, Context>(), data, ctx());
      math::Copy(
          N,
          data,
          Y->ReshapeLike(X)->template mutable_data<OutputT, Context>(),
          ctx());
    }
    return true;
  }
  return false;
}

template <class Context>
template <typename T>
void CastOp<Context>::DoRunWithType() {
  if (MaybeConvert<T, bool>()) {
  } else if (MaybeConvert<T, uint8_t>()) {
  } else if (MaybeConvert<T, int8_t>()) {
  } else if (MaybeConvert<T, int>()) {
  } else if (MaybeConvert<T, int64_t>()) {
  } else if (MaybeConvert<T, float16>()) {
  } else if (MaybeConvert<T, float>()) {
  } else if (MaybeConvert<T, double>()) {
  } else {
    LOG(FATAL) << MessageForUnsupported(
        dtype(),
        {"bool",
         "uint8",
         "int8",
         "int32",
         "int64",
         "float16",
         "float32",
         "float64"});
  }
}

template <class Context>
void CastOp<Context>::RunOnDevice() {
  SET_INPUT_SPEC(0);
  DispatchHelper<dtypes::Generic>::Call(this, Input(0));
}

template <class Context>
void CastGradientOp<Context>::RunOnDevice() {
  auto& X_ref = INPUT_SPEC(0);
  this->dtype_ = dtypes::to_string(X_ref.meta());
  DispatchHelper<dtypes::Generic>::Call(this, Input(0));
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
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(CastGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1)
    /* dY => dX */
    .AllowInplace({{0, 0}});

REGISTER_GRADIENT(Cast, SimpleGradientMaker);

} // namespace dragon
