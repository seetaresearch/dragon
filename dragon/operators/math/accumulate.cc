#include "dragon/core/workspace.h"
#include "dragon/operators/math/accumulate_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void AccumulateOp<Context>::DoRunWithType(Tensor* X, Tensor* Y) {
  CHECK_EQ(X->count(), Y->count());
  auto* x = X->template data<T, Context>();
  auto* y = Y->template mutable_data<T, Context>();
  if (beta_ == 1.f) {
    if (alpha_ == 1.f) {
      math::Add(X->count(), x, y, y, ctx());
    } else {
      math::Axpy(X->count(), alpha_, x, y, ctx());
    }
  } else {
    if (alpha_ == 0.f) {
      math::Scale(X->count(), beta_, y, y, ctx());
    } else {
      math::Axpby(X->count(), alpha_, x, beta_, y, ctx());
    }
  }
}

template <class Context>
void AccumulateOp<Context>::RunOnDevice() {
  for (int i = 0; i < InputSize(); i++) {
    Output(i)->ReshapeLike(Input(i));
    if (XIsType(Input(i), int8_t)) {
      DoRunWithType<int8_t>(&Input(i), Output(i));
    } else if (XIsType(Input(i), uint8_t)) {
      DoRunWithType<uint8_t>(&Input(i), Output(i));
    } else if (XIsType(Input(i), int)) {
      DoRunWithType<int>(&Input(i), Output(i));
    } else if (XIsType(Input(i), int64_t)) {
      DoRunWithType<int64_t>(&Input(i), Output(i));
    } else if (XIsType(Input(i), float16)) {
      DoRunWithType<float16>(&Input(i), Output(i));
    } else if (XIsType(Input(i), float)) {
      DoRunWithType<float>(&Input(i), Output(i));
    } else if (XIsType(Input(i), double)) {
      DoRunWithType<double>(&Input(i), Output(i));
    } else
      LOG(FATAL) << TypeString(
          Input(i),
          {"int8", "uint8", "int32", "int64", "float16", "float32", "float64"});
  }
}

DEPLOY_CPU(Accumulate);
#ifdef USE_CUDA
DEPLOY_CUDA(Accumulate);
#endif

OPERATOR_SCHEMA(Accumulate)
    /* X1, ... */
    .NumInputs(1, INT_MAX)
    /* Y1, ... */
    .NumOutputs(1, INT_MAX);

NO_GRADIENT(Accumulate);

} // namespace dragon
