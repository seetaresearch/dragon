#ifdef USE_MLU

#include "dragon/operators/activation/silu_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLSiluOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  CNNLSetTensorDesc<T>(input_desc_, X.dims());
  CNNL_CHECK(cnnlActivationForward(
      ctx()->cnnl_handle(),
      act_desc_,
      CNNLTraits<T>::one,
      input_desc_,
      X.template data<T, Context>(),
      CNNLTraits<T>::zero,
      input_desc_,
      Y->ReshapeLike(X)->template mutable_data<T, Context>()));
}

template <class Context>
template <typename T>
void CNNLSiluGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dY = Input(1), *dX = Output(0);
  CNNLSetTensorDesc<T>(this->input_desc_, X.dims());
  CNNL_CHECK(cnnlActivationBackward(
      ctx()->cnnl_handle(),
      this->act_desc_,
      CNNLTraits<T>::one,
      this->input_desc_,
      X.template data<T, Context>(),
      this->input_desc_,
      dY.template data<T, Context>(),
      this->input_desc_,
      X.template data<T, Context>(),
      CNNLTraits<T>::zero,
      this->input_desc_,
      dX->ReshapeLike(X)->template mutable_data<T, Context>()));
}

DEPLOY_CNNL_OPERATOR(Silu);
DEPLOY_CNNL_OPERATOR(SiluGradient);

} // namespace dragon

#endif // USE_MLU
