#ifdef USE_MLU

#include "dragon/operators/activation/hardswish_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLHardSwishOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});
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

DEPLOY_CNNL_OPERATOR(HardSwish);

} // namespace dragon

#endif // USE_MLU
