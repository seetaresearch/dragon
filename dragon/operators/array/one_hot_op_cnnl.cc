#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/array/one_hot_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLOneHotOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  vec64_t Y_dims(X.dims());
  Y_dims.push_back(depth_);

  auto* scratch = ctx()->workspace()->template data<T, Context>(2);
  math::Set(1, convert::To<T>(on_value_), scratch, ctx());
  math::Set(1, convert::To<T>(off_value_), scratch + 1, ctx());

  int* index = nullptr;
  if (TypeMeta::Make<T>() != TypeMeta::Make<int>()) {
    index = ctx()->workspace()->template data<int, Context>(
        X.count(), "BufferKernel");
    math::Cast(X.count(), X.template data<T, Context>(), index, ctx());
  } else {
    index = const_cast<int*>(X.template data<int, Context>());
  }

  CNNLSetTensorDesc<int>(input_desc_, X.dims());
  CNNL_CHECK(cnnlOneHot(
      ctx()->cnnl_handle(),
      input_desc_,
      index,
      depth_,
      scratch,
      scratch + 1,
      -1,
      CNNLGetDataType<T>(),
      Y->Reshape(Y_dims)->template mutable_data<T, Context>()));
}

DEPLOY_CNNL_OPERATOR(OneHot);

} // namespace dragon

#endif // USE_MLU
