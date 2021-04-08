#include "dragon/operators/recurrent/rnn_param_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void RNNParamSetOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  int64_t offset = -1, size = -1;
  int64_t w_count = 0, b_count = 0;
  for (int i = 0; i < num_layers_; ++i) {
    const auto dim_in = i == 0 ? input_size_ : hidden_size_ * num_directions_;
    const auto dim_out = hidden_size_;
    for (int j = 0; j < num_directions_; ++j) {
      const auto layer_id = i * num_directions_ + j;
      for (int param_id = 0; param_id < num_params_; ++param_id) {
        if (layer_id == layer_id_ && param_id == param_id_) {
          offset = (param_type_ == "matrix" ? w_count : b_count);
        }
        w_count += dim_out * (param_id < spliter_ ? dim_in : dim_out);
        b_count += dim_out;
        if (layer_id == layer_id_ && param_id == param_id_) {
          size = (param_type_ == "matrix" ? w_count : b_count) - offset;
          break;
        }
      }
    }
  }
  CHECK_EQ(size, X.count()) << "\nExcepted the size of param is " << size
                            << ", but got " << X.count();
  offset += param_type_ == "bias" ? w_count : 0;

  math::Copy(
      size,
      X.template data<T, Context>(),
      Y->template mutable_data<T, Context>() + offset,
      ctx());
}

template <class Context>
void RNNParamSetOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(RNNParamSet);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(RNNParamSet);
#endif

OPERATOR_SCHEMA(RNNParamSet)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

NO_GRADIENT(RNNParamSet);

} // namespace dragon
