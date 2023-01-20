#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/sequence/embedding_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLEmbeddingOp<Context>::DoRunWithType() {
  auto &X = Input(0), &X_index = Input(1), *Y = Output(0);
  Output("X_spec")->ReshapeLike(X);

  GET_OP_AXIS_ARG(axis, X.ndim(), -1);
  const auto N = X.count(0, axis), C = X.count(axis), K = X_index.count();
  CHECK_GT(K, 0) << "\nLength of index must > 0.";

  auto Y_dims(X_index.dims());
  Y_dims.push_back(C);

  CNNLSetTensorDesc<T>(input_desc_, {N, C});
  CNNLSetTensorDesc<int>(index_desc_, {K});
  CNNLSetTensorDesc<T>(output_desc_, {K, C});
  CNNL_CHECK(cnnlEmbeddingForward_v2(
      ctx()->cnnl_handle(),
      input_desc_,
      X.template data<T, Context>(),
      index_desc_,
      X_index.template data<int, Context>(),
      padding_index_,
      nullptr,
      nullptr,
      output_desc_,
      Y->Reshape(Y_dims)->template mutable_data<T, Context>()));
}

template <class Context>
template <typename T>
void CNNLEmbeddingGradientOp<Context>::DoRunWithType() {
  auto &X_index = Input(0), &dY = Input(1);
  auto* dX = Output(0)->ReshapeLike(Input("X_spec"));

  GET_OP_AXIS_ARG(axis, dX->ndim(), -1);
  const auto N = dX->count(0, axis), C = dX->count(axis), K = X_index.count();

  CNNLSetTensorDesc<T>(this->input_desc_, {K, C});
  CNNLSetTensorDesc<int>(this->index_desc_, {K});
  CNNLSetTensorDesc<T>(this->output_desc_, {N, C});

  size_t scratch_size = 0;
  CNNL_CHECK(cnnlGetEmbeddingBackwardWorkspaceSize(
      ctx()->cnnl_handle(),
      this->input_desc_,
      this->output_desc_,
      false, // scale_grad_by_freq
      &scratch_size));
  CNNL_CHECK(cnnlEmbeddingBackward(
      ctx()->cnnl_handle(),
      this->padding_index_,
      false, // scale_grad_by_freq
      this->index_desc_,
      X_index.template data<int, Context>(),
      this->input_desc_,
      dY.template data<T, Context>(),
      ctx()->workspace()->template data<Context>(scratch_size),
      scratch_size,
      this->output_desc_,
      dX->template mutable_data<T, Context>()));
}

DEPLOY_CNNL_OPERATOR(Embedding);
DEPLOY_CNNL_OPERATOR(EmbeddingGradient);

} // namespace dragon

#endif // USE_MLU
