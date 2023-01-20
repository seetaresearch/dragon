#ifdef USE_CUDNN

#include "dragon/core/workspace.h"
#include "dragon/operators/loss/ctc_loss_op.h"

#define CUDNN_LABEL_LENGTH_LIMIT 256

namespace dragon {

template <class Context>
void CuDNNCTCLossOp<Context>::Reshape() {
  auto &X = Input(0), &Y = Input(1);
  const auto max_seq_len = X.dim(0);
  const auto batch_size = X.dim(1);
  const auto max_num_labels = Y.dim(1);
  CHECK_EQ(batch_size, Y.dim(0))
      << "\nExcepted " << batch_size << " groups(i.e. batch_size) of labels,"
      << "\nbut got " << Input(1).dim(0) << ".";
  // CuDNN currently does not support variable input lengths.
  packed_labels_.clear();
  label_lengths_.resize(batch_size);
  input_lengths_.resize(batch_size, max_seq_len);
  auto* labels = Input(1).template data<int, CPUContext>();
  for (int n = 0; n < batch_size; ++n) {
    auto start = labels + n * max_num_labels;
    auto res = std::find(start, start + max_num_labels, (int)padding_mask_);
    auto len = int(std::distance(start, res));
    CHECK_LE(len, CUDNN_LABEL_LENGTH_LIMIT)
        << "\nThe max label length is " << CUDNN_LABEL_LENGTH_LIMIT
        << ", but got " << len << ".";
    std::copy(start, start + len, std::back_inserter(packed_labels_));
    label_lengths_[n] = len;
  }
}

template <class Context>
template <typename T>
void CuDNNCTCLossOp<Context>::DoRunWithType() {
  auto &X = Input(0), *L = Output(0)->Reshape({});
  CuDNNSetTensorDesc<T>(prob_desc_, X.dims());
  CuDNNSetTensorDesc<T>(grad_desc_, X.dims());

  CUDNN_CHECK(cudnnGetCTCLossWorkspaceSize(
      ctx()->cudnn_handle(),
      prob_desc_,
      grad_desc_,
      packed_labels_.data(),
      label_lengths_.data(),
      input_lengths_.data(),
      ctc_algo_,
      ctc_desc_,
      &workspace_size_));

  CUDNN_CHECK(cudnnCTCLoss(
      ctx()->cudnn_handle(),
      prob_desc_,
      X.template data<T, Context>(),
      packed_labels_.data(),
      label_lengths_.data(),
      input_lengths_.data(),
      L->template mutable_data<T, Context>(),
      grad_desc_,
      Output("X_grad")->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctc_algo_,
      ctc_desc_,
      ctx()->workspace()->template data<Context>(workspace_size_),
      workspace_size_));
}

template <class Context>
void CuDNNCTCLossOp<Context>::RunOnDevice() {
  Reshape();

  if (Input(0).template IsType<float>()) {
    CUDNN_CHECK(cudnnSetCTCLossDescriptor(ctc_desc_, CUDNN_DATA_FLOAT));
    DoRunWithType<float>();
  } else {
    LOG(FATAL) << MessageForUnsupported(
        dtypes::to_string(Input(0).meta()), {"float32"});
  }
}

DEPLOY_CUDNN_OPERATOR(CTCLoss);

} // namespace dragon

#endif // USE_CUDNN
