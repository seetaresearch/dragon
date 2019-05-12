#include "core/workspace.h"
#include "operators/loss/ctc_loss_op.h"

#ifdef WITH_CUDNN

#if CUDNN_VERSION_MIN(7, 0, 0)

#define CUDNN_LABEL_LENGTH_LIMIT 256

namespace dragon {

template <class Context>
void CuDNNCTCLossOp<Context>::Reshape() {
    auto max_seq_len = X(0).dim(0);
    auto batch_size = X(0).dim(1);
    auto max_num_labels = X(1).dim(1);
    CHECK_EQ(batch_size, X(1).dim(0))
        << "\nExcepted " << batch_size
        << " groups(i.e. batch_size) of labels,"
        << "\nbut got " << X(1).dim(0) << ".";
    // CuDNN currently does not support variable input lengths
    packed_labels_.clear();
    label_lengths_.resize(batch_size);
    input_lengths_.resize(batch_size, max_seq_len);
    auto* labels = X(1).template data<int, CPUContext>();
    for (int n = 0; n < batch_size; ++n) {
        auto start = labels + n * max_num_labels;
        auto res = std::find(
            start, start + max_num_labels,
                (int)padding_mask_);
        int len = (int)std::distance(start, res);
        CHECK_LE(len, CUDNN_LABEL_LENGTH_LIMIT)
            << "\nThe max label length is "
            << CUDNN_LABEL_LENGTH_LIMIT
            << ", but got " << len << ".";
        std::copy(start, start + len,
            std::back_inserter(packed_labels_));
        label_lengths_[n] = len;
    }
    Y(0)->Reshape({});
}

template <class Context> template <typename T>
void CuDNNCTCLossOp<Context>::RunImpl() {
    CuDNNSetTensorDesc<T>(&prob_desc_, X(0).dims());
    CuDNNSetTensorDesc<T>(&grad_desc_, X(0).dims());

    CUDNN_CHECK(cudnnGetCTCLossWorkspaceSize(
        ctx()->cudnn_handle(),
        prob_desc_,
        grad_desc_,
        packed_labels_.data(),
        label_lengths_.data(),
        input_lengths_.data(),
        ctc_algo_,
        ctc_desc_,
        &workspace_size_
    ));

    auto* scratch = (uint8_t*)ws()
        ->template data<Context>
            ({ workspace_size_ })[0];

    auto* g = ws()
        ->CreateTensor(unique_name("grad"))
        ->ReshapeLike(X(0))
        ->template mutable_data<T, Context>();

    auto* p = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    CUDNN_CHECK(cudnnCTCLoss(
        ctx()->cudnn_handle(),
        prob_desc_, p,
        packed_labels_.data(),
        label_lengths_.data(),
        input_lengths_.data(),
        y,
        grad_desc_, g,
        ctc_algo_,
        ctc_desc_,
        scratch, workspace_size_
    ));
}

template <class Context>
void CuDNNCTCLossOp<Context>::RunOnDevice() {
    Reshape();

    if (XIsType(X(0), float)) {
        CUDNN_CHECK(cudnnSetCTCLossDescriptor(
            ctc_desc_, CUDNN_DATA_FLOAT));
        RunImpl<float>();
    } else {
        LOG(FATAL) << DTypeString(
            X(0), { "float32" }
        );
    }
}

DEPLOY_CUDNN(CTCLoss);

}  // namespace dragon

#endif  // CUDNN_VERSION_MIN(7, 0, 0)

#endif  // WITH_CUDNN