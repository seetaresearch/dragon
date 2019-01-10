#include "core/workspace.h"
#include "operators/loss/ctc_loss_op.h"

#ifdef WITH_CUDNN

#if CUDNN_VERSION_MIN(7, 0, 0)

#define CUDNN_LABEL_LENGTH_LIMIT 256

namespace dragon {

template <class Context>
void CuDNNCTCLossOp<Context>::WrapIO() {
    const auto max_seq_len = Input(0).dim(0);
    const auto batch_size = Input(0).dim(1);
    const auto max_num_labels = Input(1).dim(1);
    CHECK_EQ(batch_size, Input(1).dim(0))
        << "\nExcepted " << batch_size
        << " groups(i.e. batch_size) of labels,"
        << "\nbut got " << Input(1).dim(0) << ".";
    // CuDNN currently does not support variable input lengths
    input_lengths = vector<int>(batch_size, max_seq_len);
    packed_labels.clear(); label_lengths.resize(batch_size);
    auto* Ldata = Input(1).template data<int, CPUContext>();
    for (int n = 0; n < batch_size; ++n) {
        auto start = Ldata + n * max_num_labels;
        auto res = std::find(
            start, start + max_num_labels,
                (int)padding_mask);
        int len = (int)std::distance(start, res);
        CHECK_LE(len, CUDNN_LABEL_LENGTH_LIMIT)
            << "\nThe max label length is "
            << CUDNN_LABEL_LENGTH_LIMIT
            << ", but got " << len << ".";
        std::copy(start, start + len,
            std::back_inserter(packed_labels));
        label_lengths[n] = len;
    }
    Output(0)->Reshape({ 1 });
}

template <class Context> template <typename T>
void CuDNNCTCLossOp<Context>::RunWithType() {
    cudnnSetTensorDesc<T>(&prob_desc, Input(0).dims());
    cudnnSetTensorDesc<T>(&grad_desc, Input(0).dims());

    CUDNN_CHECK(cudnnGetCTCLossWorkspaceSize(
        ctx()->cudnn_handle(), prob_desc, grad_desc,
            packed_labels.data(), label_lengths.data(),
                input_lengths.data(),
                    ctc_algo, ctc_desc, &workspace_size));

    auto* Pdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    auto* gradT = ws()->CreateTensor(mount_name("ctc/grads"));
    gradT->ReshapeLike(Input(0));
    auto* Gdata = gradT->template mutable_data<T, Context>();
    auto* WSdata = (uint8_t*)ws()->template
        caches<Context>({ workspace_size })[0];

    CUDNN_CHECK(cudnnCTCLoss(ctx()->cudnn_handle(),
        prob_desc, Pdata, packed_labels.data(),
            label_lengths.data(), input_lengths.data(),
                Ydata, grad_desc, Gdata,
                    ctc_algo, ctc_desc,
                        WSdata, workspace_size));
}

template <class Context>
void CuDNNCTCLossOp<Context>::RunOnDevice() {
    WrapIO();

    if (XIsType(Input(0), float)) {
        CUDNN_CHECK(cudnnSetCTCLossDescriptor(
            ctc_desc, CUDNN_DATA_FLOAT));
        RunWithType<float>();
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CUDNN(CTCLoss);

}  // namespace dragon

#endif  // CUDNN_VERSION_MIN(7, 0, 0)

#endif  // WITH_CUDNN