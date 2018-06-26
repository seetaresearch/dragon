#include "operators/recurrent/cudnn_recurrent_op.h"
#include "core/workspace.h"
#include "utils/filler.h"

#ifdef WITH_CUDNN

#if CUDNN_VERSION_MIN(5, 0, 0)

namespace dragon {

template <class Context> template <typename T>
void CuDNNRecurrentOpBase<Context>::ResetDesc(
    Tensor* X, Tensor* Hx, Tensor* Cx,
    Tensor* Y, Tensor* Hy, Tensor* Cy) {
    input_dims = Input(0).dims();
    const auto seq_length = Input(0).dim(0);
    const auto batch_size = Input(0).dim(1);
    const auto input_dim = Input(0).dim(2);
    const auto num_directions = bidirectional ? 2 : 1;
    const auto output_dim = hidden_size * num_directions;

    //  setup dropout
    if (dropout_ratio < 1.0) {
        if (!states_initialized) {
            states_initialized = true;
            CUDNN_CHECK(cudnnDropoutGetStatesSize(
                cudnn_handle(), &states_size));
            std::lock_guard<std::mutex> lk(CUDAContext::mutex());
            Tensor* states = ws()->CreateTensor("/share/cudnn/dropout:" +
                dragon_cast<string, unsigned long long>(random_seed) + "/states");
            if (states->count() > 0) {
                auto* Sdata = states->template mutable_data<uint8_t, Context>();
                CUDNN_CHECK(cudnnRestoreDropoutDescriptor(
                    dropout_desc, cudnn_handle(), dropout_ratio,
                        Sdata, states_size, random_seed));
            } else {
                states->Reshape(vector<TIndex>(1, states_size));
                auto* Sdata = states->template mutable_data<uint8_t, Context>();
                CUDNN_CHECK(cudnnSetDropoutDescriptor(
                    dropout_desc, cudnn_handle(), dropout_ratio,
                        Sdata, states_size, random_seed));
            }
        }
    }

    //  setup rnn
#if CUDNN_VERSION_MIN(7, 0, 0)
    CUDNN_CHECK(cudnnSetRNNDescriptor(
        cudnn_handle(), rnn_desc,
            hidden_size, num_layers,
                dropout_desc,
                    rnn_input_mode, rnn_direction, rnn_mode,
                        CUDNN_RNN_ALGO_STANDARD,
                            CUDNNType<T>::type));
#else
    CUDNN_CHECK(cudnnSetRNNDescriptor(
        rnn_desc,
            hidden_size, num_layers,
                dropout_desc,
                    rnn_input_mode, rnn_direction, rnn_mode,
                        CUDNNType<T>::type));
#endif

    //  setup Xs & Ys & Y
    xs_desc.reset(new cudnnTensorDescriptors(seq_length));
    xs_desc->Set<T>({ batch_size, input_dim, 1 }, { input_dim, 1, 1 });
    ys_desc.reset(new cudnnTensorDescriptors(seq_length));
    ys_desc->Set<T>({ batch_size, output_dim, 1 }, { output_dim, 1, 1 });
    CUDNN_CHECK(cudnnGetRNNWorkspaceSize(cudnn_handle(), rnn_desc,
        seq_length, xs_desc->descs(), &workspace_size));
    if (Y) Y->Reshape(vector<TIndex>({ seq_length, batch_size, output_dim }));

    //  setup Hx & Cx & Hy & Cy
    vector<TIndex> hidden_dims(
        { num_layers * num_directions, batch_size, hidden_size }
    );
    if (Hx) { TENSOR_FILL((*Hx), hidden_dims); }  //  Hx-filler
    if (Cx) { TENSOR_FILL((*Cx), hidden_dims); }  //  Cx-filler
    cudnnSetTensorDesc<T>(&hx_desc, hidden_dims);
    cudnnSetTensorDesc<T>(&cx_desc, hidden_dims);
    cudnnSetTensorDesc<T>(&hy_desc, hidden_dims);
    cudnnSetTensorDesc<T>(&cy_desc, hidden_dims);
    if (Hy) Hy->Reshape(hidden_dims);
    if (Cy) Cy->Reshape(hidden_dims);

    // setup packed weights
    size_t weights_size; TIndex weights_count;
    CUDNN_CHECK(cudnnGetRNNParamsSize(cudnn_handle(), rnn_desc,
        xs_desc->descs()[0], &weights_size, CUDNNType<T>::type));
    weights_count = (TIndex)weights_size / sizeof(T);
    CHECK_EQ(weights_count, Input(1).count())
        << "\nModel request " << "Tensor(" << Input(1).name() << ")'s "
        << "size is " << weights_count << ", \n"
        << "but now is " << Input(1).count() << ", "
        << "did you feed the incorrect Tensor before ?";
    CUDNN_CHECK(cudnnSetFilterNdDescriptor(w_desc,
        CUDNNType<T>::type, CUDNN_TENSOR_NCHW, 3,
            vector<int>({ (int)weights_count, 1, 1 }).data()));

    //  setup rnn workspace
    CUDNN_CHECK(cudnnGetRNNWorkspaceSize(cudnn_handle(), rnn_desc,
        seq_length, xs_desc->descs(), &workspace_size));
}

template <class Context> template <typename T>
void CuDNNRecurrentOp<Context>::RunWithType() {
    const auto seq_length = Input(0).dim(0);
    if (Input(0).dims() != input_dims) {
        this->template ResetDesc<T>(&Input(0), &Input(2),  // X, Cx
                   InputSize() > 3 ? &Input(3) : nullptr,  // Cy
                                               Output(0),  // Y
                  OutputSize() > 1 ? Output(1) : nullptr,  // Hy
                  OutputSize() > 2 ? Output(2) : nullptr); // Cy
    }
    auto XsData = [this](int i) {
        if (i >= InputSize()) return (const T*)NULL;
        return Input(i).template data<T, Context>();
    };
    auto YsData = [this](int i) {
        if (i >= OutputSize()) return (T*)NULL;
        if (Output(i)->name() == "ignore") return (T*)NULL;
        return Output(i)->template mutable_data<T, Context>();
    };

    auto* WSdata = ws()->template caches<Context>({ workspace_size })[0];

    if (phase() == "TRAIN") {
        CUDNN_CHECK(cudnnGetRNNTrainingReserveSize(cudnn_handle(),
            rnn_desc, seq_length, xs_desc->descs(), &reserve_size));
        auto* reserveT = ws()->CreateTensor("/mnt/" + anchor() + "/rnn/reserve");
        reserveT->Reshape({ (TIndex)reserve_size });
        auto* RSdata = reserveT->template mutable_data<uint8_t, Context>();
        CUDNN_CHECK(cudnnRNNForwardTraining(cudnn_handle(), rnn_desc,
                                                          seq_length,
                                         xs_desc->descs(), XsData(0),
                                                  hx_desc, XsData(2),
                                                  cx_desc, XsData(3),
                                                   w_desc, XsData(1),
                                         ys_desc->descs(), YsData(0),
                                                  hy_desc, YsData(1),
                                                  cy_desc, YsData(2),
                                              WSdata, workspace_size,
                                              RSdata, reserve_size));
    } else if (phase() == "TEST") {
        CUDNN_CHECK(cudnnRNNForwardInference(cudnn_handle(), rnn_desc,
                                                           seq_length,
                                          xs_desc->descs(), XsData(0),
                                                   hx_desc, XsData(2),
                                                   cx_desc, XsData(3),
                                                    w_desc, XsData(1),
                                          ys_desc->descs(), YsData(0),
                                                   hy_desc, YsData(1),
                                                   cy_desc, YsData(2),
                                             WSdata, workspace_size));

    } else LOG(FATAL) << "Incorrect Op phase: " << phase();
}

template <class Context>
void CuDNNRecurrentOp<Context>::RunOnDevice() {
    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CUDNN(Recurrent);

template <class Context> template <typename T>
void CuDNNRecurrentGradientOp<Context>::RunWithType() {
    const auto seq_length = Input(0).dim(0);
    if (Input(0).dims() != input_dims)
        this->template ResetDesc<T>(&Input(0),
            nullptr, nullptr, nullptr, nullptr, nullptr);

    auto XsData = [this](int i) { 
        if (i >= InputSize()) return (const T*)NULL;
        return Input(i).template data<T, Context>(); 
    };
    auto YsData = [this](int i) { 
        if (i >= OutputSize()) return (T*)NULL;
        if (Output(i)->name() == "ignore") (T*)NULL;
        return Output(i)->template mutable_data<T, Context>();
    };

    auto* WSdata = ws()->template caches<Context>({ workspace_size })[0];
    //  check the reserve space
    CUDNN_CHECK(cudnnGetRNNTrainingReserveSize(cudnn_handle(), rnn_desc,
        seq_length, xs_desc->descs(), &reserve_size));
    auto* reserveT = ws()->GetTensor("/mnt/" + anchor() + "/rnn/reserve");
    CHECK_EQ(reserve_size, reserveT->nbytes());
#if CUDNN_VERSION_MIN(6,0,0)
    auto* RSdata = reserveT->template mutable_data<uint8_t, Context>();
#else
    auto* RSdata = reserveT->template data<uint8_t, Context>();
#endif

    CUDNN_CHECK(cudnnRNNBackwardData(cudnn_handle(), rnn_desc,
                                                   seq_length,
                                  ys_desc->descs(), XsData(4), //   Y
                                  ys_desc->descs(), XsData(5), //  dY
                                           hy_desc, XsData(6), // dHy
                                           cy_desc, XsData(7), // dCy
                                            w_desc, XsData(1), //   W
                                           hx_desc, XsData(2), //  Hx
                                           cx_desc, XsData(3), //  Cx
                                  xs_desc->descs(), YsData(0), //  dX
                                           hx_desc, YsData(2), // dHx
                                           cx_desc, YsData(3), // dHy
                                       WSdata, workspace_size,
                                       RSdata, reserve_size));

    CUDNN_CHECK(cudnnRNNBackwardWeights(cudnn_handle(), rnn_desc,
                                                      seq_length,
                                     xs_desc->descs(), XsData(0), //   X
                                              hx_desc, XsData(2), //  Hx
                                     ys_desc->descs(), XsData(4), //   Y
                                          WSdata, workspace_size,
                                               w_desc, YsData(1), //  dW
                                          RSdata, reserve_size));
}

template <class Context>
void CuDNNRecurrentGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));  // dX
    Output(1)->ReshapeLike(Input(1));  // dW
    Output(2)->ReshapeLike(Input(2));  // dHx
    Output(3)->ReshapeLike(Input(3));  // dCx

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CUDNN(RecurrentGradient);

}    // namespace dragon

#endif

#endif    // WITH_CUDNN