#include "core/workspace.h"
#include "utils/filler.h"
#include "operators/recurrent/cudnn_recurrent_op.h"

#ifdef WITH_CUDNN

#if CUDNN_VERSION_MIN(5, 0, 0)

namespace dragon {

template <class Context> template <typename T>
void CuDNNRecurrentOpBase<Context>::ResetDesc() {
    input_dims_ = X(0).dims();
    seq_length_ = X(0).dim(0);
    auto batch_size = X(0).dim(1);
    auto x_dim = X(0).dim(2);
    auto ndirections = bidirectional_ ? 2 : 1;
    auto y_dim = hidden_size_ * ndirections;

    // Setup Dropout
    if (dropout_ratio_ < 1.f) {
#if CUDNN_VERSION_MIN(7, 0, 0)
        if (!states_initialized_) {
            states_initialized_ = 1;
            CUDNN_CHECK(cudnnDropoutGetStatesSize(
                ctx()->cudnn_handle(), &states_size_));
            std::lock_guard<std::mutex> lk(CUDAContext::mutex());
            auto* states_tensor = ws()->CreateTensor(
                "/share/cudnn/dropout:" +
                std::to_string(rng_seed_) + "/states");
            if (states_tensor->count() > 0) {
                auto* states = states_tensor->template
                    mutable_data<uint8_t, Context>();
                CUDNN_CHECK(cudnnRestoreDropoutDescriptor(
                    dropout_desc_,
                    ctx()->cudnn_handle(),
                    dropout_ratio_,
                    states,
                    states_size_,
                    rng_seed_
                ));
            } else {
                auto* states = states_tensor
                    ->Reshape({ (int64_t)states_size_ })
                    ->template mutable_data<uint8_t, Context>();
                CUDNN_CHECK(cudnnSetDropoutDescriptor(
                    dropout_desc_,
                    ctx()->cudnn_handle(),
                    dropout_ratio_,
                    states,
                    states_size_,
                    rng_seed_
                ));
            }
        }
#else
        LOG(FATAL) << "Dropout has been supported since CuDNN 7.0";
#endif
    }

    // Setup RNN
#if CUDNN_VERSION_MIN(7, 0, 0)
    CUDNN_CHECK(cudnnSetRNNDescriptor(
        ctx()->cudnn_handle(),
        rnn_desc_,
        hidden_size_,
        num_layers_,
        dropout_desc_,
        rnn_input_mode_,
        rnn_direction_,
        rnn_mode_,
        CUDNN_RNN_ALGO_STANDARD,
        CuDNNType<T>::type
    ));
#else
    CUDNN_CHECK(cudnnSetRNNDescriptor(
        rnn_desc_,
        hidden_size_,
        num_layers_,
        dropout_desc_,
        rnn_input_mode_,
        rnn_direction_,
        rnn_mode_,
        CuDNNType<T>::type
    ));
#endif

    // Setup X and Y
    output_dims_ = { seq_length_, batch_size, y_dim };
    x_descs_.reset(new CuDNNTensorDescs(seq_length_));
    y_descs_.reset(new CuDNNTensorDescs(seq_length_));
    x_descs_->Set<T>({ batch_size, x_dim, 1 }, { x_dim, 1, 1 });
    y_descs_->Set<T>({ batch_size, y_dim, 1 }, { y_dim, 1, 1 });

    // Setup Hx, Cx, Hy and Cy
    hidden_dims_ = {
        num_layers_
            * ndirections,
        batch_size,
        hidden_size_,
    };
    CuDNNSetTensorDesc<T>(&hx_desc_, hidden_dims_);
    CuDNNSetTensorDesc<T>(&cx_desc_, hidden_dims_);
    CuDNNSetTensorDesc<T>(&hy_desc_, hidden_dims_);
    CuDNNSetTensorDesc<T>(&cy_desc_, hidden_dims_);

    // Setup packed weights
    size_t w_size; int64_t w_count;
    CUDNN_CHECK(cudnnGetRNNParamsSize(
        ctx()->cudnn_handle(),
        rnn_desc_,
        x_descs_->data()[0],
        &w_size,
        CuDNNType<T>::type
    )); w_count = (int64_t)w_size / sizeof(T);
    CHECK_EQ(w_count, X(1).count())
        << "\nModel request " << "Tensor("
        << X(1).name() << ")'s " << "size is "
        << w_count << ", \n" << "but now is "
        << X(1).count() << ", "
        << "did you feed the incorrect data before?";
    CUDNN_CHECK(cudnnSetFilterNdDescriptor(
        w_desc_,
        CuDNNType<T>::type,
        CUDNN_TENSOR_NCHW, 3,
        vec32_t({ (int)w_count, 1, 1 }).data())
    );

    // Determine the RNN workspace
    CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
        ctx()->cudnn_handle(),
        rnn_desc_,
        seq_length_,
        x_descs_->data(),
        &workspace_size_
    ));
}

template <class Context> template <typename T>
void CuDNNRecurrentOp<Context>::RunImpl() {
    if (X(0).dims() != input_dims_) {
        this->template ResetDesc<T>();
    }

    if (XSize() > 2) { TENSOR_FILL(X(2), hidden_dims_); }
    if (XSize() > 3) { TENSOR_FILL(X(3), hidden_dims_); }

    Y(0)->Reshape(output_dims_);
    if (YSize() > 1) Y(1)->Reshape(hidden_dims_);
    if (YSize() > 2) Y(2)->Reshape(hidden_dims_);

    auto xAt = [this](int i) {
        if (i >= XSize()) return (const T*)NULL;
        return X(i).template data<T, Context>();
    };

    auto yAt = [this](int i) {
        if (i >= YSize()) return (T*)NULL;
        if (Y(i)->name() == "NULL") return (T*)NULL;
        return Y(i)->template mutable_data<T, Context>();
    };

    auto* scratch = ws()
        ->template data<Context>
            ({ workspace_size_ })[0];

    if (phase() == "TRAIN") {
        CUDNN_CHECK(cudnnGetRNNTrainingReserveSize(
            ctx()->cudnn_handle(),
            rnn_desc_,
            seq_length_,
            x_descs_->data(),
            &reserve_size_
        ));
        auto* reserve = ws()
            ->CreateTensor(unique_name("reserve"))
            ->Reshape({ (int64_t)reserve_size_ })
            ->template mutable_data<uint8_t, Context>();
        CUDNN_CHECK(cudnnRNNForwardTraining(
            ctx()->cudnn_handle(),
            rnn_desc_,
            seq_length_,
            x_descs_->data(), xAt(0),
            hx_desc_,         xAt(2),
            cx_desc_,         xAt(3),
            w_desc_,          xAt(1),
            y_descs_->data(), yAt(0),
            hy_desc_,         yAt(1),
            cy_desc_,         yAt(2),
            scratch, workspace_size_,
            reserve, reserve_size_
        ));
    } else if (phase() == "TEST") {
        CUDNN_CHECK(cudnnRNNForwardInference(
            ctx()->cudnn_handle(),
            rnn_desc_,
            seq_length_,
            x_descs_->data(), xAt(0),
            hx_desc_,         xAt(2),
            cx_desc_,         xAt(3),
            w_desc_,          xAt(1),
            y_descs_->data(), yAt(0),
            hy_desc_,         yAt(1),
            cy_desc_,         yAt(2),
            scratch, workspace_size_
        ));
    } else {
        LOG(FATAL) << "Unknown Phase: " << phase();
    }
}

template <class Context>
void CuDNNRecurrentOp<Context>::RunOnDevice() {
    if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else if (XIsType(X(0), float16)) {
        RunImpl<float16>();
    } else {
        LOG(FATAL) << DTypeString(X(0),
            { "float32", "float16" }
        );
    }
}

template <class Context> template <typename T>
void CuDNNRecurrentGradientOp<Context>::RunImpl() {
    if (X(0).dims() != input_dims_) {
        this->template ResetDesc<T>();
    }

    auto xAt = [this](int i) {
        if (i >= XSize()) return (const T*)NULL;
        if (X(i).name() == "NULL") return (const T*)NULL;
        return X(i).template data<T, Context>();
    };

    auto yAt = [this](int i) {
        if (i >= YSize()) return (T*)NULL;
        if (Y(i)->name() == "NULL" && i > 0) return (T*)NULL;
        return Y(i)->template mutable_data<T, Context>();
    };

    auto* scratch = ws()
        ->template data<Context>
            ({ workspace_size_ })[0];

    // Check the ReserveSpace
    CUDNN_CHECK(cudnnGetRNNTrainingReserveSize(
        ctx()->cudnn_handle(),
        rnn_desc_,
        seq_length_,
        x_descs_->data(),
        &reserve_size_
    ));
    auto* reserve_tensor = ws()
        ->GetTensor(unique_name("reserve"));
    CHECK_EQ(reserve_size_, reserve_tensor->nbytes());
#if CUDNN_VERSION_MIN(6,0,0)
    auto* reserve = reserve_tensor
        ->template mutable_data<uint8_t, Context>();
#else
    auto* reserve = reserve_tensor
        ->template data<uint8_t, Context>();
#endif

    if (Y(0)->name() != "NULL" ||
        Y(1)->name() != "NULL" ||
        Y(2)->name() != "NULL" ||
        Y(3)->name() != "NULL") {
        CUDNN_CHECK(cudnnRNNBackwardData(
            ctx()->cudnn_handle(),
            rnn_desc_,
            seq_length_,
            y_descs_->data(), xAt(4),  //   Y
            y_descs_->data(), xAt(5),  //  dY
            hy_desc_,         xAt(6),  // dHy
            cy_desc_,         xAt(7),  // dCy
            w_desc_,          xAt(1),  //   W
            hx_desc_,         xAt(2),  //  Hx
            cx_desc_,         xAt(3),  //  Cx
            x_descs_->data(), yAt(0),  //  dX
            hx_desc_,         yAt(2),  // dHx
            cx_desc_,         yAt(3),  // dHy
            scratch, workspace_size_,
            reserve, reserve_size_
        ));
    }

    if (Y(1)->name() != "NULL") {
        math::Set(
            Y(1)->count(),
            cast::to<T>(0.f),
            yAt(1), ctx()
        );  // CuDNN accumulates the gradient of weights
        CUDNN_CHECK(cudnnRNNBackwardWeights(
            ctx()->cudnn_handle(),
            rnn_desc_,
            seq_length_,
            x_descs_->data(), xAt(0),  //   X
            hx_desc_,         xAt(2),  //  Hx
            y_descs_->data(), xAt(4),  //   Y
            scratch, workspace_size_,
            w_desc_,          yAt(1),  //  dW
            reserve, reserve_size_
        ));
    }
}

template <class Context>
void CuDNNRecurrentGradientOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));  // dX
    Y(1)->ReshapeLike(X(1));  // dW
    Y(2)->ReshapeLike(X(2));  // dHx
    Y(3)->ReshapeLike(X(3));  // dCx

    if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else if (XIsType(X(0), float16)) {
        RunImpl<float16>();
    } else {
        LOG(FATAL) << DTypeString(X(0),
            { "float32", "float16" }
        );
    }
}

DEPLOY_CUDNN(Recurrent);
DEPLOY_CUDNN(RecurrentGradient);

}  // namespace dragon

#endif

#endif  // WITH_CUDNN