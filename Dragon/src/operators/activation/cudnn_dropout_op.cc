#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "operators/activation/dropout_op.h"

#ifdef WITH_CUDNN

#if CUDNN_VERSION_MIN(7, 0, 0)

namespace dragon {

template <class Context> template <typename T>
void CuDNNDropoutOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();
    CHECK(use_scale_) << "\nCuDNN only supports the scale-dropout";
    if (phase() == "TEST") {
        Y(0)->CopyFrom(X(0), ctx());
    } else if (phase() == "TRAIN") {
        // Determine the dropout states
        if (!states_initialized_) {
            states_initialized_ = true;
            CUDNN_CHECK(cudnnDropoutGetStatesSize(
                ctx()->cudnn_handle(), &states_size_));
            std::lock_guard<std::mutex> lk(CUDAContext::mutex());
            auto* states_tensor = ws()->CreateTensor(
                "/share/cudnn/dropout:" +
                    str::to(rng_seed_) + "/states");
            if (states_tensor->count() > 0) {
                auto* states = states_tensor->template
                    mutable_data<uint8_t, Context>();
                CUDNN_CHECK(cudnnRestoreDropoutDescriptor(
                    dropout_desc_,
                    ctx()->cudnn_handle(),
                    prob(),
                    states,
                    states_size_,
                    rng_seed_
                ));
            } else {
                auto* states = states_tensor
                    ->Reshape({ (int64_t) states_size_ })
                    ->template mutable_data<uint8_t, Context>();
                CUDNN_CHECK(cudnnSetDropoutDescriptor(
                    dropout_desc_,
                    ctx()->cudnn_handle(),
                    prob(),
                    states,
                    states_size_,
                    rng_seed_
                ));
            }
        }
        // Determine the faked input desc
        CuDNNSetTensor4dDesc<T>(
            &input_desc_, "NCHW",
            vec64_t({ X(0).count(), 1, 1, 1 })
        );
        CUDNN_CHECK(cudnnDropoutGetReserveSpaceSize(
            input_desc_, &reserve_size_));
        auto* mask = ws()
            ->CreateTensor(unique_name("mask"))
            ->Reshape({ (int64_t)reserve_size_ })
            ->template mutable_data<uint8_t, Context>();
        CUDNN_CHECK(cudnnDropoutForward(
            ctx()->cudnn_handle(),
            dropout_desc_,
            input_desc_, x,
            input_desc_, y,
            mask, reserve_size_
        ));
    } else {
        LOG(FATAL) << "Unknown Phase: " << phase();
    }
}

template <class Context>
void CuDNNDropoutOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <float, float16>>::Call(this, X(0));
}

template <class Context> template <typename T>
void CuDNNDropoutGradientOp<Context>::RunImpl() {
    if (phase() == "TEST") {
        NOT_IMPLEMENTED; 
    } else if (phase() == "TRAIN") {
        CHECK(use_scale_) << "\nCuDNN only supports scale-dropout";
        // Determine the dropout states
        if (!states_initialized_) {
            states_initialized_ = true;
            CUDNN_CHECK(cudnnDropoutGetStatesSize(
                ctx()->cudnn_handle(), &states_size_));
            std::lock_guard<std::mutex> lk(CUDAContext::mutex());
            auto* states_tensor = ws()->CreateTensor(
                "/share/cudnn/dropout:" +
                    str::to(rng_seed_) + "/states");
            if (states_tensor->count() > 0) {
                auto* states = states_tensor->template
                    mutable_data<uint8_t, Context>();
                CUDNN_CHECK(cudnnRestoreDropoutDescriptor(
                    dropout_desc_,
                    ctx()->cudnn_handle(),
                    prob(),
                    states,
                    states_size_,
                    rng_seed_
                ));
            } else { 
                LOG(FATAL) << "Missing states with "
                           << "seed: " << rng_seed_;
            }
        }
        auto* dy = X(-1).template data<T, Context>();
        auto* dx = Y(0)->template mutable_data<T, Context>();
        // Determine the faked input desc
        CuDNNSetTensor4dDesc<T>(
            &input_desc_, "NCHW",
            vec64_t({ X(-1).count(), 1, 1, 1 })
        );
        CUDNN_CHECK(cudnnDropoutGetReserveSpaceSize(
            input_desc_, &reserve_size_));
        auto* mask = ws()
            ->GetTensor(unique_name("mask"))
            ->template mutable_data<uint8_t, Context>();
        CUDNN_CHECK(cudnnDropoutBackward(
            ctx()->cudnn_handle(),
            dropout_desc_,
            input_desc_, dy,
            input_desc_, dx,
            mask, reserve_size_
        ));
    } else {
        LOG(FATAL) << "Unknown Phase: " << phase();
    }
}

template <class Context>
void CuDNNDropoutGradientOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <float, float16>>::Call(this, X(0));
}

DEPLOY_CUDNN(Dropout);
DEPLOY_CUDNN(DropoutGradient);

}  // namepsace dragon

#endif  // CUDNN_VERSION_MIN(7, 0, 0)

#endif  // WITH_CUDNN