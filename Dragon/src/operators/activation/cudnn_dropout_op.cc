#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "operators/activation/dropout_op.h"

#ifdef WITH_CUDNN

#if CUDNN_VERSION_MIN(7, 0, 0)

namespace dragon {

template <class Context> template <typename T>
void CuDNNDropoutOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    float scale = use_scale ? 1.f / (1.f - prob()) : 1.f;
    if (phase() == "TEST") {
        if (Output(0) != &Input(0)) {
            ctx()->template Copy<T, Context, Context>(
                Output(0)->count(), Ydata, Xdata);
            if (scale == 1.f)
                math::Scal<T, Context>(Output(0)->count(),
                    1.f - prob(), Ydata, ctx());
        }
    } else if (phase() == "TRAIN") {
        CHECK(use_scale) << "\nCuDNN only supports scale-dropout";
        Tensor* mask = ws()->CreateTensor(
            "/mnt/" + anchor() + "/dropout/mask");
        // Determine the dropout states
        if (!states_initialized) {
            states_initialized = true;
            CUDNN_CHECK(cudnnDropoutGetStatesSize(
                ctx()->cudnn_handle(), &states_size));
            std::lock_guard<std::mutex> lk(CUDAContext::mutex());
            Tensor* states = ws()->CreateTensor(
                "/share/cudnn/dropout:" + std::to_string(
                    random_seed) + "/states");
            if (states->count() > 0) {
                auto* Sdata = states->template mutable_data<uint8_t, Context>();
                CUDNN_CHECK(cudnnRestoreDropoutDescriptor(
                    dropout_desc, ctx()->cudnn_handle(), prob(),
                        Sdata, states_size, random_seed));
            } else {
                states->Reshape({ (TIndex)states_size });
                auto* Sdata = states->template mutable_data<uint8_t, Context>();
                CUDNN_CHECK(cudnnSetDropoutDescriptor(
                    dropout_desc, ctx()->cudnn_handle(), prob(),
                        Sdata, states_size, random_seed));
            }
        }
        // Determine the faked input desc
        cudnnSetTensor4dDesc<T>(&input_desc, "NCHW", 
            vector<TIndex>({ Input(0).count(), 1, 1, 1 }));
        CUDNN_CHECK(cudnnDropoutGetReserveSpaceSize(
            input_desc, &reserve_space_size));
        mask->Reshape({ (TIndex)reserve_space_size });
        auto* Rdata = mask->template mutable_data<uint8_t, Context>();
        CUDNN_CHECK(cudnnDropoutForward(
            ctx()->cudnn_handle(), dropout_desc,
                input_desc, Xdata,
                    input_desc, Ydata,
                        Rdata, reserve_space_size));
    } else LOG(FATAL) << "Incorrect Op phase: " << phase();
}

template <class Context>
void CuDNNDropoutOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CUDNN(Dropout);

template <class Context> template <typename T>
void CuDNNDropoutGradientOp<Context>::RunWithType() {
    if (phase() == "TEST") { NOT_IMPLEMENTED; }
    else if (phase() == "TRAIN") {
        CHECK(use_scale) << "\nCuDNN only supports scale-dropout";
        Tensor* mask = ws()->GetTensor(
            "/mnt/" + anchor() + "/dropout/mask");
        // Determine the dropout states
        if (!states_initialized) {
            states_initialized = true;
            CUDNN_CHECK(cudnnDropoutGetStatesSize(
                ctx()->cudnn_handle(), &states_size));
            std::lock_guard<std::mutex> lk(CUDAContext::mutex());
            Tensor* states = ws()->CreateTensor(
                "/share/cudnn/dropout:" + std::to_string(
                    random_seed) + "/states");
            if (states->count() > 0) {
                auto* Sdata = states->template mutable_data<uint8_t, Context>();
                CUDNN_CHECK(cudnnRestoreDropoutDescriptor(
                    dropout_desc, ctx()->cudnn_handle(), prob(),
                        Sdata, states_size, random_seed));
            } else { 
                LOG(FATAL) << "Missing states with seed: " << random_seed; 
            }
        }
        auto* dYdata = Input(-1).template data<T, Context>();
        auto* dXdata = Output(0)->template mutable_data<T, Context>();
        // Determine the faked input desc
        cudnnSetTensor4dDesc<T>(&input_desc, "NCHW",
            vector<TIndex>({ Input(-1).count(), 1, 1, 1 }));
        CUDNN_CHECK(cudnnDropoutGetReserveSpaceSize(
            input_desc, &reserve_space_size));
        auto* Rdata = mask->template mutable_data<uint8_t, Context>();
        CUDNN_CHECK(cudnnDropoutBackward(
            ctx()->cudnn_handle(), dropout_desc,
                input_desc, dYdata,
                    input_desc, dXdata,
                        Rdata, reserve_space_size));
    } else LOG(FATAL) << "Incorrect Op phase: " << phase();
}

template <class Context>
void CuDNNDropoutGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CUDNN(DropoutGradient);

}    // namepsace dragon

#endif

#endif  // WITH_CUDNN