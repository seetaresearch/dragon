#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/op_kernel.h"
#include "operators/norm/batch_norm_op.h"

#ifdef WITH_CUDNN

#if CUDNN_VERSION_MIN(5, 0, 0)

namespace dragon {

template <class Context> template <typename T>
void CuDNNBatchNormOp<Context>::RunWithType() {
    typedef typename CUDNNType<T>::BNParamType Tp;

    // Determine the bn desc
    if (Input(0).ndim() == 2) {
        bn_mode = CUDNN_BATCHNORM_PER_ACTIVATION;
        Tensor x_reshape;
        x_reshape.Reshape(vector<int64_t>({ N, C, 1, 1 }));
        cudnnSetTensorDesc<T>(&input_desc, &x_reshape);
        cudnnSetTensorDesc<T>(&output_desc, &x_reshape);
    } else {
        CHECK_GE((int)Input(0).ndim(), 3)
            << "The number of dimensions should be at least 3.";
        bn_mode = CUDNN_BATCHNORM_SPATIAL;
#if CUDNN_VERSION_MIN(7, 0, 0)
        if (is_training) bn_mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#endif
        if (data_format == "NCHW") {
            cudnnSetTensorDesc<T>(&input_desc, &Input(0));
            cudnnSetTensorDesc<T>(&output_desc, Output(0));
        } else if (data_format == "NHWC") {
            switch (Input(0).ndim()) {
                case 3:
                    cudnnSetTensor3dDesc<T>(
                        &input_desc, data_format, &Input(0));
                    cudnnSetTensor3dDesc<T>(
                        &output_desc, data_format, Output(0));
                    break;
                case 4:
                    cudnnSetTensor4dDesc<T>(
                        &input_desc, data_format, &Input(0));
                    cudnnSetTensor4dDesc<T>(
                        &output_desc, data_format, Output(0));
                    break;
                case 5:
                    cudnnSetTensor5dDesc<T>(
                        &input_desc, data_format, &Input(0));
                    cudnnSetTensor5dDesc<T>(
                        &output_desc, data_format, Output(0));
                    break;
                default:
                    LOG(FATAL) << "Only support the 3d/4d/5d input at NHWC.";
            }
        }
    }

    // Derive the bn desc
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bn_desc, input_desc, bn_mode));

    TENSOR_FILL_WITH_TYPE(Input(1), vector<int64_t>({ C }), Tp);
    TENSOR_FILL_WITH_TYPE(Input(2), vector<int64_t>({ C }), Tp);
    TENSOR_FILL_WITH_TYPE(Input(3), vector<int64_t>({ C }), Tp);
    TENSOR_FILL_WITH_TYPE(Input(4), vector<int64_t>({ C }), Tp);

    auto* x = Input(0).template data<T, Context>();
    auto* rm = Input(1).template mutable_data<Tp, Context>();
    auto* rv = Input(2).template mutable_data<Tp, Context>();
    auto* gamma = Input(3).template data<Tp, Context>();
    auto* beta = Input(4).template data<Tp, Context>();
    auto* y = Output(0)->template mutable_data<T, Context>();

    if (is_training) {
        auto* sm = mean->template mutable_data<Tp, Context>();
        auto* sv = var->template mutable_data<Tp, Context>();
        auto mt = is_recomputing ? 0.f : 1.f - this->momentum;
        CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
            ctx()->cudnn_handle(), bn_mode,
                CUDNNType<T>::one, CUDNNType<T>::zero,
                    input_desc, x, output_desc, y,
                        bn_desc, gamma, beta,
                            mt, rm, rv, eps64, sm, sv));
    } else {
         CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
            ctx()->cudnn_handle(), bn_mode,
                CUDNNType<T>::one, CUDNNType<T>::zero,
                    input_desc, x, output_desc, y,
                        bn_desc, gamma, beta, rm, rv, eps64));
    }
}

template <class Context>
void CuDNNBatchNormOp<Context>::Reshape() {
    // Determine the mode
    if (use_stats == -1) {
        is_training = phase() == "TRAIN" ? true : false;
    } else {
        is_training = use_stats == 0 ? true : false;
    }

    // Get the recomputing flag
    is_recomputing = ws()->GetTensor(
        "/opt/recomputing_flag")->template
            data<bool, CPUContext>()[0];

    // Determine the data format
    int64_t channel_axis = axis;
    data_format = "NCHW";
    if (channel_axis == -1) channel_axis += Input(0).ndim();
    if (channel_axis + 1 == Input(0).ndim()) data_format = "NHWC";
    N = Input(0).dim(0); C = Input(0).dim(channel_axis);

    // Create the shared resources
    mean = ws()->CreateTensor(mount_name(
        "bn/sm"))->Reshape({ C });
    var = ws()->CreateTensor(mount_name(
        "bn/sv"))->Reshape({ C });

    // Reshape
    Output(0)->ReshapeLike(Input(0));
}

template <class Context>
void CuDNNBatchNormOp<Context>::RunOnDevice() {
    Reshape();

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

template <class Context>
void CuDNNBatchNormGradientOp<Context>::Reshape() {
    // Determine the mode
    if (use_stats == -1) {
        is_training = phase() == "TRAIN" ? true : false;
    } else {
        is_training = use_stats == 0 ? true : false;
    }

    // Determine the data format
    int64_t channel_axis = axis;
    data_format = "NCHW";
    if (channel_axis == -1) channel_axis += Input(0).ndim();
    if (channel_axis + 1 == Input(0).ndim()) data_format = "NHWC";
    N = Input(0).dim(0); C = Input(0).dim(channel_axis);
    S = Input(0).count() / N / C;

    // Get the shared resources
    mean = ws()->GetTensor(mount_name("bn/sm"));
    var = ws()->GetTensor(mount_name("bn/sv"));

    // Reshape
    Output(0)->ReshapeLike(Input(0));  // dx
    Output(1)->Reshape({ C });         // dgamma
    Output(2)->Reshape({ C });         // dbeta
}

template <class Context> template <typename T>
void CuDNNBatchNormGradientOp<Context>::TrainingRunWithType() {
    typedef typename CUDNNType<T>::BNParamType Tp;

    // Determine the bn desc
    if (Input(0).ndim() == 2) {
        bn_mode = CUDNN_BATCHNORM_PER_ACTIVATION;
        Tensor x_reshape;
        x_reshape.Reshape(vector<int64_t>({ N, C, 1, 1 }));
        cudnnSetTensorDesc<T>(&input_desc, &x_reshape);
        cudnnSetTensorDesc<T>(&output_desc, &x_reshape);
    } else {
        CHECK_GE((int)Input(0).ndim(), 3)
            << "The number of dimensions should be at least 3.";
        bn_mode = CUDNN_BATCHNORM_SPATIAL;
#if CUDNN_VERSION_MIN(7, 0, 0)
        if (is_training) bn_mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#endif
        if (data_format == "NCHW") {
            cudnnSetTensorDesc<T>(&input_desc, &Input(-1));
            cudnnSetTensorDesc<T>(&output_desc, Output(0));
        } else if (data_format == "NHWC") {
            switch (Input(0).ndim()) {
                case 3:
                    cudnnSetTensor3dDesc<T>(
                        &input_desc, data_format, &Input(-1));
                    cudnnSetTensor3dDesc<T>(
                        &output_desc, data_format, Output(0));
                    break;
                case 4:
                    cudnnSetTensor4dDesc<T>(
                        &input_desc, data_format, &Input(-1));
                    cudnnSetTensor4dDesc<T>(
                        &output_desc, data_format, Output(0));
                    break;
                case 5:
                    cudnnSetTensor5dDesc<T>(
                        &input_desc, data_format, &Input(-1));
                    cudnnSetTensor5dDesc<T>(
                        &output_desc, data_format, Output(0));
                    break;
                default:
                    LOG(FATAL) << "Only support the 3d/4d/5d input at NHWC.";
            }
        }
    }

    // Derive the bn desc
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(
        bn_desc, input_desc, bn_mode));

    auto* x = Input(0).template data<T, Context>();
    auto* sm = mean->template data<Tp, Context>();
    auto* sv = var->template data<Tp, Context>();
    auto* gamma = Input(3).template data<Tp, Context>();
    auto* dy = Input(-1).template data<T, Context>();
    auto* dx = Output(0)->template mutable_data<T, Context>();
    auto* dgamma = Output(1)->template mutable_data<Tp, Context>();
    auto* dbeta = Output(2)->template mutable_data<Tp, Context>();

    CUDNN_CHECK(cudnnBatchNormalizationBackward(
        ctx()->cudnn_handle(), bn_mode,
            CUDNNType<T>::one, CUDNNType<T>::zero,
                CUDNNType<T>::one, CUDNNType<T>::zero,
                    output_desc, x, input_desc, dy,
                        output_desc, dx, bn_desc,
                            gamma, dgamma, dbeta,
                                eps64, sm, sv));
}

template <class Context> template <typename T>
void CuDNNBatchNormGradientOp<Context>::InferenceRunWithType() {
    typedef typename CUDNNType<T>::BNParamType Tp;

    auto* x = Input(0).template data<T, Context>();
    auto* rm = Input(1).template data<Tp, Context>();
    auto* rv = Input(2).template data<Tp, Context>();
    auto* gamma = Input(3).template data<Tp, Context>();
    auto* dy = Input(-1).template data<T, Context>();
    auto* dx = Output(0)->template mutable_data<T, Context>();
    auto* rsig = var->template mutable_data<Tp, Context>();

    Tp* dgamma = nullptr, *dbeta = nullptr;

    // Gradient w.r.t. gamma or beta if necessary
    if (Output(1)->name() != "ignore" ||
            Output(2)->name() != "ignore") {
        dgamma = Output(1)->template mutable_data<Tp, Context>();
        dbeta = Output(2)->template mutable_data<Tp, Context>();
    }

    math::InvStd(C, (float)eps64, rv, rsig, ctx());

    kernel::BatchNormBackwardInference(
        N, C, S, data_format,
            x, rm, rsig, gamma, dy,
                dx, dgamma, dbeta, ctx());

}

template <class Context>
void CuDNNBatchNormGradientOp<Context>::RunOnDevice() {
    Reshape();

    if (XIsType(Input(0), float)) {
        if (is_training) {
            TrainingRunWithType<float>();
        } else {
            InferenceRunWithType<float>();
        }
    } else if (XIsType(Input(0), float16)) {
        if (is_training) {
            TrainingRunWithType<float16>();
        } else {
            // We will support it some day -:)
            LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
        }
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CUDNN(BatchNorm);
DEPLOY_CUDNN(BatchNormGradient);

}  // namespace dragon

#endif  // CUDNN_VERSION_MIN(5, 0, 0)

#endif  // WITH_CUDNN