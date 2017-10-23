#include "operators/norm/batch_norm_op.h"
#include "core/workspace.h"
#include "utils/filler.h"

#ifdef WITH_CUDNN

#if CUDNN_VERSION_MIN(5, 0, 0)

namespace dragon {

template <class Context> template <typename T>
void CuDNNBNOp<Context>::SpatialRunWithType() {
    cudnnSetTensorDesc<T>(&input_desc, &input(0));
    cudnnSetTensorDesc<T>(&output_desc, output(0));
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bn_desc, input_desc, CUDNN_BATCHNORM_SPATIAL));

    TENSOR_FILL(input(1), vector<TIndex>(1, channels));    //  history_mean
    TENSOR_FILL(input(2), vector<TIndex>(1, channels));    //  history_var
    TENSOR_FILL(input(3), vector<TIndex>(1, channels));    //  scale
    TENSOR_FILL(input(4), vector<TIndex>(1, channels));    //  bias

    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();

    
    auto* hMean_data = input(1).template mutable_data<T, Context>();
    auto* hVar_data = input(2).template mutable_data<T, Context>();
    auto* Sdata = input(3).template data<T, Context>();
    auto* Bdata = input(4).template data<T, Context>();

    if (use_global_stats) {
        CUDNN_CHECK(cudnnBatchNormalizationForwardInference(cudnn_handle(),
                                                   CUDNN_BATCHNORM_SPATIAL,
                                                         CUDNNType<T>::one,
                                                        CUDNNType<T>::zero,
                                                         input_desc, Xdata,
                                                        output_desc, Ydata,
                                                                   bn_desc, 
                                                                     Sdata, 
                                                                     Bdata,
                                                                hMean_data, 
                                                                 hVar_data,
                                                               this->eps));

    } else {
        auto* tMean_data = mean->template mutable_data<T, Context>();
        auto* tVar_data = var->template mutable_data<T, Context>();
        CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(cudnn_handle(),
                                                  CUDNN_BATCHNORM_SPATIAL,
                                                        CUDNNType<T>::one,
                                                       CUDNNType<T>::zero,
                                                        input_desc, Xdata,
                                                       output_desc, Ydata,
                                                                  bn_desc,
                                                                    Sdata,
                                                                    Bdata,
                              is_recomputing ? 0.0 : 1.0 - this->momentum,
                                                               hMean_data,
                                                                hVar_data,
                                                                this->eps,
                                                               tMean_data,
                                                               tVar_data));
    }
}

template <class Context> template <typename T>
void CuDNNBNOp<Context>::PerActivationRunWithType() {
    Tensor x_reshape;
    x_reshape.Reshape(vector<TIndex>({ num, channels, 1, 1 }));
    cudnnSetTensorDesc<T>(&input_desc,  &x_reshape);
    cudnnSetTensorDesc<T>(&output_desc, &x_reshape);
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bn_desc, input_desc, CUDNN_BATCHNORM_PER_ACTIVATION));

    TENSOR_FILL(input(1), vector<TIndex>(1, channels));  //  history_mean
    TENSOR_FILL(input(2), vector<TIndex>(1, channels));  //  history_var
    TENSOR_FILL(input(3), vector<TIndex>(1, channels));  //  scale
    TENSOR_FILL(input(4), vector<TIndex>(1, channels));  //  bias

    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();

    auto* hMean_data = input(1).template mutable_data<T, Context>();
    auto* hVar_data = input(2).template mutable_data<T, Context>();
    auto* Sdata = input(3).template data<T, Context>();
    auto* Bdata = input(4).template data<T, Context>();

    if (use_global_stats) {
        CUDNN_CHECK(cudnnBatchNormalizationForwardInference(cudnn_handle(),
                                            CUDNN_BATCHNORM_PER_ACTIVATION,
                                                         CUDNNType<T>::one,
                                                        CUDNNType<T>::zero,
                                                         input_desc, Xdata,
                                                        output_desc, Ydata,
                                                                   bn_desc, 
                                                                     Sdata, 
                                                                     Bdata,
                                                                hMean_data, 
                                                                 hVar_data,
                                                               this->eps));

    } else {
        auto* tMean_data = mean->template mutable_data<T, Context>();
        auto* tVar_data = var->template mutable_data<T, Context>();
        CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(cudnn_handle(),
                                           CUDNN_BATCHNORM_PER_ACTIVATION,
                                                        CUDNNType<T>::one,
                                                       CUDNNType<T>::zero,
                                                        input_desc, Xdata,
                                                       output_desc, Ydata,
                                                                  bn_desc,
                                                                    Sdata,
                                                                    Bdata,
                              is_recomputing ? 0.0 : 1.0 - this->momentum,
                                                               hMean_data,
                                                                hVar_data,
                                                                this->eps,
                                                               tMean_data,
                                                               tVar_data));
    }
}

template <class Context>
void CuDNNBNOp<Context>::RunOnDevice() {
    num = input(0).dim(0); 
    channels = input(0).dim(1);
    spatial_dim = input(0).count(2); 

    mean = ws()->CreateTensor("_t_" + anchor() + "_bn_mean");
    var = ws()->CreateTensor("_t_" + anchor() + "_bn_var");
    mean->ReshapeLike(input(1)); var->ReshapeLike(input(2));

    output(0)->ReshapeLike(input(0));

    if (this->use_stats == -1) use_global_stats = phase() == "TEST" ? true : false;
    else use_global_stats = this->use_stats == 1 ? true : false;
    is_recomputing = ws()->GetTensor("_t_global_recompute_flag")
                         ->template data<bool, CPUContext>()[0];

    if (input(0).template IsType<float>()) {
        if (input(0).ndim() == 4) SpatialRunWithType<float>();
        else if (input(0).ndim() == 2) PerActivationRunWithType<float>();
        else LOG(FATAL) << "The ndim of input tensor should be 2 or 4.";
    } else { LOG(FATAL) << "Unsupported input types."; }
}

DEPLOY_CPU(BN);
#ifdef WITH_CUDA
DEPLOY_CUDA(BN);
#endif
OPERATOR_SCHEMA(BN).NumInputs(5).NumOutputs(1);
DEPLOY_CUDNN(BN);

template <class Context> template <typename T>
void CuDNNBNGradientOp<Context>::SpatialRunWithType() {
    cudnnSetTensorDesc<T>(&input_desc, &input(-1));
    cudnnSetTensorDesc<T>(&output_desc, output(0));
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bn_desc, output_desc, CUDNN_BATCHNORM_SPATIAL));

    if (use_global_stats) {
        if (output(0)->name() != "ignore") {
            INIT_MULTIPLIER(num_multiplier, num);
            INIT_MULTIPLIER(spatial_multiplier, spatial_dim);

            //  get buffer
            stddev = ws()->GetBuffer();
            stddev->ReshapeLike(input(0));

            auto* dYdata = input(-1).template data<T, Context>();
            auto* dXdata = output(0)->template mutable_data<T, Context>();
            auto* Std_data = stddev->template mutable_data<T, Context>();
            auto* Sdata = input(3).template data<T, Context>();
            auto* hVar_data = input(2).template data<T, Context>();
            auto* tVar_data = var->template mutable_data<T, Context>();
            auto* SMul_data = spatial_multiplier->template data<T, Context>();
            auto* NMul_data = num_multiplier->template data<T, Context>();
            auto* NByC_data = num_by_chans.template mutable_data<T, Context>();

            //  use the moving average var
            ctx().template Copy<T, Context, Context>(var->count(), tVar_data, hVar_data);
            math::AddScalar<T, Context>(var->count(), this->eps, tVar_data);
            math::Sqrt<T, Context>(var->count(), tVar_data, tVar_data);

            //  divide scale by std
            math::Div<T, Context>(var->count(), Sdata, tVar_data, tVar_data);

            //  broadcast
            math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, num, channels, 1,
                                                                            1.0,
                                                           NMul_data, tVar_data,
                                                                            0.0,
                                                                     NByC_data);
            math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, nbychans, spatial_dim, 1,
                                                                                    1.0,
                                                                   NByC_data, SMul_data,
                                                                                    0.0,
                                                                              Std_data);
            //  elementwise multiply top grad with(slope / std)
            math::Mul<T, Context>(output(0)->count(), dYdata, Std_data, dXdata);

            //  release buffer
            ws()->ReleaseBuffer(stddev);
        }
        return;
    }

    if (output(0)->name() != "ignore" ||
        output(1)->name() != "ignore" ||
        output(2)->name() != "ignore") {

        auto* dYdata = input(-1).template data<T, Context>();
        auto* dXdata = output(0)->template mutable_data<T, Context>();
        auto* Xdata = input(0).template data<T, Context>();
        auto* Sdata = input(3).template data<T, Context>();
        auto* dSdata = output(1)->template mutable_data<T, Context>();
        auto* dBdata = output(2)->template mutable_data<T, Context>();
        auto* tMean_data = mean->template data<T, Context>();
        auto* tVar_data = var->template data<T, Context>();

        CUDNN_CHECK(cudnnBatchNormalizationBackward(cudnn_handle(),
                                           CUDNN_BATCHNORM_SPATIAL,
                                                 CUDNNType<T>::one,
                                                CUDNNType<T>::zero,
                                                 CUDNNType<T>::one,
                                                 CUDNNType<T>::one,
                                                output_desc, Xdata,
                                                input_desc, dYdata,
                                               output_desc, dXdata,
                                                           bn_desc,
                                                             Sdata,
                                                            dSdata,
                                                            dBdata,
                                                         this->eps,
                                                        tMean_data,
                                                       tVar_data));
    }
}

template <class Context> template <typename T>
void CuDNNBNGradientOp<Context>::PerActivationRunWithType() {
    Tensor x_reshape;
    x_reshape.Reshape(vector<TIndex>({ num, channels, 1, 1 }));
    cudnnSetTensorDesc<T>(&input_desc, &x_reshape);
    cudnnSetTensorDesc<T>(&output_desc, &x_reshape);
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bn_desc, output_desc, CUDNN_BATCHNORM_PER_ACTIVATION));

    if (use_global_stats) {
        if (output(0)->name() != "ignore") {
            INIT_MULTIPLIER(num_multiplier, num);
            INIT_MULTIPLIER(spatial_multiplier, spatial_dim);

            //  get buffer
            stddev = ws()->GetBuffer();
            stddev->ReshapeLike(input(0));

            auto* dYdata = input(-1).template data<T, Context>();
            auto* dXdata = output(0)->template mutable_data<T, Context>();
            auto* Std_data = stddev->template mutable_data<T, Context>();
            auto* Sdata = input(3).template data<T, Context>();
            auto* hVar_data = input(2).template data<T, Context>();
            auto* tVar_data = var->template mutable_data<T, Context>();
            auto* SMul_data = spatial_multiplier->template data<T, Context>();
            auto* NMul_data = num_multiplier->template data<T, Context>();
            auto* NByC_data = num_by_chans.template mutable_data<T, Context>();

            //  use the moving average var
            ctx().template Copy<T, Context, Context>(var->count(), tVar_data, hVar_data);
            math::AddScalar<T, Context>(var->count(), this->eps, tVar_data);
            math::Sqrt<T, Context>(var->count(), tVar_data, tVar_data);

            //  divide scale by std
            math::Div<T, Context>(var->count(), Sdata, tVar_data, tVar_data);

            //  broadcast
            math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, num, channels, 1,
                                                                            1.0,
                                                           NMul_data, tVar_data,
                                                                            0.0,
                                                                     NByC_data);
            math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, nbychans, spatial_dim, 1,
                                                                                    1.0,
                                                                   NByC_data, SMul_data,
                                                                                    0.0,
                                                                              Std_data);
            //  elementwise multiply top grad with(slope / std)
            math::Mul<T, Context>(output(0)->count(), dYdata, Std_data, dXdata);

            //  release buffer
            ws()->ReleaseBuffer(stddev);
        }
        return;
    }

    if (output(0)->name() != "ignore" ||
        output(1)->name() != "ignore" ||
        output(2)->name() != "ignore") {

        auto* dYdata = input(-1).template data<T, Context>();
        auto* dXdata = output(0)->template mutable_data<T, Context>();
        auto* Xdata = input(0).template data<T, Context>();
        auto* Sdata = input(3).template data<T, Context>();
        auto* dSdata = output(1)->template mutable_data<T, Context>();
        auto* dBdata = output(2)->template mutable_data<T, Context>();
        auto* tMean_data = mean->template data<T, Context>();
        auto* tVar_data = var->template data<T, Context>();

        CUDNN_CHECK(cudnnBatchNormalizationBackward(cudnn_handle(),
                                    CUDNN_BATCHNORM_PER_ACTIVATION,
                                                 CUDNNType<T>::one,
                                                CUDNNType<T>::zero,
                                                 CUDNNType<T>::one,
                                                 CUDNNType<T>::one,
                                                output_desc, Xdata,
                                                input_desc, dYdata,
                                               output_desc, dXdata,
                                                           bn_desc,
                                                             Sdata,
                                                            dSdata,
                                                            dBdata,
                                                         this->eps,
                                                        tMean_data,
                                                       tVar_data));
    }
}

template <class Context>
void CuDNNBNGradientOp<Context>::RunOnDevice() {
    num = input(0).dim(0); channels = input(0).dim(1);
    spatial_dim = input(0).count(2); nbychans = num * channels;

    mean = ws()->GetTensor("_t_" + anchor() + "_bn_mean");
    var = ws()->GetTensor("_t_" + anchor() + "_bn_var");
    num_by_chans.Reshape(vector<TIndex>(1, nbychans));

    output(0)->ReshapeLike(input(0));   // dX
    output(1)->ReshapeLike(input(3));   // dScale
    output(2)->ReshapeLike(input(3));   // dBias

    if (this->use_stats == -1) use_global_stats = phase() == "TEST" ? true : false;
    else use_global_stats = this->use_stats == 1 ? true : false;

    if (input(0).template IsType<float>()) {
        if (input(0).ndim() == 4) SpatialRunWithType<float>();
        else if (input(0).ndim() == 2) PerActivationRunWithType<float>();
        else LOG(FATAL) << "The ndim of input tensor must be 2 or 4.";
    } else { LOG(FATAL) << "Unsupported input types."; }
}

template <class Context>
void BNGradientOp<Context>::ShareGradient() {
    if (use_global_stats) {
        if (output(0)->name() != "ignore") {
            Tensor* dX = ws()->GetBuffer("Grad");
            output(0)->Replace(*dX);
        }
    } else {
        if (output(0)->name() != "ignore" ||
            output(1)->name() != "ignore" ||
            output(2)->name() != "ignore") {
            Tensor* dX = ws()->GetBuffer("Grad");
            output(0)->Replace(*dX);
        }
    }
}

DEPLOY_CPU(BNGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(BNGradient);
#endif
OPERATOR_SCHEMA(BNGradient).NumInputs(5).NumOutputs(3);
DEPLOY_CUDNN(BNGradient);

}    // namespace dragon

#endif

#endif  // WITH_CUDNN