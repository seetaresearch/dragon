#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/op_kernel.h"
#include "operators/norm/batch_norm_op.h"

#ifdef WITH_CUDNN

#if CUDNN_VERSION_MIN(5, 0, 0)

namespace dragon {

template <class Context> template <typename T>
void CuDNNBatchNormOp<Context>::RunImpl() {
    typedef typename CuDNNType<T>::BNParamType Tp;

    // Determine the bn desc
    if (X(0).ndim() == 2) {
        bn_mode_ = CUDNN_BATCHNORM_PER_ACTIVATION;
        Tensor fake_input;
        fake_input.Reshape(vec64_t({ N_, C_, 1, 1 }));
        CuDNNSetTensorDesc<T>(&input_desc_, &fake_input);
    } else {
        CHECK_GE((int)X(0).ndim(), 3)
            << "The number of dimensions should be at least 3.";
        bn_mode_ = CUDNN_BATCHNORM_SPATIAL;
#if CUDNN_VERSION_MIN(7, 0, 0)
        if (is_training_) bn_mode_ =
            CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#endif
        if (data_format() == "NCHW") {
            CuDNNSetTensorDesc<T>(&input_desc_, &X(0));
        } else if (data_format() == "NHWC") {
            switch (X(0).ndim()) {
                case 3:
                    CuDNNSetTensor3dDesc<T>(
                        &input_desc_,
                        data_format(),
                        &X(0)
                    ); break;
                case 4:
                    CuDNNSetTensor4dDesc<T>(
                        &input_desc_,
                        data_format(),
                        &X(0)
                    ); break;
                case 5:
                    CuDNNSetTensor5dDesc<T>(
                        &input_desc_,
                        data_format(),
                        &X(0)
                    ); break;
                default:
                    LOG(FATAL) << "Excepted 3d/4d/5d input at NHWC.";
            }
        }
    }

    // Derive the bn desc
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(
        bn_desc_, input_desc_, bn_mode_));

    TENSOR_FILL_WITH_TYPE(X(1), vec64_t({ C_ }), Tp);
    TENSOR_FILL_WITH_TYPE(X(2), vec64_t({ C_ }), Tp);
    TENSOR_FILL_WITH_TYPE(X(3), vec64_t({ C_ }), Tp);
    TENSOR_FILL_WITH_TYPE(X(4), vec64_t({ C_ }), Tp);

    auto* x = X(0).template data<T, Context>();
    auto* rm = X(1).template mutable_data<Tp, Context>();
    auto* rv = X(2).template mutable_data<Tp, Context>();
    auto* gamma = X(3).template data<Tp, Context>();
    auto* beta = X(4).template data<Tp, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    if (is_training_) {
        auto* sm = mean_->template mutable_data<Tp, Context>();
        auto* sv = var_->template mutable_data<Tp, Context>();
        auto mt = is_recomp_ ? 0.f : 1.f - this->momentum_;
        CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
            ctx()->cudnn_handle(),
            bn_mode_,
            CuDNNType<T>::one,
            CuDNNType<T>::zero,
            input_desc_, x,
            input_desc_, y,
            bn_desc_, gamma, beta,
            mt, rm, rv, eps64_, sm, sv
        ));
    } else {
        CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
            ctx()->cudnn_handle(),
            bn_mode_,
            CuDNNType<T>::one,
            CuDNNType<T>::zero,
            input_desc_, x,
            input_desc_, y,
            bn_desc_, gamma, beta,
            rm, rv, eps64_
         ));
    }
}

template <class Context>
void CuDNNBatchNormOp<Context>::Reshape() {
    // Determine the mode
    if (use_stats_ == -1) {
        is_training_ = phase() == "TRAIN" ? 1 : 0;
    } else {
        is_training_ = use_stats_ > 0 ? 0 : 1;
    }

    // Get the recomputing flag
    is_recomp_ = ws()
        ->GetTensor("/opt/recomp_flag")
        ->template data<bool, CPUContext>()[0];

    // Determine the data format
    auto axis = axis_;
    this->data_format_ = "NCHW";
    if (axis == -1) axis += X(0).ndim();
    if (axis + 1 == X(0).ndim())
        this->data_format_ = "NHWC";
    N_ = X(0).dim(0);
    C_ = X(0).dim(axis);

    // Create the shared resources
    mean_ = ws()
        ->CreateTensor(unique_name("sm"))
        ->Reshape({ C_ });

    var_ = ws()
        ->CreateTensor(unique_name("sv"))
        ->Reshape({ C_ });

    // Reshape
    Y(0)->ReshapeLike(X(0));
}

template <class Context>
void CuDNNBatchNormOp<Context>::RunOnDevice() {
    Reshape();

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

template <class Context>
void CuDNNBatchNormGradientOp<Context>::Reshape() {
    // Determine the mode
    if (use_stats_ == -1) {
        is_training_ = phase() == "TRAIN" ? 1 : 0;
    } else {
        is_training_ = use_stats_ > 0 ? 0 : 1;
    }

    // Determine the data format
    auto axis = axis_;
    this->data_format_ = "NCHW";
    if (axis == -1) axis += X(0).ndim();
    if (axis + 1 == X(0).ndim())
        this->data_format_ = "NHWC";
    N_ = X(0).dim(0);
    C_ = X(0).dim(axis);
    S_ = X(0).count() / N_ / C_;

    // Get the shared resources
    mean_ = ws()->GetTensor(unique_name("sm"));
    var_ = ws()->GetTensor(unique_name("sv"));

    // Reshape
    Y(0)->ReshapeLike(X(0));  // dx
    Y(1)->Reshape({ C_ });    // dgamma
    Y(2)->Reshape({ C_ });    // dbeta
}

template <class Context> template <typename T>
void CuDNNBatchNormGradientOp<Context>::TrainingImpl() {
    typedef typename CuDNNType<T>::BNParamType Tp;

    // Determine the bn desc
    if (X(0).ndim() == 2) {
        bn_mode_ = CUDNN_BATCHNORM_PER_ACTIVATION;
        Tensor fake_input;
        fake_input.Reshape(vec64_t({ N_, C_, 1, 1 }));
        CuDNNSetTensorDesc<T>(&input_desc_, &fake_input);
    } else {
        CHECK_GE((int)X(0).ndim(), 3)
            << "The number of dimensions should be at least 3.";
        bn_mode_ = CUDNN_BATCHNORM_SPATIAL;
#if CUDNN_VERSION_MIN(7, 0, 0)
        if (is_training_) bn_mode_ =
            CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#endif
        if (data_format() == "NCHW") {
            CuDNNSetTensorDesc<T>(&input_desc_, &X(-1));
        } else if (data_format() == "NHWC") {
            switch (X(0).ndim()) {
                case 3:
                    CuDNNSetTensor3dDesc<T>(
                        &input_desc_,
                        data_format(),
                        &X(-1)
                    ); break;
                case 4:
                    CuDNNSetTensor4dDesc<T>(
                        &input_desc_,
                        data_format(),
                        &X(-1)
                    ); break;
                case 5:
                    CuDNNSetTensor5dDesc<T>(
                        &input_desc_,
                        data_format(),
                        &X(-1)
                    ); break;
                default:
                    LOG(FATAL) << "Excepted 3d/4d/5d input at NHWC.";
            }
        }
    }

    // Derive the bn desc
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(
        bn_desc_, input_desc_, bn_mode_));

    auto* x = X(0).template data<T, Context>();
    auto* sm = mean_->template data<Tp, Context>();
    auto* sv = var_->template data<Tp, Context>();
    auto* gamma = X(3).template data<Tp, Context>();
    auto* dy = X(-1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();
    auto* dgamma = Y(1)->template mutable_data<Tp, Context>();
    auto* dbeta = Y(2)->template mutable_data<Tp, Context>();

    CUDNN_CHECK(cudnnBatchNormalizationBackward(
        ctx()->cudnn_handle(),
        bn_mode_,
        CuDNNType<T>::one,
        CuDNNType<T>::zero,
        CuDNNType<T>::one,
        CuDNNType<T>::zero,
        input_desc_, x,
        input_desc_, dy,
        input_desc_, dx,
        bn_desc_, gamma, dgamma, dbeta,
        eps64_, sm, sv
    ));
}

template <class Context> template <typename T>
void CuDNNBatchNormGradientOp<Context>::InferenceImpl() {
    typedef typename CuDNNType<T>::BNParamType Tp;

    auto* x = X(0).template data<T, Context>();
    auto* rm = X(1).template data<Tp, Context>();
    auto* rv = X(2).template data<Tp, Context>();
    auto* gamma = X(3).template data<Tp, Context>();
    auto* dy = X(-1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();
    auto* rsig = var_->template mutable_data<Tp, Context>();

    Tp* dgamma = nullptr, *dbeta = nullptr;

    // Gradient w.r.t. gamma or beta if necessary
    if (Y(1)->name() != "NULL" ||
        Y(2)->name() != "NULL") {
        dgamma = Y(1)->template mutable_data<Tp, Context>();
        dbeta = Y(2)->template mutable_data<Tp, Context>();
    }

    math::InvStd(C_, (float)eps64_, rv, rsig, ctx());

    kernel::BatchNormBackwardInference(
        N_, C_, S_,
        data_format(),
        x, rm, rsig, gamma, dy,
        dx, dgamma, dbeta, ctx()
    );
}

template <class Context>
void CuDNNBatchNormGradientOp<Context>::RunOnDevice() {
    Reshape();

    if (XIsType(X(0), float)) {
        if (is_training_) {
            TrainingImpl<float>();
        } else {
            InferenceImpl<float>();
        }
    } else if (XIsType(X(0), float16)) {
        if (is_training_) {
            TrainingImpl<float16>();
        } else {
            // We will support it some day -:)
            LOG(FATAL) << DTypeString(
                X(0), { "float32" }
            );
        }
    } else {
        LOG(FATAL) << DTypeString(X(0),
            { "float32", "float16" }
        );
    }
}

DEPLOY_CUDNN(BatchNorm);
DEPLOY_CUDNN(BatchNormGradient);

}  // namespace dragon

#endif  // CUDNN_VERSION_MIN(5, 0, 0)

#endif  // WITH_CUDNN