#ifdef WITH_CUDNN

#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/op_kernel.h"
#include "operators/vision/conv_transpose_op.h"

namespace dragon {

#define WORKSPACE_LIMIT_BYTES 1024 * 1024 * 1024 // 1G

template <class Context>
void CuDNNConvTranspose2dOp<Context>::SetConvDescFromInputs() {
    if (XIsType(X(0), float)) {
#if CUDNN_VERSION_MIN(6, 0, 0)
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
            conv_desc_,
            pad_l_[0], pad_l_[1],
            stride_[0], stride_[1],
            dilation_[0], dilation_[1],
            CUDNN_CROSS_CORRELATION,
            CUDNN_DATA_FLOAT
        ));
#else
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
            conv_desc_,
            pad_l_[0], pad_l_[1],
            stride_[0], stride_[1],
            1, 1,
            CUDNN_CROSS_CORRELATION
        ));
#endif
    } else if (XIsType(X(0), float16)) {
#if CUDNN_VERSION_MIN(6, 0, 0)
        compute_type_ = CUDNN_DATA_FLOAT;
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
            conv_desc_,
            pad_l_[0], pad_l_[1],
            stride_[0], stride_[1],
            dilation_[0], dilation_[1],
            CUDNN_CROSS_CORRELATION,
            compute_type_
        ));
#else
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
            conv_desc_,
            pad_l_[0], pad_l_[1],
            stride_[0], stride_[1],
            1, 1,
            CUDNN_CROSS_CORRELATION
        ));
#endif
    }
#if CUDNN_VERSION_MIN(7, 0, 0)
    CUDNN_CHECK(
        cudnnSetConvolutionGroupCount(
            conv_desc_, group_
        )
    );
    if (enable_tensor_core_)
        CUDNN_CHECK(
            cudnnSetConvolutionMathType(
                conv_desc_,
                CUDNN_TENSOR_OP_MATH
            )
        );
#endif
}

template <class Context> template <typename T>
void CuDNNConvTranspose2dOp<Context>::ResetDesc() {
    bool output_changed = (Y(0)->dims() != output_dims_);
    bool filter_changed = (X(1).dims() != filter_dims_);
    if (output_changed || filter_changed) {
        if (output_changed) {
            // Determine the input & output shape
            output_dims_ = Y(0)->dims();
            CuDNNSetTensor4dDescWithGroup<T>(
                &input_desc_,
                data_format(),
                X(0).dims(),
                cudnn_group_
            );
            CuDNNSetTensor4dDescWithGroup<T>(
                &output_desc_,
                data_format(),
                Y(0)->dims(),
                cudnn_group_
            );
            if (HasBias()) {
                CuDNNSetTensor4dDesc<T>(
                    &output2b_desc_,
                    data_format(),
                    Y(0)->dims()
                );
            }
            // Determine the misc
            if (data_format() == "NCHW") {
                x_ofs_ = X(0).count(1) / cudnn_group_;
                y_ofs_ = Y(0)->count(1) / cudnn_group_;
            } else if (data_format() == "NHWC") {
                x_ofs_ = X(0).dim(-1) / cudnn_group_;
                y_ofs_ = Y(0)->dim(-1) / cudnn_group_;
            }
        }
        if (filter_changed) {
#if CUDNN_VERSION_MIN(5, 0, 0)
            CUDNN_CHECK(cudnnSetFilter4dDescriptor(
                filter_desc_,
                CuDNNType<T>::type,
                format_,
                channels_ / cudnn_group_,
                num_output_ / group_,
                kshape_[0], kshape_[1]
            ));
#else
            CUDNN_CHECK(cudnnSetFilter4dDescriptor_v4(
                filter_desc_,
                CuDNNType<T>::type,
                format_,
                channels_ / cudnn_group_,
                num_output_ / group_,
                kshape_[0], kshape_[1]
            ));
#endif
            // Determine the bias shape
            if (HasBias()) {
                if (data_format() == "NCHW") {
                    CuDNNSetTensor4dDesc<T>(
                        &bias_desc_, data_format(),
                        vec64_t({ 1, num_output_, 1, 1 })
                    );
                } else if (data_format() == "NHWC") {
                    CuDNNSetTensor4dDesc<T>(
                        &bias_desc_, data_format(),
                        vec64_t({ 1, 1, 1, num_output_ })
                    );
                }
            }
        }

        SetConvDescFromInputs();

        // Now, Select the appropriate algorithm
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
            ctx()->cudnn_handle(),
            filter_desc_,
            input_desc_,
            conv_desc_,
            output_desc_,
            CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
            WORKSPACE_LIMIT_BYTES,
            &fwd_algo_
        ));

        CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
            ctx()->cudnn_handle(),
            filter_desc_,
            input_desc_,
            conv_desc_,
            output_desc_,
            fwd_algo_,
            &fwd_data_size_
        ));
    }
}

template <class Context> template <typename T>
void CuDNNConvTranspose2dOp<Context>::RunImpl() {
    TENSOR_FILL(X(1), w_shape_);
    if (HasBias()) TENSOR_FILL(X(2), b_shape_);

    ResetDesc<T>();

    auto* x = X(0).template data<T, Context>();
    auto* w = X(1).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    auto* scratch = ws()
        ->template data<Context>
            ({ fwd_data_size_ })[0];

    for (int g = 0; g < cudnn_group_; g++) {
        CUDNN_CHECK(cudnnConvolutionBackwardData(
            ctx()->cudnn_handle(),
            CuDNNType<T>::one,
            filter_desc_, w + w_ofs_ * g,
            input_desc_, x + x_ofs_ * g,
            conv_desc_, fwd_algo_,
            scratch, fwd_data_size_,
            CuDNNType<T>::zero,
            output_desc_, y + y_ofs_ * g
        ));
    }

    if (HasBias()) {
        auto* b = X(2).template data<T, Context>();
        CUDNN_CHECK(cudnnAddTensor(
            ctx()->cudnn_handle(),
            CuDNNType<T>::one,
            bias_desc_, b,
            CuDNNType<T>::one,
            output2b_desc_, y
        ));
    }
}

template <class Context>
void CuDNNConvTranspose2dOp<Context>::RunOnDevice() {
#if CUDNN_VERSION_MAX(6, 0, 0)
    for (int i = 0; i < dilation_.size(); i++)
        if (dilation_[i] != 1)
            return ConvTranspose2dOp<Context>::RunOnDevice();
#endif
    ConvOpBase<Context>::Reshape();

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

DEPLOY_CUDNN(ConvTranspose2d);

template <class Context>
void CuDNNConvTranspose2dGradientOp<Context>::SetConvDescFromInputs() {
    if (XIsType(X(0), float)) {
#if CUDNN_VERSION_MIN(6, 0, 0)
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
            conv_desc_,
            pad_l_[0], pad_l_[1],
            stride_[0], stride_[1],
            dilation_[0], dilation_[1],
            CUDNN_CROSS_CORRELATION,
            CUDNN_DATA_FLOAT
        ));
#else
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
            conv_desc_,
            pad_l_[0], pad_l_[1],
            stride_[0], stride_[1],
            1, 1,
            CUDNN_CROSS_CORRELATION
        ));
#endif
    } else if (XIsType(X(0), float16)) {
#if CUDNN_VERSION_MIN(6, 0, 0)
        compute_type_ = CUDNN_DATA_FLOAT;
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
            conv_desc_,
            pad_l_[0], pad_l_[1],
            stride_[0], stride_[1],
            dilation_[0], dilation_[1],
            CUDNN_CROSS_CORRELATION,
            compute_type_
        ));
#else
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
            conv_desc_,
            pad_l_[0], pad_l_[1],
            stride_[0], stride_[1],
            1, 1,
            CUDNN_CROSS_CORRELATION
        ));
#endif
    }
#if CUDNN_VERSION_MIN(7, 0, 0)
    CUDNN_CHECK(
        cudnnSetConvolutionGroupCount(
            conv_desc_, group_
        )
    );
    if (enable_tensor_core_)
        CUDNN_CHECK(
            cudnnSetConvolutionMathType(
                conv_desc_,
                CUDNN_TENSOR_OP_MATH
            )
        );
#endif
}

template <class Context> template <typename T>
void CuDNNConvTranspose2dGradientOp<Context>::ResetDesc() {
    bool output_changed = (X(-1).dims() != output_dims_);
    bool filter_changed = (X(1).dims() != filter_dims_);
    if (output_changed || filter_changed) {
        if (output_changed) {
            // Determine the input & output shape
            output_dims_ = X(-1).dims();
            CuDNNSetTensor4dDescWithGroup<T>(
                &input_desc_,
                data_format(),
                X(-1).dims(),
                cudnn_group_
            );
            CuDNNSetTensor4dDescWithGroup<T>(
                &output_desc_,
                data_format(),
                X(0).dims(),
                cudnn_group_
            );
            if (HasBias()) {
                CuDNNSetTensor4dDesc<T>(
                    &input2b_desc_,
                    data_format(),
                    X(-1).dims()
                );
            }
            // Determine the misc
            if (data_format() == "NCHW") {
                x_ofs_ = X(0).stride(0) / cudnn_group_;
                y_ofs_ = X(-1).stride(0) / cudnn_group_;
            } else if (data_format() == "NHWC") {
                x_ofs_ = X(0).dim(-1) / cudnn_group_;
                y_ofs_ = X(-1).dim(-1) / cudnn_group_;
            }
        }
        if (filter_changed) {
#if CUDNN_VERSION_MIN(5, 0, 0)
            CUDNN_CHECK(cudnnSetFilter4dDescriptor(
                filter_desc_,
                CuDNNType<T>::type,
                format_,
                channels_ / cudnn_group_,
                num_output_ / group_,
                kshape_[0], kshape_[1]
            ));
#else
            CUDNN_CHECK(cudnnSetFilter4dDescriptor_v4(
                filter_desc,
                CuDNNType<T>::type,
                format_,
                channels_ / cudnn_group_,
                num_output_ / group_,
                kshape_[0], kshape_[1]
            ));
#endif
            // Determine the bias shape
            if (HasBias()) {
                if (data_format() == "NCHW") {
                    CuDNNSetTensor4dDesc<T>(
                        &bias_desc_, data_format(),
                        vec64_t({ 1, num_output_, 1, 1 })
                    );
                } else if (data_format() == "NHWC") {
                    CuDNNSetTensor4dDesc<T>(
                        &bias_desc_, data_format(),
                        vec64_t({ 1, 1, 1, num_output_ })
                    );
                }
            }
        }

        SetConvDescFromInputs();

        // Now, Select the appropriate algorithm
        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(
            ctx()->cudnn_handle(),
            input_desc_,
            output_desc_,
            conv_desc_,
            filter_desc_,
            CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
            WORKSPACE_LIMIT_BYTES,
            &bwd_filter_algo_
        ));

        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            ctx()->cudnn_handle(),
            input_desc_,
            output_desc_,
            conv_desc_,
            filter_desc_,
            bwd_filter_algo_,
            &bwd_filter_size_
        ));

        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
            ctx()->cudnn_handle(),
            input_desc_,
            filter_desc_,
            conv_desc_,
            output_desc_,
            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
            WORKSPACE_LIMIT_BYTES,
            &bwd_data_algo_
        ));

        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
            ctx()->cudnn_handle(),
            input_desc_,
            filter_desc_,
            conv_desc_,
            output_desc_,
            bwd_data_algo_,
            &bwd_data_size_
        ));
    }
}

template <class Context> template <typename T>
void CuDNNConvTranspose2dGradientOp<Context>::RunImpl() {
    ResetDesc<T>();

    auto* scratch = ws()
        ->template data<Context>
        ({ std::max(
            bwd_data_size_,
            bwd_filter_size_
        )})[0];

    auto* dy = X(-1).template data<T, Context>();

    if (Y(2)->name() != "NULL") {
        auto* db = Y(2)->template mutable_data<T, Context>();
        CUDNN_CHECK(cudnnConvolutionBackwardBias(
            ctx()->cudnn_handle(),
            CuDNNType<T>::one,
            input2b_desc_, dy,
            CuDNNType<T>::zero,
            bias_desc_, db
        ));
    }

    for (int g = 0; g < cudnn_group_; g++) {
        if (Y(1)->name() != "NULL") {
            auto* x = X(0).template data<T, Context>();
            auto* dw = Y(1)->template mutable_data<T, Context>();
            CUDNN_CHECK(cudnnConvolutionBackwardFilter(
                ctx()->cudnn_handle(),
                CuDNNType<T>::one,
                input_desc_, dy + y_ofs_ * g,
                output_desc_, x + x_ofs_ * g,
                conv_desc_,
                bwd_filter_algo_,
                scratch, bwd_filter_size_,
                CuDNNType<T>::zero,
                filter_desc_, dw + w_ofs_ * g
            ));
        }
        if (Y(0)->name() != "NULL") {
            auto* w = X(1).template data<T, Context>();
            auto* dx = Y(0)->template mutable_data<T, Context>();
            CUDNN_CHECK(cudnnConvolutionForward(
                ctx()->cudnn_handle(),
                CuDNNType<T>::one,
                input_desc_, dy + y_ofs_ * g,
                filter_desc_, w + w_ofs_ * g,
                conv_desc_,
                bwd_data_algo_,
                scratch, bwd_data_size_,
                CuDNNType<T>::zero,
                output_desc_, dx + x_ofs_ * g
            ));
        }
    }
}

template <class Context>
void CuDNNConvTranspose2dGradientOp<Context>::RunOnDevice() {
#if CUDNN_VERSION_MAX(6, 0, 0)
    for (int i = 0; i < dilation_.size(); i++)
        if (dilation_[i] != 1)
            return ConvTranspose2dGradientOp<Context>::RunOnDevice();
#endif
    ConvOpBase<Context>::Reshape(true);

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

DEPLOY_CUDNN(ConvTranspose2dGradient);

}  // namespace dragon

#endif  // WITH_CUDNN