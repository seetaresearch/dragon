#ifdef WITH_CUDNN

#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/math_functions.h"
#include "operators/arithmetic/affine_op.h"

#if CUDNN_VERSION_MIN(6, 0, 0)

namespace dragon {

template <class Context> template <typename DT, typename CT>
void CuDNNAffineOp<Context>::RunWithType() {
    this->template ResetDesc<DT>(Input(0));
    const auto& dim_start = Input(0).dims().begin() + start_axis;
    const auto& dim_end = dim_start + num_axes;
    vector<TIndex> param_dims(dim_start, dim_end);
    TENSOR_FILL_WITH_TYPE(Input(1), param_dims, DT);
    auto* Xdata = Input(0).template data<DT, Context>();
    auto* Adata = Input(1).template data<DT, Context>();
    auto* Ydata = Output(0)->template mutable_data<DT, Context>();

    // Y = alpha * X
    CUDNN_CHECK(cudnnSetOpTensorDescriptor(
        mul_desc, CUDNN_OP_TENSOR_MUL,
            CUDNNType<CT>::type, CUDNN_PROPAGATE_NAN));
    CUDNN_CHECK(cudnnOpTensor(
        ctx()->cudnn_handle(), mul_desc,
            CUDNNType<DT>::one, input_desc, Xdata,
                CUDNNType<DT>::one, param_desc, Adata,
                    CUDNNType<DT>::zero, input_desc, Ydata));

    // Y += beta
    if (InputSize() > 2) {
        TENSOR_FILL_WITH_TYPE(Input(2), param_dims, DT);
        auto* Bdata = Input(2).template data<DT, Context>();
        CUDNN_CHECK(cudnnSetOpTensorDescriptor(
            add_desc, CUDNN_OP_TENSOR_ADD,
                CUDNNType<CT>::type, CUDNN_PROPAGATE_NAN));
        CUDNN_CHECK(cudnnOpTensor(
            ctx()->cudnn_handle(), add_desc,
                CUDNNType<DT>::one, input_desc, Ydata,
                    CUDNNType<DT>::one, param_desc, Bdata,
                        CUDNNType<DT>::zero, input_desc, Ydata));
    }
}

template <class Context>
void CuDNNAffineOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float, float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16, float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CUDNN(Affine);

template <class Context> template <typename DT, typename CT>
void CuDNNAffineGradientOp<Context>::RunWithType() {
    this->template ResetDesc<DT>(Input(-1));
    outer_dim = Input(-1).count(0, start_axis);
    inner_dim = Input(-1).count(start_axis + num_axes);
    scale_dim = Input(1).count();
    sum_dim = std::max(outer_dim, inner_dim);
    dim = scale_dim * inner_dim;
    Output(0)->ReshapeLike(Input(-1));

    auto* dYdata = Input(-1).template data<DT, Context>();
    auto* Adata = Input(1).template data<DT, Context>();
    auto* dXdata = Output(0)->template mutable_data<DT, Context>();

    CUDNN_CHECK(cudnnSetOpTensorDescriptor(
        mul_desc, CUDNN_OP_TENSOR_MUL,
            CUDNNType<CT>::type, CUDNN_PROPAGATE_NAN));

    // dA = X * dY
    if (Output(1)->name() != "ignore") {
        Output(1)->ReshapeLike(Input(1));
        auto* Xdata = Input(0).template data<DT, Context>();
        auto* dAdata = Output(1)->template mutable_data<DT, Context>(ctx());
        // Eltwise
        if (Input(0).count() == Input(1).count()) {
            CUDNN_CHECK(cudnnOpTensor(
                ctx()->cudnn_handle(), mul_desc,
                    CUDNNType<DT>::one, input_desc, Xdata,
                        CUDNNType<DT>::one, input_desc, dYdata,
                            CUDNNType<DT>::one, param_desc, dAdata));
        } else {
            CUDNN_CHECK(cudnnOpTensor(
                ctx()->cudnn_handle(), mul_desc,
                    CUDNNType<DT>::one, input_desc, Xdata,
                        CUDNNType<DT>::one, input_desc, dYdata,
                            CUDNNType<DT>::zero, input_desc, dXdata));
            ComputeScaleGradient_v2<DT>(dXdata, dAdata);
        }
    }

    // dB = dY
    if (Output(2)->name() != "ignore") {
        Output(2)->ReshapeLike(Input(1));
        auto* dBdata = Output(2)->template mutable_data<DT, Context>(ctx());
        // Eltwise
        if (Input(-1).count() == Input(1).count()) {
            math::Axpy<DT, Context>(Output(2)->count(),
                1.f, dYdata, dBdata, ctx());
        } else {
            ComputeBiasGradient_v2<DT>(dYdata, dBdata);
        }
    }

    // dX = alpha * dY
    CUDNN_CHECK(cudnnOpTensor(
        ctx()->cudnn_handle(), mul_desc,
            CUDNNType<DT>::one, input_desc, dYdata,
                CUDNNType<DT>::one, param_desc, Adata,
                    CUDNNType<DT>::zero, input_desc, dXdata));
}

template <class Context> template <typename DT, typename CT>
void CuDNNAffineGradientOp<Context>::ComputeScaleGradient(
    DT*                     dYxX,
    DT*                     dA) {
#if CUDNN_VERSION_MIN(6, 0, 0)
    CUDNN_CHECK(cudnnSetReduceTensorDescriptor(
        reduce_desc, CUDNN_REDUCE_TENSOR_ADD,
            CUDNNType<CT>::type, CUDNN_PROPAGATE_NAN,
                CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));
    size_t workspace_size = 0;
    CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
        ctx()->cudnn_handle(), reduce_desc,
            input_desc, param_desc, &workspace_size));
    auto* WSdata = ws()->template caches<Context>({ workspace_size })[0];;
    CUDNN_CHECK(cudnnReduceTensor(
        ctx()->cudnn_handle(), reduce_desc,
            nullptr, 0, WSdata, workspace_size,
                CUDNNType<DT>::one, input_desc, dYxX,
                    CUDNNType<DT>::one, param_desc, dA));
#endif
}

template <class Context> template <typename T>
void CuDNNAffineGradientOp<Context>::ComputeScaleGradient_v2(
    T*                      dYxX,
    T*                      dA) {
    DECLARE_MULTIPLIER(multiplier, sum_dim);
    sum_result.Reshape({ outer_dim * scale_dim });

    T* SRes_data = nullptr;
    // Reduce inner dimensions
    if (inner_dim == 1) {
        SRes_data = dYxX;
    } else {
        SRes_data = (outer_dim == 1) ?
            dA : sum_result.template mutable_data<T, Context>();
        math::Gemv<T, Context>(
            CblasNoTrans, sum_result.count(), inner_dim,
                1.f, dYxX, multiplier,
                    SRes_data == dA ? 1.f : 0.f, SRes_data, ctx());
    }
    // Reduce outer dimensions
    if (outer_dim != 1) {
        math::Gemv<T, Context>(
            CblasTrans, outer_dim, scale_dim,
                1.f, SRes_data, multiplier,
                    1.f, dA, ctx());
    }
}

template <class Context> template <typename DT, typename CT>
void CuDNNAffineGradientOp<Context>::ComputeBiasGradient(
    const DT*               dY,
    DT*                     dB) {
#if CUDNN_VERSION_MIN(6, 0, 0)
    CUDNN_CHECK(cudnnSetReduceTensorDescriptor(
        reduce_desc, CUDNN_REDUCE_TENSOR_ADD,
            CUDNNType<CT>::type, CUDNN_PROPAGATE_NAN,
                CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));
    size_t workspace_size = 0;
    CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
        ctx()->cudnn_handle(), reduce_desc,
            input_desc, param_desc, &workspace_size));
    auto* WSdata = ws()->template caches<Context>({ workspace_size })[0];
    CUDNN_CHECK(cudnnReduceTensor(
        ctx()->cudnn_handle(), reduce_desc,
            nullptr, 0, WSdata, workspace_size,
                CUDNNType<DT>::one, input_desc, dY,
                    CUDNNType<DT>::one, param_desc, dB));
#endif
}

template <class Context> template <typename T>
void CuDNNAffineGradientOp<Context>::ComputeBiasGradient_v2(
    const T*        dY,
    T*              dB) {
    DECLARE_MULTIPLIER(multiplier, inner_dim);
    for (int n = 0; n < outer_dim; n++) {
        math::Gemv<T, Context>(
            CblasNoTrans, scale_dim, inner_dim,
                1.f, dY, multiplier,
                    1.f, dB, ctx());
        dY += dim;
    }
}

template <class Context>
void CuDNNAffineGradientOp<Context>::RunOnDevice() {
    if (XIsType(Input(-1), float)) RunWithType<float, float>();
    else if (XIsType(Input(-1), float16)) RunWithType<float16, float>();
    else LOG(FATAL) << DTypeHelper(Input(-1), { "float32", "float16" });
}

DEPLOY_CUDNN(AffineGradient);

}  // namespace dragon

#endif

#endif  // WITH_CUDNN