#ifdef WITH_CUDNN

#include "operators/arithmetic/affine_op.h"
#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/math_functions.h"

namespace dragon {

template <class Context> template <typename T>
void CuDNNAffineOp<Context>::RunWithType() {
    this->template ResetDesc<T>();
    const auto& dim_start = Input(0).dims().begin() + start_axis;
    const auto& dim_end = dim_start + num_axes;
    vector<TIndex> param_dims(dim_start, dim_end);
    TENSOR_FILL(Input(1), param_dims);
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Adata = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    //  y = alpha * x
    CUDNN_CHECK(cudnnSetOpTensorDescriptor(
        mul_desc, CUDNN_OP_TENSOR_MUL,
            CUDNNType<T>::type, CUDNN_PROPAGATE_NAN));
    CUDNN_CHECK(cudnnOpTensor(
        ctx().cudnn_handle(), mul_desc,
            CUDNNType<T>::one, input_desc, Xdata,
                CUDNNType<T>::one, param_desc, Adata,
                    CUDNNType<T>::zero, input_desc, Ydata));

    //  y += beta
    if (InputSize() > 2) {
        TENSOR_FILL(Input(2), param_dims);
        auto* Bdata = Input(2).template data<T, Context>();
        CUDNN_CHECK(cudnnSetOpTensorDescriptor(
            add_desc, CUDNN_OP_TENSOR_ADD,
                CUDNNType<T>::type, CUDNN_PROPAGATE_NAN));
        CUDNN_CHECK(cudnnOpTensor(
            ctx().cudnn_handle(), add_desc,
                CUDNNType<T>::one, input_desc, Ydata,
                    CUDNNType<T>::one, param_desc, Bdata,
                        CUDNNType<T>::zero, input_desc, Ydata));
    }
}

template <class Context>
void CuDNNAffineOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CUDNN(Affine);

template <class Context> template <typename T>
void CuDNNAffineGradientOp<Context>::RunWithType() {
    this->template ResetDesc<T>();
    outer_dim = Input(0).count(0, start_axis);
    inner_dim = Input(0).count(start_axis + num_axes);
    scale_dim = Input(1).count();
    sum_dim = std::max(outer_dim, inner_dim);
    dim = scale_dim * inner_dim;
    Output(0)->ReshapeLike(Input(0));

    auto* dYdata = Input(-1).template data<T, Context>();
    auto* Adata = Input(1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();

    CUDNN_CHECK(cudnnSetOpTensorDescriptor(
        mul_desc, CUDNN_OP_TENSOR_MUL,
            CUDNNType<T>::type, CUDNN_PROPAGATE_NAN));

    //  da = x * dy
    if (Output(1)->name() != "ignore") {
        Output(1)->ReshapeLike(Input(1));
        auto* Xdata = Input(0).template data<T, Context>();
        auto* dAdata = Output(1)->template mutable_data<T, Context>();
        //  eltwise
        if (Input(0).count() == Input(1).count()) {
            CUDNN_CHECK(cudnnOpTensor(
                ctx().cudnn_handle(), mul_desc,
                    CUDNNType<T>::one, input_desc, Xdata,
                        CUDNNType<T>::one, input_desc, dYdata,
                            CUDNNType<T>::one, param_desc, dAdata));
        } else {
            CUDNN_CHECK(cudnnOpTensor(
                ctx().cudnn_handle(), mul_desc,
                    CUDNNType<T>::one, input_desc, Xdata,
                        CUDNNType<T>::one, input_desc, dYdata,
                            CUDNNType<T>::zero, input_desc, dXdata));
            ComputeScaleGradient_v2<T>(dXdata, dAdata);
        }
    }

    //  db = dy
    if (Output(2)->name() != "ignore") {
        Output(2)->ReshapeLike(Input(1));
        auto* dBdata = Output(2)->template mutable_data<T, Context>();
        //  eltwise
        if (Input(0).count() == Input(1).count()) {
            math::Axpy<T, Context>(Output(2)->count(),
                1.f, dYdata, dBdata, &ctx());
        } else {
            ComputeBiasGradient_v2<T>(dYdata, dBdata);
        }
    }

    //  dx = alpha * dy
    CUDNN_CHECK(cudnnOpTensor(
        ctx().cudnn_handle(), mul_desc,
            CUDNNType<T>::one, input_desc, dYdata,
                CUDNNType<T>::one, param_desc, Adata,
                    CUDNNType<T>::zero, input_desc, dXdata));
}

template <class Context> template <typename T>
void CuDNNAffineGradientOp<Context>::ComputeScaleGradient(
    T*              dYxX,
    T*              dA) {
#if CUDNN_VERSION_MIN(6, 0, 0)
    CUDNN_CHECK(cudnnSetReduceTensorDescriptor(
        reduce_desc, CUDNN_REDUCE_TENSOR_ADD,
            CUDNNType<T>::type, CUDNN_PROPAGATE_NAN,
                CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));
    size_t workspace_size = 0;
    CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
        ctx().cudnn_handle(), reduce_desc,
            input_desc, param_desc, &workspace_size));
    auto* WSdata = ws()->template caches<Context>({ workspace_size })[0];;
    CUDNN_CHECK(cudnnReduceTensor(
        ctx().cudnn_handle(), reduce_desc,
            nullptr, 0, WSdata, workspace_size,
                CUDNNType<T>::one, input_desc, dYxX,
                    CUDNNType<T>::one, param_desc, dA));
#endif
}

template <class Context> template <typename T>
void CuDNNAffineGradientOp<Context>::ComputeScaleGradient_v2(
    T*              dYxX,
    T*              dA) {
    DECLARE_MULTIPLIER(multiplier, sum_dim);
    sum_result.Reshape({ outer_dim * scale_dim });

    T* SRes_data = nullptr;
    if (inner_dim == 1) SRes_data = dYxX;
    else if (sum_result.count() == 1) {
        auto* dAC = Output(1)->template mutable_data<T, CPUContext>();
        T result = math::Dot<T, Context>(
            inner_dim, dYxX, multiplier, &ctx());
        *dAC += result;
    } else {
        SRes_data = (outer_dim == 1) ?
            dA : sum_result.template mutable_data<T, Context>();
        math::Gemv<T, Context>(
            CblasNoTrans, sum_result.count(), inner_dim,
                1.0, dYxX, multiplier,
                    SRes_data == dA ? 1.0 : 0.0, SRes_data, &ctx());
    }
    if (outer_dim != 1) {
        if (scale_dim == 1) {
            auto* dAC = Output(1)->template mutable_data<T, CPUContext>();
            T result = math::Dot<T, Context>(
                outer_dim, multiplier, SRes_data, &ctx());
            *dAC += result;
        } else {
            math::Gemv<T, Context>(
                CblasTrans, outer_dim, scale_dim,
                    1.0, SRes_data, multiplier,
                        1.0, dA, &ctx());
        }
    }
}

template <class Context> template <typename T>
void CuDNNAffineGradientOp<Context>::ComputeBiasGradient(
    const T*        dY,
    T*              dB) {
#if CUDNN_VERSION_MIN(6, 0, 0)
    CUDNN_CHECK(cudnnSetReduceTensorDescriptor(
        reduce_desc, CUDNN_REDUCE_TENSOR_ADD,
            CUDNNType<T>::type, CUDNN_PROPAGATE_NAN,
                CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));
    size_t workspace_size = 0;
    CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
        ctx().cudnn_handle(), reduce_desc,
            input_desc, param_desc, &workspace_size));
    auto* WSdata = ws()->template caches<Context>({ workspace_size })[0];
    CUDNN_CHECK(cudnnReduceTensor(
        ctx().cudnn_handle(), reduce_desc,
            nullptr, 0, WSdata, workspace_size,
                CUDNNType<T>::one, input_desc, dY,
                    CUDNNType<T>::one, param_desc, dB));
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
                1.0, dY, multiplier,
                    1.0, dB, &ctx());
        dY += dim;
    }
}

template <class Context>
void CuDNNAffineGradientOp<Context>::RunOnDevice() {
    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CUDNN(AffineGradient);

}    // namespace dragon

#endif    // WITH_CUDNN