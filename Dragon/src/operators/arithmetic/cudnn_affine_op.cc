#ifdef WITH_CUDNN

#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/arithmetic/affine_op.h"

#if CUDNN_VERSION_MIN(6, 0, 0)

namespace dragon {

#define DETERMINE_RUNTIME_ARGS(X) \
    axis_ = OpArg<int64_t>("axis", 1); \
    num_axes_ = OpArg<int64_t>("num_axes", 1); \
    if (axis_ < 0) axis_ += X.ndim(); \
    if (num_axes_ < 0) num_axes_ = X.ndim() - axis_; \
    else if (num_axes_ == 0) num_axes_ = 1; \
    CHECK(axis_ >= 0 && axis_ + num_axes_ <= X.ndim())

template <class Context> template <typename T>
void CuDNNAffineOpBase<Context>::ResetDesc(const Tensor& X) {
    // Determine the runtime arguments
    DETERMINE_RUNTIME_ARGS(X);
    // Determine the input desc
    vec64_t input_dims = X.dims();
    // CuDNN requires ndimensions range from [4, 5]
    if (input_dims.size() < 4) input_dims.resize(4, 1);
    else if (input_dims.size() > 5)
        LOG(FATAL) << "CuDNN Affine the dimensions up to 5.";
    CuDNNSetTensorDesc<T>(&input_desc_, input_dims);
    // Determine the scale desc
    vec64_t param_dims(input_dims.size(), 1);
    for (int i = axis_; i < axis_ + num_axes_; i++)
        param_dims[i] = input_dims[i];
    CuDNNSetTensorDesc<T>(&param_desc_, param_dims);
}

template <class Context>
template <typename DT, typename CT>
void CuDNNAffineOp<Context>::RunImpl() {
    this->template ResetDesc<DT>(X(0));
    const auto& dim_start = X(0).dims().begin() + axis_;
    const auto& dim_end = dim_start + num_axes_;
    vec64_t param_dims(dim_start, dim_end);
    TENSOR_FILL_WITH_TYPE(X(1), param_dims, DT);

    auto* x = X(0).template data<DT, Context>();
    auto* alpha = X(1).template data<DT, Context>();
    auto* y = Y(0)->template mutable_data<DT, Context>();

    // Y = alpha * X
    CUDNN_CHECK(cudnnSetOpTensorDescriptor(
        mul_op_,
        CUDNN_OP_TENSOR_MUL,
        CuDNNType<CT>::type,
        CUDNN_PROPAGATE_NAN
    ));
    CUDNN_CHECK(cudnnOpTensor(
        ctx()->cudnn_handle(),
        mul_op_,
        CuDNNType<DT>::one,
        input_desc_, x,
        CuDNNType<DT>::one,
        param_desc_, alpha,
        CuDNNType<DT>::zero,
        input_desc_, y
    ));

    // Y += beta
    if (XSize() > 2) {
        TENSOR_FILL_WITH_TYPE(X(2), param_dims, DT);
        auto* beta = X(2).template data<DT, Context>();
        CUDNN_CHECK(cudnnSetOpTensorDescriptor(
            add_op_,
            CUDNN_OP_TENSOR_ADD,
            CuDNNType<CT>::type,
            CUDNN_PROPAGATE_NAN
        ));
        CUDNN_CHECK(cudnnOpTensor(
            ctx()->cudnn_handle(),
            add_op_,
            CuDNNType<DT>::one,
            input_desc_, y,
            CuDNNType<DT>::one,
            param_desc_, beta,
            CuDNNType<DT>::zero,
            input_desc_, y
        ));
    }
}

template <class Context>
void CuDNNAffineOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    if (XIsType(X(0), float)) {
        RunImpl<float, float>();
    } else if (XIsType(X(0), float16)) {
        RunImpl<float16, float>();
    } else {
        LOG(FATAL) << DTypeString(X(0),
            { "float32", "float16" }
        );
    }
}

template <class Context> template <typename DT, typename CT>
void CuDNNAffineGradientOp<Context>::RunImpl() {
    this->template ResetDesc<DT>(X(-1));
    scale_dim_ = X(1).count();
    outer_dim_ = X(-1).count(0, axis_);
    inner_dim_ = X(-1).count(axis_ + num_axes_);
    dim_ = scale_dim_ * inner_dim_;
    reduce_dim_ = std::max(outer_dim_, inner_dim_);

    Y(0)->ReshapeLike(X(-1));

    auto* alpha = X(1).template data<DT, Context>();
    auto* dy = X(-1).template mutable_data<DT, Context>();
    auto* dx = Y(0)->template mutable_data<DT, Context>();

    CUDNN_CHECK(cudnnSetOpTensorDescriptor(
        mul_op_,
        CUDNN_OP_TENSOR_MUL,
        CuDNNType<CT>::type,
        CUDNN_PROPAGATE_NAN
    ));

    // dA = X * dY
    if (Y(1)->name() != "NULL") {
        Y(1)->ReshapeLike(X(1));
        auto* x = X(0).template data<DT, Context>();
        auto* dalpha = Y(1)->template mutable_data<DT, Context>();
        // Eltwise
        if (X(0).count() == X(1).count()) {
            math::Mul(
                Y(0)->count(),
                dy, x,
                dalpha, ctx()
            );
        } else {
            math::Mul(
                Y(0)->count(),
                dy, x,
                dx, ctx()
            );
#if CUDNN_VERSION_MIN(6, 0, 0)
            /*!
             *  ReduceSum is faster and cleaner
             *  CuDNNReduce<DT, CT>(dx, dalpha);
             */
            Reduce(dx, dalpha);
#else
            Reduce(dx, dalpha);
#endif
        }
    }

    // dB = dY
    if (Y(2)->name() != "NULL") {
        Y(2)->ReshapeLike(X(1));
        auto* dbeta = Y(2)->template mutable_data<DT, Context>();
        // Eltwise
        if (X(-1).count() == X(1).count()) {
            ctx()->template Copy<DT, Context, Context>(
                Y(2)->count(), dbeta, dy);
        } else {
#if CUDNN_VERSION_MIN(6, 0, 0)
            /*!
             *  ReduceSum is faster and cleaner
             *  CuDNNReduce<DT, CT>(dy, dbeta);
             */
            Reduce(dy, dbeta);
#else
            Reduce(dy, dbeta);
#endif
        }
    }

    // dX = alpha * dY
    if (Y(0)->name() != "NULL") {
        CUDNN_CHECK(cudnnOpTensor(
            ctx()->cudnn_handle(),
            mul_op_,
            CuDNNType<DT>::one,
            input_desc_, dy,
            CuDNNType<DT>::one,
            param_desc_, alpha,
            CuDNNType<DT>::zero,
            input_desc_, dx
        ));
    }
}

template <class Context> template <typename DT, typename CT>
void CuDNNAffineGradientOp<Context>::CuDNNReduce(
    DT*                     x,
    DT*                     y) {
#if CUDNN_VERSION_MIN(6, 0, 0)
    CUDNN_CHECK(cudnnSetReduceTensorDescriptor(
        reduce_desc_,
        CUDNN_REDUCE_TENSOR_ADD,
        CuDNNType<CT>::type,
        CUDNN_PROPAGATE_NAN,
        CUDNN_REDUCE_TENSOR_NO_INDICES,
        CUDNN_32BIT_INDICES
    ));
    size_t workspace_size = 0;
    CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
        ctx()->cudnn_handle(),
        reduce_desc_,
        input_desc_,
        param_desc_,
        &workspace_size
    ));
    auto* scratch = ws()->template data
        <Context>({ workspace_size })[0];
    CUDNN_CHECK(cudnnReduceTensor(
        ctx()->cudnn_handle(),
        reduce_desc_,
        nullptr, 0,
        scratch, workspace_size,
        CuDNNType<DT>::one,
        input_desc_, x,
        CuDNNType<DT>::zero,
        param_desc_, y
    ));
#endif
}

template <class Context> template <typename T>
void CuDNNAffineGradientOp<Context>::Reduce(
    T*                      x,
    T*                      y) {
    vec32_t dims = {
        (int)outer_dim_,
        (int)scale_dim_,
        (int)inner_dim_,
    }, axes = { 0, 2 };
    kernel::ReduceSum(
        3, dims.data(),
        2, axes.data(),
        1.f, x,
        y, ctx()
    );
}

template <class Context>
void CuDNNAffineGradientOp<Context>::RunOnDevice() {
    if (XIsType(X(-1), float)) {
        RunImpl<float, float>();
    } else if (XIsType(X(-1), float16)) {
        RunImpl<float16, float>();
    } else {
        LOG(FATAL) << DTypeString(X(-1),
            { "float32", "float16" }
        );
    }
}

DEPLOY_CUDNN(Affine);
DEPLOY_CUDNN(AffineGradient);

#undef DETERMINE_RUNTIME_ARGS

}  // namespace dragon

#endif  // CUDNN_VERSION_MIN(6, 0, 0)

#endif  // WITH_CUDNN