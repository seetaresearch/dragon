#include "utils/math_functions.h"
#include "operators/arithmetic/matmul_op.h"

namespace dragon {

template <class Context> template <typename T>
void MatmulOp<Context>::RunImpl() {
    auto* a = X(0).template data<T, Context>();
    auto* b = X(1).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    for (int i = 0; i < batch_size_; ++i) {
        math::Gemm(
            transA_ ? CblasTrans : CblasNoTrans,
            transB_ ? CblasTrans : CblasNoTrans,
            M_, N_, K1_,
            1.f,
            a + i * A_stride_,
            b + i * B_stride_,
            0.f,
            y + i * Y_stride_, ctx()
        );
    }
}

template <class Context>
void MatmulOp<Context>::RunOnDevice() {
    CHECK_GE(X(0).ndim(), 2)
        << "\nTensor(" << X(0).name() + ") must be a matrix"
        << "(or rank > 2, representing batches of matrices).";
    CHECK_GE(X(1).ndim(), 2)
        << "\nTensor(" << X(1).name() + ") must be a matrix"
        << "(or rank > 2, representing batches of matrices).";

    M1_ = X(0).dim(-2), N1_ = X(0).dim(-1);
    M2_ = X(1).dim(-2), N2_ = X(1).dim(-1);

    M_  = transA_ ? N1_ : M1_;
    N_  = transB_ ? M2_ : N2_;
    K1_ = transA_ ? M1_ : N1_;
    K2_ = transB_ ? N2_ : M2_;

    A_stride_ = M1_ * N1_;
    B_stride_ = M2_ * N2_;
    Y_stride_ = M_ * N_;

    batch_size_ = X(0).count() / A_stride_;

    CHECK_EQ(K1_, K2_) << "\nTensor(" << X(0).name()
                       << "): " << X(0).DimString()
                       << " can not mul with Tensor"
                       << "(" << X(1).name() << "): "
                       << X(1).DimString();

    CHECK_EQ(batch_size_, X(1).count() / B_stride_)
        << "\nTensor(" << X(0).name()
        << "): " << X(0).DimString()
        << " can not mul with Tensor"
        << "(" << X(1).name() << "): "
        << X(1).DimString();

    auto out_shape = X(0).dims();
    out_shape[out_shape.size() - 2] = M_;
    out_shape[out_shape.size() - 1] = N_;

    Y(0)->Reshape(out_shape);

    if (XIsType(X(0), float16)) {
        RunImpl<float16>();
    } else if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else if (XIsType(X(0), double)) {
        RunImpl<double>();
    } else {
        LOG(FATAL) << DTypeString(X(0),
            { "float16", "float32", "float64" }
        );
    }
}

template <class Context> template <typename T>
void MatmulGradientOp<Context>::RunImpl() {
    auto* a  = X(0).template data<T, Context>();
    auto* b  = X(1).template data<T, Context>();
    auto* dy = X(2).template data<T, Context>();

    auto* da = Y(0)->name() == "NULL" ? nullptr :
               Y(0)->template mutable_data<T, Context>();
    auto* db = Y(1)->name() == "NULL" ? nullptr :
               Y(1)->template mutable_data<T, Context>();

    for (int i = 0; i < batch_size_; ++i) {
        if (Y(0)->name() != "NULL") {
            if (transA_) {
                math::Gemm(
                    transB_ ? CblasTrans : CblasNoTrans,
                    CblasTrans,
                    K1_, M_, N_,
                    1.f,
                    b + i * B_stride_,
                    dy + i * Y_stride_,
                    0.f,
                    da + i * A_stride_, ctx()
                );
            } else {
                math::Gemm(
                    CblasNoTrans,
                    transB_ ? CblasNoTrans : CblasTrans,
                    M_, K1_, N_,
                    1.f,
                    dy + i * Y_stride_,
                    b + i * B_stride_,
                    0.f, 
                    da + i * A_stride_, ctx()
                );
            }
        }
        if (Y(1)->name() != "NULL") {
            if (transB_) {
                math::Gemm(
                    CblasTrans,
                    transA_ ? CblasTrans : CblasNoTrans,
                    N_, K1_, M_,
                    1.f,
                    dy + i * Y_stride_,
                    a + i * A_stride_,
                    0.f,
                    db + i * B_stride_, ctx()
                );
            } else {
                math::Gemm(
                    transA_ ? CblasNoTrans : CblasTrans,
                    CblasNoTrans,
                    K1_, N_, M_,
                    1.f,
                    a + i * A_stride_,
                    dy + i * Y_stride_,
                    0.f,
                    db + i * B_stride_, ctx()
                );
            }
        }
    }
}

template <class Context>
void MatmulGradientOp<Context>::RunOnDevice() {
    CHECK_GE(X(0).ndim(), 2)
        << "\nTensor(" << X(0).name() + ") must be a matrix"
        << "(or rank > 2, representing batches of matrices).";
    CHECK_GE(X(1).ndim(), 2)
        << "\nTensor(" << X(1).name() + ") must be a matrix"
        << "(or rank > 2, representing batches of matrices).";

    M1_ = X(0).dim(-2), N1_ = X(0).dim(-1);
    M2_ = X(1).dim(-2), N2_ = X(1).dim(-1);

    M_  = transA_ ? N1_ : M1_;
    N_  = transB_ ? M2_ : N2_;
    K1_ = transA_ ? M1_ : N1_;
    K2_ = transB_ ? N2_ : M2_;

    A_stride_ = M1_ * N1_;
    B_stride_ = M2_ * N2_;
    Y_stride_ = M_ * N_;

    batch_size_ = X(0).count() / A_stride_;

    CHECK_EQ(K1_, K2_) << "\nTensor(" << X(0).name()
                       << "): " << X(0).DimString()
                       << " can not mul with Tensor"
                       << "(" << X(1).name() << "): "
                       << X(1).DimString();

    CHECK_EQ(batch_size_, X(1).count() / B_stride_)
        << "\nTensor(" << X(0).name()
        << "): " << X(0).DimString()
        << " can not mul with Tensor"
        << "(" << X(1).name() << "): "
        << X(1).DimString();

    Y(0)->ReshapeLike(X(0));
    Y(1)->ReshapeLike(X(1));

    if (XIsType(X(0), float16)) {
        RunImpl<float16>();
    } else if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else if (XIsType(X(0), double)) {
        RunImpl<double>();
    } else {
        LOG(FATAL) << DTypeString(X(0),
            { "float16", "float32", "float64" }
        );
    }
}

DEPLOY_CPU(Matmul);
#ifdef WITH_CUDA
DEPLOY_CUDA(Matmul);
#endif

DEPLOY_CPU(MatmulGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(MatmulGradient);
#endif

OPERATOR_SCHEMA(Matmul)
     /* A, B */
    .NumInputs(2)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(MatmulGradient)
     /* A, B, dY */
    .NumInputs(3)
     /* dA, dB */
    .NumOutputs(2);

REGISTER_GRADIENT(Matmul, SimpleGradientMaker);

}  // namespace dragon