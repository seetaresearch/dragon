#include "utils/math_functions.h"
#include "operators/arithmetic/dot_op.h"

namespace dragon {

template <class Context> template <typename T>
void DotOp<Context>::DotRunImpl() {
    CHECK_EQ(X(0).dim(0), X(1).dim(0))
        << "\nTensor(" << X(0).name()
        << "): " << X(0).DimString()
        << " can not Dot with Tensor"
        << "(" << X(1).name() << "): "
        << X(1).DimString();

    Y(0)->Reshape({});

    auto* a = X(0).template data<T, Context>();
    auto* b = X(1).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    T yHost;
    math::Dot(X(0).count(), a, b, &yHost, ctx());
    ctx()->template Copy<T, Context, CPUContext>(1, y, &yHost);
}

template <class Context> template <typename T>
void DotOp<Context>::GemmRunImpl() {
    M1_ = X(0).count() / X(0).dim(-1);
    N1_ = X(0).dim(-1);
    M2_ = X(1).dim(0);
    N2_ = X(1).dim(1);
    M_ = transA_ ? N1_ : M1_;
    N_ = transB_ ? M2_ : N2_;
    K1_ = transA_ ? M1_ : N1_;
    K2_ = transB_ ? N2_ : M2_;

    CHECK_EQ(K1_, K2_) << "\nTensor(" << X(0).name()
                       << "): " << X(0).DimString()
                       << " can not Dot with Tensor"
                       << "(" << X(1).name() << "): "
                       << X(1).DimString();

    auto dims = X(0).dims(); dims.back() = N_;
    Y(0)->Reshape(transA_ ? vec64_t({ M_, N_ }) : dims);

    auto* a = X(0).template data<T, Context>();
    auto* b = X(1).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    math::Gemm(
        transA_ ? CblasTrans : CblasNoTrans,
        transB_ ? CblasTrans : CblasNoTrans,
        M_, N_, K1_,
        1.f, a, b,
        0.f, y, ctx()
    );
}

template <class Context> template <typename T>
void DotOp<Context>::GemvRunImpl() {
    M1_ = X(0).count() / X(0).dim(-1);
    N1_ = X(0).dim(-1);
    K1_ = transA_ ? M1_ : N1_;
    K2_ = X(1).dim(0);

    CHECK_EQ(K1_, K2_) << "\nTensor(" << X(0).name()
                       << "): " << X(0).DimString()
                       << " can not Dot with Tensor"
                       << "(" << X(1).name() << "): "
                       << X(1).DimString();

    auto dims = X(0).dims(); dims.pop_back();
    Y(0)->Reshape(transA_ ? vec64_t({ N1_ }) : dims);

    auto* a = X(0).template data<T, Context>();
    auto* b = X(1).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    math::Gemv(
        transA_ ? CblasTrans : CblasNoTrans,
        M1_, N1_,
        1.f, a, b,
        0.f, y, ctx()
    );
}

template <class Context>
void DotOp<Context>::RunOnDevice() {
    if (X(0).ndim() == 1 && X(1).ndim() == 1) {
        if (XIsType(X(0), float16)) {
            DotRunImpl<float16>();
        } else if (XIsType(X(0), float)) {
            DotRunImpl<float>();
        } else if (XIsType(X(0), double)) {
            DotRunImpl<double>();
        } else {
            LOG(FATAL) << DTypeString(X(0),
                { "float16", "float32", "float64" }
            );
        }
    } else if (X(0).ndim() >= 2 && X(1).ndim() == 2) {
        if (XIsType(X(0), float16)) {
            GemmRunImpl<float16>();
        } else if (XIsType(X(0), float)) {
            GemmRunImpl<float>();
        } else if (XIsType(X(0), double)) {
            GemmRunImpl<double>();
        } else {
            LOG(FATAL) << DTypeString(X(0),
                { "float16", "float32", "float64" }
            );
        }
    } else if (X(0).ndim() >= 2 && X(1).ndim() == 1) {
        if (XIsType(X(0), float16)) {
            GemvRunImpl<float16>();
        } else if (XIsType(X(0), float)) {
            GemvRunImpl<float>();
        } else if (XIsType(X(0), double)) {
            GemvRunImpl<double>();
        } else {
            LOG(FATAL) << DTypeString(X(0),
                {"float16", "float32", "float64" }
            );
        }
    } else {
        LOG(FATAL) << "\nTensor(" << X(0).name()
                   << "): " << X(0).DimString()
                   << " can not Dot with Tensor"
                   << "(" << X(1).name() << "): "
                   << X(1).DimString();
    }
}

template <class Context> template <typename T>
void DotGradientOp<Context>::DotRunImpl() {
    CHECK_EQ(X(0).count(), X(1).count())
        << "\nTensor(" << X(0).name()
        << "): " << X(0).DimString()
        << " can not Dot with Tensor"
        << "(" << X(1).name() << "): "
        << X(1).DimString();

    auto* a = X(0).template data<T, Context>();
    auto* b = X(1).template data<T, Context>();
    auto* dy = X(-1).template data<T, CPUContext>();

    if (Y(0)->name() != "NULL") {
        auto* da = Y(0)->template
            mutable_data<T, Context>();
        math::Scale(
            Y(0)->count(),
            cast::to<float>(dy[0]),
            b, da, ctx()
        );
    }

    if (Y(1)->name() != "NULL") {
        auto* db = Y(1)->template
            mutable_data<T, Context>();
        math::Scale(
            Y(0)->count(),
            cast::to<float>(dy[0]),
            a, db, ctx()
        );
    }
}

template <class Context> template <typename T>
void DotGradientOp<Context>::GemmRunImpl() {
    M1_ = X(0).count() / X(0).dim(-1);
    N1_ = X(0).dim(-1);
    M2_ = X(1).dim(0);
    N2_ = X(1).dim(1);
    M_ = transA_ ? N1_ : M1_;
    N_ = transB_ ? M2_ : N2_;
    K1_ = transA_ ? M1_ : N1_;
    K2_ = transB_ ? N2_ : M2_;

    CHECK_EQ(K1_, K2_) << "\nTensor(" << X(0).name()
                       << "): " << X(0).DimString()
                       << " can not Dot with Tensor"
                       << "(" << X(1).name() << "): "
                       << X(1).DimString();

    auto* a = X(0).template data<T, Context>();
    auto* b = X(1).template data<T, Context>();
    auto* dy = X(-1).template data<T, Context>();

    if (Y(0)->name() != "NULL") {
        auto* da = Y(0)->template
            mutable_data<T, Context>();
        if (transA_) {
            math::Gemm(
                transB_ ? CblasTrans : CblasNoTrans,
                CblasTrans,
                K1_, M_, N_,
                1.f, b, dy,
                0.f, da, ctx()
            );
        } else {
            math::Gemm(
                CblasNoTrans,
                transB_ ? CblasNoTrans : CblasTrans,
                M_, K1_, N_,
                1.f, dy, b,
                0.f, da, ctx()
            );
        }
    }

    if (Y(1)->name() != "NULL") {
        auto* db = Y(1)->template
            mutable_data<T, Context>();
        if (transB_) {
           math::Gemm(
               CblasTrans,
               transA_ ? CblasTrans : CblasNoTrans,
               N_, K1_, M_,
               1.f, dy, a,
               0.f, db, ctx()
           );
        } else {
            math::Gemm(
                transA_ ? CblasNoTrans : CblasTrans,
                CblasNoTrans,
                K1_, N_, M_,
                1.f, a, dy,
                0.f, db, ctx()
            );
        }
    }
}

template <class Context> template <typename T>
void DotGradientOp<Context>::GemvRunImpl() {
    M1_ = X(0).count() / X(0).dim(-1);
    N1_ = X(0).dim(-1);
    K1_ = transA_ ? M1_ : N1_;
    K2_ = X(1).dim(0);
    M_ = transA_ ? N1_ : M1_;
    N_ = K2_;  // Keep M, Remove N

    CHECK_EQ(K1_, K2_) << "\nTensor(" << X(0).name()
                       << "): " << X(0).DimString()
                       << " can not Dot with Tensor"
                       << "(" << X(1).name() << "): "
                       << X(1).DimString();

    auto* a = X(0).template data<T, Context>();
    auto* b = X(1).template data<T, Context>();
    auto* dy = X(-1).template data<T, Context>();

    if (Y(0)->name() != "NULL") {
        auto* da = Y(0)->template
            mutable_data<T, Context>();
        math::Gemm(
            CblasNoTrans,
            CblasNoTrans,
            M_, N_, 1,
            1.f, dy, b,
            0.f, da, ctx()
        );
    }

    if (Y(1)->name() != "NULL") {
        auto* db = Y(1)->template
            mutable_data<T, Context>();
        math::Gemv(
            transA_ ? CblasNoTrans : CblasTrans,
            M1_, N1_,
            1.f, a, dy,
            0.f, db, ctx()
        );
    }
}

template <class Context>
void DotGradientOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));
    Y(1)->ReshapeLike(X(1));

    if (X(0).ndim() == 1 && X(1).ndim() == 1) {
        if (XIsType(X(0), float16)) {
            DotRunImpl<float16>();
        } else if (XIsType(X(0), float)) {
            DotRunImpl<float>();
        } else if (XIsType(X(0), double)) {
            DotRunImpl<double>();
        } else {
            LOG(FATAL) << DTypeString(X(0),
                { "float16", "float32", "float64" }
            );
        }
    } else if (X(0).ndim() >= 2 && X(1).ndim() == 2) {
        if (XIsType(X(0), float16)) {
            GemmRunImpl<float16>();
        } else if (XIsType(X(0), float)) {
            GemmRunImpl<float>();
        } else if (XIsType(X(0), double)) {
            GemmRunImpl<double>();
        } else {
            LOG(FATAL) << DTypeString(X(0),
                { "float16", "float32", "float64" }
            );
        }
    } else if (X(0).ndim() >= 2 && X(1).ndim() == 1) {
        if (XIsType(X(0), float16)) {
            GemvRunImpl<float16>();
        } else if (XIsType(X(0), float)) {
            GemvRunImpl<float>();
        } else if (XIsType(X(0), double)) {
            GemvRunImpl<double>();
        } else {
            LOG(FATAL) << DTypeString(X(0),
                { "float16", "float32", "float64" }
            );
        }
    } else {
        LOG(FATAL) << "\nTensor(" << X(0).name()
                   << "): " << X(0).DimString()
                   << " can not Dot with Tensor"
                   << "(" << X(1).name() << "): "
                   << X(1).DimString();
    }
}

DEPLOY_CPU(Dot);
#ifdef WITH_CUDA
DEPLOY_CUDA(Dot);
#endif

DEPLOY_CPU(DotGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(DotGradient);
#endif

OPERATOR_SCHEMA(Dot)
     /* A, B */
    .NumInputs(2)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(DotGradient)
     /* A, B, dY */
    .NumInputs(3)
     /* dA, dB */
    .NumOutputs(2);

REGISTER_GRADIENT(Dot, SimpleGradientMaker);

}  // namespace dragon