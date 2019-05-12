#include "core/workspace.h"
#include "utils/filler.h"
#include "operators/arithmetic/fully_connected_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGS(ref) \
    axis_ = OpArg<int64_t>("axis", 1); \
    axis_ = axis_ < 0 ? axis_ + ref.ndim() : axis_; \
    CHECK(axis_ >= 0 && axis_ < ref.ndim()) \
       << "\nExcepted axis in [-" << ref.ndim() \
       << ", " << ref.ndim() << "), got " \
       << OpArg<int64_t>("axis", 1) << ".";

template <class Context> template <typename T>
void FullyConnectedOp<Context>::TransRunImpl() {
    vec64_t w_shape({ N_, K_ }), b_shape({ N_ });
    TENSOR_FILL(X(1), w_shape);
    if (XSize() > 2) TENSOR_FILL(X(2), b_shape);

    CHECK(X(1).ndim() == 2 && X(1).dim(1) == K_)
        << "\nWeights dimensions should be [N, K].\n"
        << "Got X as (" << M_ << ", " << K_ << "), "
        << "and W as " << X(1).DimString();

    auto* x = X(0).template data<T, Context>();
    auto* w = X(1).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    math::Gemm(
        CblasNoTrans,
        CblasTrans,
        M_, N_, K_,
        1.f, x, w,
        0.f, y, ctx()
    );

    if (XSize() > 2) {
        DECLARE_MULTIPLIER(multiplier, M_);
        auto* b = X(2).template data<T, Context>();
        math::Gemm(
            CblasNoTrans,
            CblasNoTrans,
            M_, N_, 1,
            1.f, multiplier, b,
            1.f, y, ctx()
        );
    }
}

template <class Context> template <typename T>
void FullyConnectedOp<Context>::NoTransRunImpl() {
    vec64_t w_shape({ K_, N_ }), b_shape({ N_ });
    TENSOR_FILL(X(1), w_shape);
    if (XSize() > 2) TENSOR_FILL(X(2), b_shape);

    CHECK(X(1).ndim() == 2 && X(1).dim(0) == K_)
        << "\nWeights dimensions should be [K, N].\n"
        << "Got X as (" << M_ << ", " << K_ << "), "
        << "and W as " << X(1).DimString();

    auto* x = X(0).template data<T, Context>();
    auto* w = X(1).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    math::Gemm(
        CblasNoTrans,
        CblasNoTrans,
        M_, N_, K_,
        1.f, x, y,
        0.f, y, ctx()
    );

    if (XSize() > 2) {
        DECLARE_MULTIPLIER(multiplier, M_);
        auto* b = X(2).template data<T, Context>();
        math::Gemm(
            CblasNoTrans,
            CblasNoTrans,
            M_, N_, 1,
            1.f, multiplier, b,
            1.f, y, ctx()
        );
    }
}

template <class Context>
void FullyConnectedOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));
    M_ = X(0).count(0, axis_), K_ = X(0).count(axis_);

    if (N_ <= 0) {
        // Infer the "N" from the weights shape
        N_ = X(1).count() / K_;
        CHECK_GT(N_, 0)
            << "\nFailed to infer the N from "
            << "the weights: " << X(1).DimString();
    }

    vec64_t out_shape(axis_ + 1);
    for (int i = 0; i < axis_ + 1; i++) {
        out_shape[i] = i < axis_ ? X(0).dim(i) : N_;
    }
    Y(0)->Reshape(out_shape);

    if (XIsType(X(0), float16)) {
        if (transW_) {
            TransRunImpl<float16>();
        } else {
            NoTransRunImpl<float16>();
        }
    } else if (XIsType(X(0), float)) {
        if (transW_) {
            TransRunImpl<float>();
        } else {
            NoTransRunImpl<float>();
        }
    } else if (XIsType(X(0), double)) {
        if (transW_) {
            TransRunImpl<double>();
        } else {
            NoTransRunImpl<double>();
        }
    } else {
        LOG(FATAL) << DTypeString(X(0),
            { "float16",  "float32", "float64" }
        );
    }
}

template <class Context> template <typename T>
void FullyConnectedGradientOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* w = X(1).template data<T, Context>();
    auto* dy = X(-1).template data<T, Context>();

    if (Y(1)->name() != "NULL") {
        Y(1)->ReshapeLike(X(1));
        auto* dw = Y(1)->template
            mutable_data<T, Context>();
        if (transW_) {
            math::Gemm(
                CblasTrans,
                CblasNoTrans,
                N_, K_, M_,
                1.f, dy, x,
                0.f, dw, ctx()
            );
        } else {
            math::Gemm(
                CblasTrans,
                CblasNoTrans,
                K_, N_, M_,
                1.f, x, dy,
                0.f, dw, ctx()
            );
        }
    }

    if (Y(2)->name() != "NULL") {
        DECLARE_MULTIPLIER(multiplier, M_);
        Y(2)->Reshape({ N_ });
        auto* db = Y(2)->template
            mutable_data<T, Context>();
        math::Gemv(
            CblasTrans,
            M_, N_,
            1.f, dy, multiplier,
            0.f, db, ctx()
        );
    }

    if (Y(0)->name() != "NULL") {
        Y(0)->ReshapeLike(X(0));
        auto* dx = Y(0)->template
            mutable_data<T, Context>();
        if (transW_) {
            math::Gemm(
                CblasNoTrans,
                CblasNoTrans,
                M_, K_, N_,
                1.f, dy, w,
                0.f, dx, ctx()
            );
        } else {
            math::Gemm(
                CblasNoTrans,
                CblasTrans,
                M_, K_, N_,
                1.f, dy, w,
                0.f, dx, ctx()
            );
        }
    }
}

template <class Context>
void FullyConnectedGradientOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));
    M_ = X(0).count(0, axis_);
    K_ = X(0).count(axis_);

    if (N_ <= 0) {
        // Infer the "N" from the weights shape
        N_ = X(1).count() / K_;
        CHECK_GT(N_, 0)
            << "\nFailed to infer the N from "
            << "the weights shape: "
            << X(1).DimString();
    }

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

DEPLOY_CPU(FullyConnected);
#ifdef WITH_CUDA
DEPLOY_CUDA(FullyConnected);
#endif

DEPLOY_CPU(FullyConnectedGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(FullyConnectedGradient);
#endif

OPERATOR_SCHEMA(FullyConnected)
     /* X, W, B */
    .NumInputs(2, 3)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(FullyConnectedGradient)
     /* X, W, dY */
    .NumInputs(3)
     /* dX, dW, dB */
    .NumOutputs(3);

namespace {

class GradientMaker : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GradientMaker);
    vector<OperatorDef> MakeDef() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), GO(0) }),
            vector<string>({ GI(0), GI(1), GI(2) })
        );
    }
};

}  // namespace

REGISTER_GRADIENT(FullyConnected, GradientMaker);

#undef DETERMINE_RUNTIME_ARGS

}  // namespace dragon