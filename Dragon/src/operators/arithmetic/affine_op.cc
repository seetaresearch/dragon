#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/op_kernel.h"
#include "operators/arithmetic/affine_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGS(X) \
    axis_ = OpArg<int64_t>("axis", 1); \
    num_axes_ = OpArg<int64_t>("num_axes", 1); \
    if (axis_ < 0) axis_ += X.ndim(); \
    if (num_axes_ < 0) num_axes_ = X.ndim() - axis_; \
    else if (num_axes_ == 0) num_axes_ = 1; \
    CHECK(axis_ >= 0 && axis_ + num_axes_ <= X.ndim())

template <class Context> template <typename T>
void AffineOp<Context>::RunImpl() {
    const auto& dim_start = X(0).dims().begin() + axis_;
    const auto& dim_end = dim_start + num_axes_;
    vec64_t param_dims(dim_start, dim_end);
    scale_dim_ = X(1).count();
    outer_dim_ = X(0).count(0, axis_);
    inner_dim_ = X(0).count(axis_ + num_axes_);

    TENSOR_FILL(X(1), param_dims);
    if (XSize() > 2) TENSOR_FILL(X(2), param_dims);

    auto* x = X(0).template data<T, Context>();
    auto* alpha = X(1).template data<T, Context>();
    auto* beta = XSize() <= 2 ? nullptr :
                 X(2).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    kernel::Affine(
        outer_dim_,
        scale_dim_,
        inner_dim_,
        x, alpha, beta,
        y, ctx()
    );
}

template <class Context>
void AffineOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));

    Y(0)->ReshapeLike(X(0));

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

template <class Context> template <typename T>
void AffineGradientOp<Context>::RunImpl() {
    auto* alpha = X(1).template data<T, Context>();
    auto* dy = X(-1).template mutable_data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

    // dA = X * dY
    if (Y(1)->name() != "NULL") {
        Y(1)->ReshapeLike(X(1));
        auto* x = X(0).template data<T, Context>();
        auto* dalpha = Y(1)->template mutable_data<T, Context>();
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
            Reduce(dx, dalpha);
        }
    }

    // dB = dY
    if (Y(2)->name() != "NULL") {
        Y(2)->ReshapeLike(X(1));
        auto* dbeta = Y(2)->template
            mutable_data<T, Context>();
        // Eltwise
        if (X(-1).count() == X(1).count()) {
            math::Copy(X(1).count(), dy, dbeta, ctx());
        } else {
            Reduce(dy, dbeta);
        }
    }

    // dX = alpha * dY
    if (Y(0)->name() != "NULL") {
        kernel::AffineGrad(
            outer_dim_,
            scale_dim_,
            inner_dim_,
            dy, alpha,
            dx, ctx()
        );
    }
}

template <class Context> template <typename T>
void AffineGradientOp<Context>::Reduce(
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
void AffineGradientOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));
    scale_dim_ = X(1).count();
    outer_dim_ = X(-1).count(0, axis_);
    inner_dim_ = X(-1).count(axis_ + num_axes_);
    dim_ = scale_dim_ * inner_dim_;
    reduce_dim_ = std::max(outer_dim_, inner_dim_);

    Y(0)->ReshapeLike(X(-1));

    if (XIsType(X(-1), float)) {
        RunImpl<float>();
    } else if (XIsType(X(-1), float16)) {
        RunImpl<float16>();
    } else {
        LOG(FATAL) << DTypeString(X(-1),
            { "float32", "float16" }
        );
    }
}

DEPLOY_CPU(Affine);
#ifdef WITH_CUDA
DEPLOY_CUDA(Affine);
#endif

DEPLOY_CPU(AffineGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(AffineGradient);
#endif

OPERATOR_SCHEMA(Affine)
     /* X, A, B */
    .NumInputs(2, 3)
     /* Y */
    .NumOutputs(1)
     /* X => Y */
    .Inplace({ { 0, 0 } });

OPERATOR_SCHEMA(AffineGradient)
     /* X, A, dY */
    .NumInputs(3)
     /* dX, dA, dB */
    .NumOutputs(3)
     /* dY => dX */
    .Inplace({ { 2, 0 } });

namespace {

class GradientMaker final : public GradientMakerBase {
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

REGISTER_GRADIENT(Affine, GradientMaker);

#undef DETERMINE_RUNTIME_ARGS

}  // namespace dragon