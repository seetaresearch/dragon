#include "core/workspace.h"
#include "utils/math_functions.h"
#include "operators/arithmetic/pow_op.h"

namespace dragon {

template <class Context> template <typename T>
void PowOp<Context>::RunImpl() {
    auto nelements = X(0).count();
    auto* y = Y(0)->template mutable_data<T, Context>();

    if (power_scale_ == 0.f) {
        // Y is a constant Vector
        float c = power_ == 0.f ? 1.f : pow(shift_, power_);
        math::Set(
            nelements,
            cast::to<T>(c),
            y, ctx()
        ); return;
    }

    // Compute Y = (Ax + b) ** p
    auto* x = X(0).template data<T, Context>();
    math::Scale(nelements, scale_, x, y, ctx());
    math::AddScalar(nelements, shift_, y, ctx());
    if (power_ != 1.f) {
        math::Pow(nelements, power_, y, y, ctx());
    }
}

template <class Context>
void PowOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

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
void PowGradientOp<Context>::RunImpl() {
    auto nelements = X(0).count();
    auto* dy = X(-1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

    if (power_scale_ == 0.f || power_ == 1.f) {
        auto c = cast::to<T>(power_scale_);
        math::Set(nelements, c, dx, ctx());
    } else {
        auto* x = X(0).template data<T, Context>();
        if (power_ == 2.f) {
            math::Axpby(
                nelements,
                power_scale_ * scale_, x,
                0.f, dx, ctx()
            );
            if (shift_ != 0.f) {
                math::AddScalar(
                    nelements,
                    power_scale_ * shift_,
                    dx, ctx()
                );
            }
        } else if (shift_ == 0.f) {
            auto* y = X(1).template data<T, Context>();
            math::Div(nelements, y, x, dx, ctx());
            math::Scale(nelements, power_, dx, dx, ctx());
        } else {
            auto* y = X(1).template data<T, Context>();
            math::Scale(nelements, scale_, x, dx, ctx());
            math::AddScalar(nelements, shift_, dx, ctx());
            math::Div(nelements, y, dx, dx, ctx());
            math::Scale(nelements, power_scale_, dx, dx, ctx());
        }
    }
    math::Mul(nelements, dy, dx, dx, ctx());
}

template <class Context>
void PowGradientOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

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

DEPLOY_CPU(Pow);
#ifdef WITH_CUDA
DEPLOY_CUDA(Pow);
#endif

DEPLOY_CPU(PowGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(PowGradient);
#endif

OPERATOR_SCHEMA(Pow)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(PowGradient)
     /* X, Y, dY */
    .NumInputs(3)
     /* dX */
    .NumOutputs(1);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GradientMaker);
    vector<OperatorDef> MakeDef() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), O(0), GO(0)}),
            vector<string>({ GI(0) })
        );
    }
};

}  // namespace

REGISTER_GRADIENT(Pow, GradientMaker);

}  // namespace dragon