#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/loss/smooth_l1_loss_op.h"

namespace dragon {

template <class Context> template <typename T>
void SmoothL1LossOp<Context>::RunImpl() {
    auto nelements = X(0).count();
    auto* x = X(0).template data<T, Context>();
    auto* target = X(1).template data<T, Context>();

    auto* diff = ws()
        ->CreateTensor(unique_name("diff"))
        ->ReshapeLike(X(0))
        ->template mutable_data<T, Context>();

    auto* err = ws()
        ->CreateTensor("/share/smoothl1/err")
        ->ReshapeLike(X(0))
        ->template mutable_data<T, Context>();

    math::Sub(nelements, x, target, diff, ctx());

    if (XSize() > 2) {
        CHECK_EQ(X(2).count(), nelements);
        auto* iw = X(2).template data<T, Context>();
        math::Mul(nelements, iw, diff, diff, ctx());
    }

    kernel::SmoothL1(nelements, beta_, diff, err, ctx());

    if (XSize() > 3) {
        CHECK_EQ(X(3).count(), nelements);
        auto* ow = X(3).template data<T, Context>();
        math::Mul(nelements, ow, err, err, ctx());
    }

    double normalizer = 1.;
    if (reduction_ == "BATCH_SIZE") {
        normalizer = X(0).dim(0);
    } else if (reduction_ == "MEAN") {
        normalizer = X(0).count();
    }

    auto* y = Y(0)->template mutable_data<T, Context>();
    math::Sum(nelements, 1. / normalizer, err, y, ctx());
}

template <class Context>
void SmoothL1LossOp<Context>::RunOnDevice() {
    CHECK(X(0).count() == X(1).count());

    Y(0)->Reshape({});

    DispatchHelper<TensorTypes
        <float>>::Call(this, X(0));
}

template <class Context> template <typename T>
void SmoothL1LossGradientOp<Context>::RunImpl() {
    auto nelements = X(0).count();
    auto* dy = X(-1).template data<T, Context>();

    T dyHost; ctx()->template Copy
        <T, CPUContext, Context>(
            1, &dyHost, dy);
    ctx()->FinishDeviceCompution();

    auto* diff = ws()
        ->GetTensor(unique_name("diff"))
        ->template mutable_data<T, Context>();

    kernel::SmoothL1Grad(
        nelements,
        beta_, diff,
        diff, ctx()
    );

    if (reduction_ == "BATCH_SIZE") {
        dyHost /= X(0).dim(0);
    } else if (reduction_ == "MEAN") {
        dyHost /= X(0).count();
    }

    for (int i = 0; i < YSize(); i++) {
        if (Y(i)->name() == "NULL") continue;
        Y(i)->ReshapeLike(X(i));
        auto* dx = Y(i)->template mutable_data<T, Context>();
        math::Scale(
            nelements,
            dyHost * (i == 0 ? 1.f : -1.f),
            diff, dx, ctx()
        );
        if (XSize() > 3) {
            auto* iw = X(2).template data<T, Context>();
            math::Mul(nelements, iw, dx, dx, ctx());
        }
        if (XSize() > 4) {
            auto* ow = X(3).template data<T, Context>();
            math::Mul(nelements, ow, dx, dx, ctx());
        }
    }
}

template <class Context>
void SmoothL1LossGradientOp<Context>::RunOnDevice() {
    DispatchHelper<TensorTypes
        <float>>::Call(this, X(0));
}

DEPLOY_CPU(SmoothL1Loss);
#ifdef WITH_CUDA
DEPLOY_CUDA(SmoothL1Loss);
#endif

DEPLOY_CPU(SmoothL1LossGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SmoothL1LossGradient);
#endif

OPERATOR_SCHEMA(SmoothL1Loss)
     /* X, T, IW, OW */
    .NumInputs(2, 4)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(SmoothL1LossGradient)
     /* X, T, IW, OW, dY */
    .NumInputs(3, 5)
     /* dX, dT */
    .NumOutputs(2);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GradientMaker);
    vector<OperatorDef> MakeDef() override {
        vector<string> inputs;
        for (auto e : def.input()) inputs.push_back(e);
        inputs.push_back(GO(0));
        return SingleDef(def.type() + "Gradient", "",
            inputs, vector<string>({ GI(0), GI(1) })
        );
    }
};

}  // namespace

REGISTER_GRADIENT(SmoothL1Loss, GradientMaker);

}  // namespace dragon