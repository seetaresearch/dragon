#include "core/workspace.h"
#include "utils/math_functions.h"
#include "operators/loss/l2_loss_op.h"

namespace dragon {

template <class Context> template <typename T>
void L2LossOp<Context>::RunImpl() {
    auto nelements = X(0).count();
    auto* x = X(0).template data<T, Context>();

    auto* diff = ws()
        ->CreateTensor(unique_name("diff"))
        ->ReshapeLike(X(0))
        ->template mutable_data<T, Context>();

    auto* y = Y(0)
        ->Reshape({})
        ->template mutable_data<float, Context>();

    if (XSize() > 1) {
        auto* target = X(1).template data<T, Context>();
        math::Sub(nelements, x, target, diff, ctx());
    } else {
        math::Copy(nelements, x, diff, ctx());
    }

    if (XSize() > 2) {
        auto* mask = X(2).template data<T, Context>();
        math::Mul(nelements, mask, diff, diff, ctx());
    }

    double normalizer = 2. / scale_;
    if (reduction_ == "BATCH_SIZE") {
        normalizer *= X(0).dim(0);
    } else if (reduction_ == "MEAN") {
        normalizer *= X(0).count();
    }

    T loss;
    math::Dot(nelements, diff, diff, &loss, ctx());
    math::Set(
        1, cast::to<float>(
                cast::to<float>(loss)
                    / normalizer),
        y, ctx()
    );
}

template <class Context>
void L2LossOp<Context>::RunOnDevice() {
    for (int i = 1; i < XSize(); i++) {
        CHECK_EQ(X(0).count(), X(i).count())
            << "\nTensor(" << X(i).name() << ") takes the "
            << "dimensions of " << X(i).DimString() << ", "
            << "while " << X(0).DimString() << " is required.";
    }

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
void L2LossGradientOp<Context>::RunImpl() {
    auto nelements = X(0).count();
    auto* dy = X(-1).template data<float, Context>();

    auto* diff = ws()
        ->GetTensor(unique_name("diff"))
        ->template data<T, Context>();

    float dyHost; ctx()->template Copy
        <float, CPUContext, Context>(
            1, &dyHost, dy);
    ctx()->FinishDeviceCompution();

    if (reduction_ == "BATCH_SIZE") {
        dyHost /= (X(0).dim(0) / scale_);
    } else if (reduction_ == "MEAN") {
        dyHost /= (nelements / scale_);
    } else {
        dyHost *= scale_;
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
        if (X(2).name() != "NULL") {
            auto* mask = X(2).template data<T, Context>();
            math::Mul(nelements, mask, dx, dx, ctx());
        }
    }
}

template <class Context>
void L2LossGradientOp<Context>::RunOnDevice() {
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

DEPLOY_CPU(L2Loss);
#ifdef WITH_CUDA
DEPLOY_CUDA(L2Loss);
#endif

DEPLOY_CPU(L2LossGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(L2LossGradient);
#endif

OPERATOR_SCHEMA(L2Loss)
     /* X, T, W */
    .NumInputs(1, 3)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(L2LossGradient)
     /* X, T, W, dY */
    .NumInputs(4)
     /* dX, dT */
    .NumOutputs(2);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GradientMaker);
    vector<OperatorDef> MakeDef() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), I(2), GO(0) }),
            vector<string>({ GI(0), GI(1) })
        );
    }
};

}  // namespace

REGISTER_GRADIENT(L2Loss, GradientMaker);

}  // namespace dragon