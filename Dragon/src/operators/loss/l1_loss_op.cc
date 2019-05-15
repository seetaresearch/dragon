#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/loss/l1_loss_op.h"

namespace dragon {

template <class Context> template <typename T>
void L1LossOp<Context>::RunImpl() {
    auto nelements = X(0).count();
    auto* x = X(0).template data<T, Context>();

    auto* diff = ws()
        ->CreateTensor(unique_name("diff"))
        ->ReshapeLike(X(0))
        ->template mutable_data<T, Context>();

    auto* y = Y(0)->template mutable_data<T, Context>();

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

    double normalizer = 1. / scale_;
    if (reduction_ == "BATCH_SIZE") {
        normalizer *= X(0).dim(0);
    } else if (reduction_ == "MEAN") {
        normalizer *= nelements;
    }

    T loss = math::ASum(nelements, diff, ctx());
    math::Set(1, T(loss / normalizer), y, ctx());
}

template <class Context>
void L1LossOp<Context>::RunOnDevice() {
    ctx()->set_stream_id(0);  // Enforce DefaultStream

    for (int i = 1; i < XSize(); i++) {
        CHECK_EQ(X(0).count(), X(i).count())
            << "\nTensor(" << X(i).name() << ") takes the "
            << "dimensions of " << X(i).DimString() << ", "
            << "while " << X(0).DimString() << " is required.";
    }

    Y(0)->Reshape({});

    DispatchHelper<TensorTypes
        <float>>::Call(this, X(0));
}

template <class Context> template <typename T>
void L1LossGradientOp<Context>::RunImpl() {
    auto nelements = X(0).count();
    auto* dy = X(-1).template data<T, Context>();

    T dyHost; ctx()->template Copy
        <T, CPUContext, Context>(
            1, &dyHost, dy);
    ctx()->FinishDeviceCompution();

    auto* diff = ws()
        ->GetTensor(unique_name("diff"))
        ->template mutable_data<T, Context>();

    kernel::AbsGrad(nelements, diff, diff, ctx());

    if (reduction_ == "BATCH_SIZE") {
        dyHost /= (X(0).dim(0) / scale_);
    } else if (reduction_ == "MEAN") {
        dyHost /= (X(0).count() / scale_);
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
void L1LossGradientOp<Context>::RunOnDevice() {
    DispatchHelper<TensorTypes
        <float>>::Call(this, X(0));
}

DEPLOY_CPU(L1Loss);
#ifdef WITH_CUDA
DEPLOY_CUDA(L1Loss);
#endif

DEPLOY_CPU(L1LossGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(L1LossGradient);
#endif

OPERATOR_SCHEMA(L1Loss)
     /* X, T, W */
    .NumInputs(1, 3)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(L1LossGradient)
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

REGISTER_GRADIENT(L1Loss, GradientMaker);

}  // namespace dragon