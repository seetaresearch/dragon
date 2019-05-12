#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/loss/sigmoid_ce_loss_op.h"

namespace dragon {

template <class Context> template <typename T>
void SigmoidCrossEntropyOp<Context>::RunImpl() {
    auto* logit  = X(0).template data<T, Context>();
    auto* target = X(1).template data<T, Context>();
    auto* loss = loss_.template mutable_data<T, Context>();
    auto* flag = flag_.template mutable_data<int, Context>();

    kernel::SigmoidCrossEntropy(
        X(0).count(),
        logit, target,
        loss, flag, ctx()
    );

    if (reduction_ == "NONE") {
        Y(0)->ReshapeLike(loss_)
            ->CopyFrom(loss_, ctx());
        return;
    }

    double normalizer = 1.;
    if (reduction_ == "VALID") {
        normalizer = std::max(
            math::Sum(
                flag_.count(),
                1.f, flag, ctx()
            ), 1);
    } else if (reduction_ == "BATCH_SIZE") {
        normalizer = X(0).dim(0);
    } else if (reduction_ == "MEAN") {
        normalizer = X(0).count();
    }

    Y(0)->Reshape({});
    auto* y = Y(0)->template mutable_data<T, Context>();
    math::Sum(loss_.count(), 1. / normalizer, loss, y, ctx());
}

template <class Context>
void SigmoidCrossEntropyOp<Context>::RunOnDevice() {
    CHECK_EQ(X(0).count(), X(1).count())
        << "\nNum of preds must match the num of labels.";

    loss_.ReshapeLike(X(0));
    flag_.ReshapeLike(X(0));

    if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else {
        LOG(FATAL) << DTypeString(
            X(0), { "float32" }
        );
    }
}

template <class Context> template <typename T>
void SigmoidCrossEntropyGradientOp<Context>::RunImpl() {
    auto* logit  = X(0).template data<T, Context>();
    auto* target = X(1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();
    auto* flag = flag_.template mutable_data<int, Context>();

    kernel::SigmoidCrossEntropyGrad(
        X(0).count(),
        logit, target,
        dx, flag, ctx()
    );

    if (reduction_ == "NONE") {
        auto* dy = X(-1).template data<T, Context>();
        math::Mul(
            Y(0)->count(),
            dy, dx,
            dx, ctx()
        );
        return;
    }

    double normalizer = 1.;
    if (reduction_ == "VALID") {
        normalizer = std::max(
            math::Sum(
                flag_.count(),
                1.f, flag, ctx()
            ), 1);
    } else if (reduction_ == "BATCH_SIZE") {
        normalizer = X(0).dim(0);
    } else if (reduction_ == "MEAN") {
        normalizer = X(0).count();
    }

    auto* dy = X(-1).template data<T, Context>();
    T dyHost; ctx()->template Copy
        <T, CPUContext, Context>(
            1, &dyHost, dy);
    ctx()->FinishDeviceCompution();

    math::Scale(
        Y(0)->count(),
        dyHost / normalizer,
        dx, dx, ctx()
    );
}

template <class Context>
void SigmoidCrossEntropyGradientOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));
    flag_.ReshapeLike(X(0));

    if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else {
        LOG(FATAL) << DTypeString(
            X(0), { "float32" }
        );
    }
}

DEPLOY_CPU(SigmoidCrossEntropy);
#ifdef WITH_CUDA
DEPLOY_CUDA(SigmoidCrossEntropy);
#endif

DEPLOY_CPU(SigmoidCrossEntropyGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SigmoidCrossEntropyGradient);
#endif

OPERATOR_SCHEMA(SigmoidCrossEntropy)
     /* X, T */
    .NumInputs(2)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(SigmoidCrossEntropyGradient)
     /* X, T, dY */
    .NumInputs(3)
     /* dX */
    .NumOutputs(1);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GradientMaker);
    vector<OperatorDef> MakeDef() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), GO(0) }),
            vector<string>({ GI(0) })
        );
    }
};

}  // namespace

REGISTER_GRADIENT(SigmoidCrossEntropy, GradientMaker);

}  // namespace dragon