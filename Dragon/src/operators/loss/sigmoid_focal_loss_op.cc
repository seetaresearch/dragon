#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/loss/sigmoid_focal_loss_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGS(X) \
    axis_ = OpArg<int64_t>("axis", 1); \
    axis_ = axis_ < 0 ? axis_ + X.ndim() : axis_; \
    CHECK(axis_ >= 0 && axis_ < X.ndim()) \
        << "\nExcepted the axis in [-" << X.ndim() \
        << ", " << X.ndim() << "), got " \
        << OpArg<int64_t>("axis", 1) << ".";

template <class Context>
template <typename Tx, typename Ty>
void SigmoidFocalLossOp<Context>::RunImpl() {
    auto* logit  = X(0).template data<Tx, Context>();
    auto* target = X(1).template data<Ty, Context>();
    auto* loss = loss_.template mutable_data<Tx, Context>();
    auto* flag = flag_.template mutable_data<int, Context>();

    kernel::SigmoidFocalLoss(
        outer_dim_,
        X(0).dim(axis_),
        inner_dim_,
        pos_alpha_,
        neg_alpha_,
        gamma_,
        neg_id_,
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
    auto* y = Y(0)->template mutable_data<Tx, Context>();
    math::Sum(loss_.count(), 1. / normalizer, loss, y, ctx());
}

template <class Context>
void SigmoidFocalLossOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));

    outer_dim_ = X(0).count(0, axis_);
    inner_dim_ = X(0).count(axis_ + 1);
    CHECK_EQ(outer_dim_ * inner_dim_, X(1).count())
        << "\nNum of preds must match the num of labels.";

    loss_.ReshapeLike(X(0));
    flag_.ReshapeLike(X(0));

    if (XIsType(X(0), float)) {
        if (XIsType(X(1), float)) {
            RunImpl<float, float>();
        } else if (XIsType(X(1), int64_t)) {
            RunImpl<float, int64_t>();
        } else {
            LOG(FATAL) << DTypeString(X(1),
                { "float32", "int64" }
            );
        }
    } else {
        LOG(FATAL) << DTypeString(
            X(0), { "float32" }
        );
    }
}

template <class Context> template <typename Tx, typename Ty>
void SigmoidFocalLossGradientOp<Context>::RunImpl() {
    auto* logit = X(0).template data<Tx, Context>();
    auto* target = X(1).template data<Ty, Context>();
    auto* dx = Y(0)->template mutable_data<Tx, Context>();
    auto* flag = flag_.template mutable_data<int, Context>();

    kernel::SigmoidFocalLossGrad(
        outer_dim_,
        X(0).dim(axis_),
        inner_dim_,
        pos_alpha_,
        neg_alpha_,
        gamma_,
        neg_id_,
        logit, target,
        dx, flag, ctx()
    );

    if (reduction_ == "NONE") {
        auto* dy = X(-1).template data<Tx, Context>();
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

    auto* dy = X(-1).template data<Tx, Context>();
    Tx dyHost; ctx()->template Copy
        <Tx, CPUContext, Context>(
            1, &dyHost, dy);
    ctx()->FinishDeviceCompution();

    math::Scale(
        Y(0)->count(),
        dyHost / normalizer,
        dx, dx, ctx()
    );
}

template <class Context>
void SigmoidFocalLossGradientOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));

    outer_dim_ = X(0).count(0, axis_);
    inner_dim_ = X(0).count(axis_ + 1);

    Y(0)->ReshapeLike(X(0));
    flag_.ReshapeLike(X(0));

    if (XIsType(X(0), float)) {
        if (XIsType(X(1), float)) {
            RunImpl<float, float>();
        } else if (XIsType(X(1), int64_t)) {
            RunImpl<float, int64_t>();
        } else {
            LOG(FATAL) << DTypeString(X(1),
                { "float32", "int64" }
            );
        }
    } else {
        LOG(FATAL) << DTypeString(X(0),
            { "float32" }
        );
    }
}

DEPLOY_CPU(SigmoidFocalLoss);
#ifdef WITH_CUDA
DEPLOY_CUDA(SigmoidFocalLoss);
#endif

DEPLOY_CPU(SigmoidFocalLossGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SigmoidFocalLossGradient);
#endif

OPERATOR_SCHEMA(SigmoidFocalLoss)
     /* X, T */
    .NumInputs(2)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(SigmoidFocalLossGradient)
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

REGISTER_GRADIENT(SigmoidFocalLoss, GradientMaker);

#undef DETERMINE_RUNTIME_ARGS

}  // namespace dragon