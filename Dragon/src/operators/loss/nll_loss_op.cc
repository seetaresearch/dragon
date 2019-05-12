#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/loss/nll_loss_op.h"

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
void NLLLossOp<Context>::RunImpl() {
    auto* log_p  = X(0).template data<Tx, Context>();
    auto* target = X(1).template data<Ty, Context>();
    auto* loss   = loss_.template mutable_data<Tx, Context>();
    auto* flag   = flag_.template mutable_data<int, Context>();
    auto* ignore = !ignore_.count() ? nullptr :
                    ignore_.template data<int, Context>();

    kernel::NLLLoss(
        outer_dim_,
        X(0).dim(axis_),
        inner_dim_,
        ignore_.count(),
        ignore, log_p, target,
        loss, flag, ctx()
    );

    if (reduction_ == "NONE") {
        auto out_shape = X(0).dims();
        out_shape.erase(out_shape.begin() + axis_);
        Y(0)->Reshape(out_shape)
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
        normalizer = outer_dim_ * inner_dim_;
    }

    Y(0)->Reshape({});
    auto* y = Y(0)->template mutable_data<Tx, Context>();
    math::Sum(loss_.count(), 1. / normalizer, loss, y, ctx());
}

template <class Context>
void NLLLossOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));

    outer_dim_ = X(0).count(0, axis_);
    inner_dim_ = X(0).count(axis_ + 1);
    CHECK_EQ(outer_dim_ * inner_dim_, X(1).count())
        << "\nNum of preds must match the num of labels.";
    loss_.Reshape({ outer_dim_ * inner_dim_ });
    flag_.Reshape({ outer_dim_ * inner_dim_ });

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

template <class Context> template <typename Tx, typename Ty>
void NLLLossGradientOp<Context>::RunImpl() {
    auto* log_p  = X(0).template data<Tx, Context>();
    auto* target = X(1).template data<Ty, Context>();
    auto* ignore = !ignore_.count() ? nullptr :
                    ignore_.template data<int, Context>();
    auto* dx = Y(0)->template mutable_data<Tx, Context>();
    auto* flag = flag_.template mutable_data<int, Context>();

    math::Set(
        Y(0)->count(),
        cast::to<Tx>(0.f),
        dx, ctx()
    );

    kernel::NLLLossGrad(
        outer_dim_,
        Y(0)->dim(axis_),
        inner_dim_,
        ignore_.count(),
        ignore, log_p, target,
        dx, flag, ctx()
    );

    if (reduction_ == "NONE") {
        auto* dy = X(-1).template data<Tx, Context>();
        auto* scratch = ws()
            ->template data<Tx, Context>(
                { X(0).count() })[0];
        kernel::Repeat(
            outer_dim_,
            inner_dim_,
            1,
            X(0).dim(axis_),
            dy, scratch, ctx()
        );
        math::Mul(
            Y(0)->count(),
            scratch, dx,
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
        normalizer = outer_dim_ * inner_dim_;
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
void NLLLossGradientOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));
    outer_dim_ = X(0).count(0, axis_);
    inner_dim_ = X(0).count(axis_ + 1);

    Y(0)->ReshapeLike(X(0));
    flag_.Reshape({ outer_dim_ * inner_dim_ });

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

DEPLOY_CPU(NLLLoss);
#ifdef WITH_CUDA
DEPLOY_CUDA(NLLLoss);
#endif

DEPLOY_CPU(NLLLossGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(NLLLossGradient);
#endif

OPERATOR_SCHEMA(NLLLoss)
     /* X, T */
    .NumInputs(2)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(NLLLossGradient)
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

REGISTER_GRADIENT(NLLLoss, GradientMaker);

#undef DETERMINE_RUNTIME_ARGS

}  // namespace dragon