#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "utils/proto_utils.h"
#include "operators/loss/softmax_ce_loss_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGS(X) \
    axis_ = OpArg<int64_t>("axis", 1); \
    axis_ = axis_ < 0 ? axis_ + X.ndim() : axis_; \
    CHECK(axis_ >= 0 && axis_ < X.ndim()) \
        << "\nExcepted the axis in [-" << X.ndim() \
        << ", " << X.ndim() << "), got " \
        << OpArg<int64_t>("axis", 1) << ".";

template <class Context>
void SoftmaxCrossEntropyOp<Context>::SoftmaxRun() {
    auto softmax_def = MakeOperatorDef(
        "Softmax", "",
        vector<string>({ X(0).name() }),
        vector<string>({ unique_name("prob") })
    );
    softmax_def.add_arg()->CopyFrom(this->arg("axis"));
    if (def().has_device_option())
        softmax_def.mutable_device_option()
            ->CopyFrom(def().device_option());
    if (softmax_op_) { softmax_op_->UpdateFrom(softmax_def); }
    else { softmax_op_.reset(NewOperator(softmax_def, ws())); }
    softmax_op_->Run(ctx()->stream_id());
}

template <class Context> template <typename T>
void SoftmaxCrossEntropyOp<Context>::RunImpl() {
    auto* target = X(1).template data<T, Context>();
    auto* loss = loss_.template mutable_data<T, Context>();

    auto* prob = ws()
        ->GetTensor(unique_name("prob"))
        ->template data<T, Context>();

    kernel::SoftmaxCrossEntropy(
        X(0).count(),
        prob, target,
        loss, ctx()
    );

    if (reduction_ == "NONE") {
        auto out_shape = X(0).dims();
        out_shape.erase(out_shape.begin() + axis_);
        auto* y = Y(0)
            ->Reshape(out_shape)
            ->template mutable_data<T, Context>();
        vec32_t dims = {
            (int)outer_dim_,
            (int)X(0).dim(axis_),
            (int)inner_dim_,
        }, axes = { 1 };
        kernel::ReduceSum(
            3, dims.data(),
            1, axes.data(),
            1.f, loss,
            y, ctx()
        );
        return;
    }

    double normalizer = 1.;
    if (reduction_ == "BATCH_SIZE") {
        normalizer = X(0).dim(0);
    } else if (reduction_ == "MEAN") {
        normalizer = outer_dim_ * inner_dim_;
    }

    Y(0)->Reshape({});
    auto* y = Y(0)->template mutable_data<T, Context>();
    math::Sum(loss_.count(), 1. / normalizer, loss, y, ctx());
}

template <class Context>
void SoftmaxCrossEntropyOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));

    outer_dim_ = X(0).count(0, axis_);
    inner_dim_ = X(0).count(axis_ + 1);
    CHECK_EQ(outer_dim_ * inner_dim_, X(1).count())
        << "\nNum of preds must match the num of labels.";

    SoftmaxRun();
    loss_.ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <float>>::Call(this, X(0));
}

template <class Context> template <typename T>
void SoftmaxCrossEntropyGradientOp<Context>::RunImpl() {
    auto* target = X(1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

    auto* prob = ws()
        ->GetTensor(unique_name("prob"))
        ->template mutable_data<T, Context>();

    math::Copy(Y(0)->count(), prob, dx, ctx());

    math::Axpy(
        Y(0)->count(),
        -1.f, target,
        dx, ctx()
    );

    if (reduction_ == "NONE") {
        auto* dy = X(-1).template data<T, Context>();
        kernel::Repeat(
            outer_dim_,
            inner_dim_,
            1,
            X(0).dim(axis_),
            dy, prob, ctx()
        );
        math::Mul(
            Y(0)->count(),
            prob, dx,
            dx, ctx()
        );
        return;
    }

    double normalizer = 1;
    if (reduction_ == "BATCH_SIZE") {
        normalizer = X(0).dim(0);
    } else if (reduction_ == "MEAN") {
        normalizer = outer_dim_ * inner_dim_;
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
void SoftmaxCrossEntropyGradientOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));

    outer_dim_ = X(0).count(0, axis_);
    inner_dim_ = X(0).count(axis_ + 1);

    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <float>>::Call(this, X(0));
}

DEPLOY_CPU(SoftmaxCrossEntropy);
#ifdef WITH_CUDA
DEPLOY_CUDA(SoftmaxCrossEntropy);
#endif

DEPLOY_CPU(SoftmaxCrossEntropyGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SoftmaxCrossEntropyGradient);
#endif

OPERATOR_SCHEMA(SoftmaxCrossEntropy)
     /* X, T */
    .NumInputs(2)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(SoftmaxCrossEntropyGradient)
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

REGISTER_GRADIENT(SoftmaxCrossEntropy, GradientMaker);

#undef DETERMINE_RUNTIME_ARGS

}  // namespace dragon