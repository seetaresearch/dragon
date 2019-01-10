#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "utils/proto_utils.h"
#include "operators/loss/softmax_ce_loss_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGUMENTS(X) \
    axis = OperatorBase::Arg<int64_t>("axis", 1); \
    axis = axis < 0 ? axis + X.ndim() : axis; \
    CHECK(axis >= 0 && axis < X.ndim()) \
       << "\nExcepted the axis in [-" << X.ndim() << ", " << X.ndim() \
       << "), got " << OperatorBase::Arg<int64_t>("axis", 1) << ".";

template <class Context>
void SoftmaxCrossEntropyOp<Context>::SoftmaxRun() {
    OperatorDef softmax_def = MakeOperatorDef("Softmax", "",
        vector<string>({ Input(0).name() }),
        vector<string>({ mount_name("softmax/prob") }));
    softmax_def.add_arg()->CopyFrom(this->arg("axis"));
    if (def().has_device_option())
        softmax_def.mutable_device_option()
            ->CopyFrom(def().device_option());
    if (softmax_op) { softmax_op->MutableOp(softmax_def); }
    else { softmax_op.reset(CreateOperator(softmax_def, ws())); }
    softmax_op->Run();
}

template <class Context> template <typename T>
void SoftmaxCrossEntropyOp<Context>::RunWithType() {
    auto* Pdata = prob->template data<T, Context>();
    auto* Tdata = Input(1).template data<T, Context>();
    auto* Ldata = losses.template mutable_data<T, Context>();

    kernel::SoftmaxCrossEntropy<T, Context>(
        Input(0).count(), Pdata, Tdata, Ldata, ctx());

    if (normalization == "UNIT") {
        vector<int64_t> output_dims = Input(0).dims();
        output_dims.erase(output_dims.begin() + axis);
        Output(0)->Reshape(output_dims);
        vector<int> dims = {
            (int)outer_dim, (int)Input(0).dim(axis),
                (int)inner_dim }, axes = { 1 };
        auto* Ydata = Output(0)->template mutable_data<T, Context>();
        kernel::ReduceSum<T, Context>(3, dims.data(), 1, axes.data(),
            1.f, Ldata, Ydata, ctx()); return;
    }

    T normalizer = 1;
    if (normalization == "BATCH_SIZE") {
        normalizer = Input(0).dim(0);
    } else if (normalization == "FULL") {
        normalizer = outer_dim * inner_dim;
    }

    Output(0)->Reshape(vector<int64_t>());
    auto* Ydata = Output(0)->template mutable_data<float, Context>();
    math::Sum(losses.count(), 1.f / normalizer, Ldata, Ydata, ctx());
}

template <class Context>
void SoftmaxCrossEntropyOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGUMENTS(Input(0));

    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
    CHECK_EQ(Input(0).count(), Input(1).count())
        << "\nNumber of predictions must match the number of labels.";
    losses.ReshapeLike(Input(0));
    prob = ws()->CreateTensor(mount_name("softmax/prob"));

    SoftmaxRun();

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(SoftmaxCrossEntropy);
#ifdef WITH_CUDA
DEPLOY_CUDA(SoftmaxCrossEntropy);
#endif
OPERATOR_SCHEMA(SoftmaxCrossEntropy).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void SoftmaxCrossEntropyGradientOp<Context>::RunWithType() {
    auto* Tdata = Input(1).template data<T, Context>();
    auto* Pdata = prob->template mutable_data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    ctx()->template Copy<T, Context, Context>(prob->count(), dXdata, Pdata);
    math::Axpy<T, Context>(Output(0)->count(),
        -1.f, Tdata, dXdata, ctx());

    if (normalization == "UNIT") {
        auto* dYdata = Input(-1).template data<T, Context>();
        kernel::Repeat(outer_dim, 1, inner_dim,
            Input(0).dim(axis), dYdata, Pdata, ctx());
        math::Mul<T, Context>(Output(0)->count(),
            Pdata, dXdata, dXdata, ctx()); return;
    }

    T normalizer = 1;
    if (normalization == "BATCH_SIZE") {
        normalizer = Input(0).dim(0);
    } else if (normalization == "FULL") {
        normalizer = outer_dim * inner_dim;
    }

    auto* dYdata = Input(-1).template data<T, Context>();
    T dYHost; ctx()->template Copy
        <T, CPUContext, Context>(1, &dYHost, dYdata);
    math::Scale<T, Context>(Output(0)->count(),
        dYHost / normalizer, dXdata, dXdata, ctx());
}

template <class Context>
void SoftmaxCrossEntropyGradientOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGUMENTS(Input(0));

    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
    Output(0)->ReshapeLike(Input(0));

    prob = ws()->GetTensor(mount_name("softmax/prob"));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(SoftmaxCrossEntropyGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SoftmaxCrossEntropyGradient);
#endif

OPERATOR_SCHEMA(SoftmaxCrossEntropyGradient)
    .NumInputs(3).NumOutputs(1);

class GetSoftmaxCrossEntropyGradient
    final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetSoftmaxCrossEntropyGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), GO(0) }),
            vector<string>({ GI(0) }));
    }
};

REGISTER_GRADIENT(
    SoftmaxCrossEntropy,
    GetSoftmaxCrossEntropyGradient
);

#undef DETERMINE_RUNTIME_ARGUMENTS

}  // namespace dragon