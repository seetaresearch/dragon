#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "utils/proto_utils.h"
#include "operators/loss/sparse_softmax_ce_loss_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGUMENTS(X) \
    axis = OperatorBase::Arg<int64_t>("axis", 1); \
    axis = axis < 0 ? axis + X.ndim() : axis; \
    CHECK(axis >= 0 && axis < X.ndim()) \
       << "\nExcepted the axis in [-" << X.ndim() << ", " << X.ndim() \
       << "), got " << OperatorBase::Arg<int64_t>("axis", 1) << ".";

template <class Context>
void SparseSoftmaxCrossEntropyOp<Context>::SoftmaxRun() {
    auto softmax_def = MakeOperatorDef("Softmax", "",
        vector<string>({ Input(0).name() }),
        vector<string>({ mount_name("softmax/prob") }));
    softmax_def.add_arg()->CopyFrom(this->arg("axis"));
    if (def().has_device_option())
        softmax_def.mutable_device_option()->CopyFrom(
            def().device_option());
    if (softmax_op) { softmax_op->MutableOp(softmax_def); }
    else { softmax_op.reset(CreateOperator(softmax_def, ws())); }
    softmax_op->Run();
}

template <class Context> template <typename Tx, typename Ty>
void SparseSoftmaxCrossEntropyOp<Context>::RunWithType() {
    auto* Pdata = prob->template data<Tx, Context>();
    auto* Tdata = Input(1).template data<Ty, Context>();
    auto* Idata = !ignores.count() ? nullptr :
        ignores.template data<int, Context>();
    auto* Ldata = losses.template mutable_data<float, Context>();
    auto* Fdata = flags.template mutable_data<float, Context>();

    kernel::SparseSoftmaxCrossEntropy(
        outer_dim, Input(0).dim(axis), inner_dim, ignores.count(),
            Pdata, Tdata, Idata, Ldata, Fdata, ctx());

    if (normalization == "UNIT") {
        vector<int64_t> output_dims = Input(0).dims();
        output_dims.erase(output_dims.begin() + axis);
        Output(0)->Reshape(output_dims);
        Output(0)->template CopyFrom<Context>(
            losses, ctx()); return;
    }

    float normalizer = 1;
    if (normalization == "VALID") {
        normalizer = std::max(
            math::Sum(flags.count(),
                1.f, Fdata, ctx()), 1.f);
    } else if (normalization == "BATCH_SIZE") {
        normalizer = Input(0).dim(0);
    } else if (normalization == "FULL") {
        normalizer = outer_dim * inner_dim;
    }

    Output(0)->Reshape(vector<int64_t>());
    auto* Ydata = Output(0)->template mutable_data<float, Context>();
    math::Sum(losses.count(), 1.f / normalizer, Ldata, Ydata, ctx());
}

template <class Context>
void SparseSoftmaxCrossEntropyOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGUMENTS(Input(0));

    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
    CHECK_EQ(outer_dim * inner_dim, Input(1).count())
        << "\nNumber of predictions must match the number of labels.";
    losses.Reshape({ outer_dim * inner_dim });
    flags.Reshape({ outer_dim * inner_dim });
    prob = ws()->CreateTensor(mount_name("softmax/prob"));

    SoftmaxRun();

    if (XIsType(Input(0), float)) {
        if (XIsType(Input(1), float)) RunWithType<float, float>();
        else if (XIsType(Input(1), int64_t)) RunWithType<float, int64_t>();
        else LOG(FATAL) << DTypeHelper(Input(1), { "float32", "int64" });
    } else if (XIsType(Input(0), float16)) {
        if (XIsType(Input(1), float)) RunWithType<float16, float>();
        else if (XIsType(Input(1), int64_t)) RunWithType<float16, int64_t>();
        else LOG(FATAL) << DTypeHelper(Input(1), { "float32", "int64" });
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(SparseSoftmaxCrossEntropy);
#ifdef WITH_CUDA
DEPLOY_CUDA(SparseSoftmaxCrossEntropy);
#endif
OPERATOR_SCHEMA(SparseSoftmaxCrossEntropy).NumInputs(2).NumOutputs(1);

template <class Context> template <typename Tx, typename Ty>
void SparseSoftmaxCrossEntropyGradientOp<Context>::RunWithType() {
    auto* Pdata = prob->template mutable_data<Tx, Context>();
    auto* Tdata = Input(1).template data<Ty, Context>();
    auto* Idata = !ignores.count() ? nullptr :
        ignores.template data<int, Context>();
    auto* dXdata = Output(0)->template mutable_data<Tx, Context>();
    auto* Fdata = flags.template mutable_data<float, Context>();
    ctx()->template Copy<Tx, Context, Context>(
        prob->count(), dXdata, Pdata);

    kernel::SparseSoftmaxCrossEntropyGrad(
        outer_dim, Output(0)->dim(axis), inner_dim, ignores.count(),
            Pdata, Tdata, Idata, dXdata, Fdata, ctx());

    if (normalization == "UNIT") {
        auto* dYdata = Input(-1).template data<float, Context>();
        auto* WSdata = ws()->template caches
            <float, Context>({ Input(0).count() })[0];
        kernel::Repeat(outer_dim, 1, inner_dim,
            Input(0).dim(axis), dYdata, WSdata, ctx());
        kernel::TypeA2B(Input(0).count(), WSdata, Pdata, ctx());
        math::Mul(Output(0)->count(), Pdata, dXdata, dXdata, ctx());
        return;
    }

    float normalizer = 1;
    if (normalization == "VALID") {
        normalizer = std::max(
            math::Sum(flags.count(),
                1.f, Fdata, ctx()), 1.f);
    } else if (normalization == "BATCH_SIZE") {
        normalizer = Input(0).dim(0);
    } else if (normalization == "FULL") {
        normalizer = outer_dim * inner_dim;
    }

    auto* dYdata = Input(-1).template data<float, Context>();
    float dYHost; ctx()->template Copy<
        float, CPUContext, Context>(1, &dYHost, dYdata);
    math::Scale(Output(0)->count(),
        dYHost / normalizer, dXdata, dXdata, ctx());
}

template <class Context>
void SparseSoftmaxCrossEntropyGradientOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGUMENTS(Input(0));

    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
    Output(0)->ReshapeLike(Input(0));
    flags.Reshape({ outer_dim * inner_dim });

    prob = ws()->GetTensor(mount_name("softmax/prob"));

    if (XIsType(Input(0), float)) {
        if (XIsType(Input(1), float)) RunWithType<float, float>();
        else if (XIsType(Input(1), int64_t)) RunWithType<float, int64_t>();
        else LOG(FATAL) << DTypeHelper(Input(1), { "float32", "int64" });
    } else if (XIsType(Input(0), float16)) {
        if (XIsType(Input(1), float)) RunWithType<float16, float>();
        else if (XIsType(Input(1), int64_t)) RunWithType<float16, int64_t>();
        else LOG(FATAL) << DTypeHelper(Input(1), { "float32", "int64" });
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(SparseSoftmaxCrossEntropyGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SparseSoftmaxCrossEntropyGradient);
#endif

OPERATOR_SCHEMA(SparseSoftmaxCrossEntropyGradient)
    .NumInputs(3).NumOutputs(1);

class GetSparseSoftmaxCrossEntropyGradient
    final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetSparseSoftmaxCrossEntropyGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), GO(0) }),
            vector<string>({ GI(0) }));
    }
};

REGISTER_GRADIENT(
    SparseSoftmaxCrossEntropy,
    GetSparseSoftmaxCrossEntropyGradient
);

#undef DETERMINE_RUNTIME_ARGUMENTS

}  // namespace dragon