#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "utils/proto_utils.h"
#include "operators/activation/softmax_op.h"
#include "operators/loss/sparse_softmax_cross_entropy_op.h"

namespace dragon {

template <class Context>
void SparseSoftmaxCrossEntropyOp<Context>::SoftmaxRun() {
    OperatorDef softmax_def = MakeOperatorDef("Softmax", "",
        vector<string>({ Input(0).name() }),
        vector<string>({ "/mnt/" + anchor() + "/softmax/prob" }));
    softmax_def.add_arg()->CopyFrom(this->arg("axis"));
    if (def().has_device_option())
        softmax_def.mutable_device_option()->CopyFrom(
            def().device_option());
    if (!softmax_op) softmax_op.reset(CreateOperator(softmax_def, ws()));
    else softmax_op->MutableOp(softmax_def);
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

    kernel::SparseSoftmaxCrossEntropy<Tx, Ty, Context>(
        outer_dim, Input(0).dim(axis), inner_dim,
            Pdata, Tdata, Idata, ignores.count(),
                Ldata, Fdata, ctx());

    if (normalization == "UNIT") {
        vector<TIndex> output_dims = Input(0).dims();
        output_dims.erase(output_dims.begin() + axis);
        Output(0)->Reshape(output_dims);
        Output(0)->template CopyFrom<Context>(losses, ctx());
        return;
    }

    float normalizer = 1;
    if (normalization == "VALID") {
        normalizer = std::max(
            math::ASum<float, Context>(
                flags.count(), Fdata), 1.f);
    } else if (normalization == "BATCH_SIZE") {
        normalizer = Input(0).dim(0);
    } else if (normalization == "FULL") {
        normalizer = outer_dim * inner_dim;
    }

    float loss = math::ASum<float, Context>(losses.count(), Ldata);
    Output(0)->Reshape({ 1 });
    auto* Ydata = Output(0)->template mutable_data<float, Context>();
    math::Set<float, Context>(1, loss / normalizer, Ydata, ctx());
}

template <class Context>
void SparseSoftmaxCrossEntropyOp<Context>::RunOnDevice() {
    ctx()->set_stream_id(0);  //  enforce default stream

    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
    CHECK_EQ(outer_dim * inner_dim, Input(1).count())
        << "\nNumber of predictions must match the number of labels.";
    losses.Reshape({ outer_dim * inner_dim });
    flags.Reshape({ outer_dim * inner_dim });

    prob = ws()->CreateTensor("/mnt/" + anchor() + "/softmax/prob");
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

    kernel::SparseSoftmaxCrossEntropyGrad<Tx, Ty, Context>(
        outer_dim, Output(0)->dim(axis), inner_dim,
            Pdata, Tdata, Idata, ignores.count(),
                dXdata, Fdata, ctx());

    if (normalization == "UNIT") {
        auto* dYdata = Input(-1).template data<float, Context>();
        auto* WSdata = ws()->template caches<float, Context>(
            { Input(0).count() })[0];
        kernel::SumGrad<float, Context>(
            Input(0).count() / Input(0).dim(axis),
                Input(0).dim(axis), inner_dim,
                    1.0, dYdata, WSdata, ctx());
        kernel::TypeA2B<float, Tx, Context>(
            Input(0).count(), WSdata, Pdata, ctx());
        math::Mul<Tx, Context>(Output(0)->count(),
            Pdata, dXdata, dXdata, ctx());
        return;
    }

    float normalizer = 1;
    if (normalization == "VALID") {
        normalizer = std::max(
            math::ASum<float, Context>(
                flags.count(), Fdata), 1.f);
    } else if (normalization == "BATCH_SIZE") {
        normalizer = Input(0).dim(0);
    } else if (normalization == "FULL") {
        normalizer = outer_dim * inner_dim;
    }

    auto* dYdata = Input(-1).template data<float, Context>();
    float dYdata_host; ctx()->template Copy<float, CPUContext, Context>(
        1, &dYdata_host, dYdata);
    math::Scal<Tx, Context>(Output(0)->count(),
        dYdata_host / normalizer, dXdata, ctx());
}

template <class Context>
void SparseSoftmaxCrossEntropyGradientOp<Context>::RunOnDevice() {
    ctx()->set_stream_id(0);  //  enforce default stream

    prob = ws()->GetTensor("/mnt/" + anchor() + "/softmax/prob");
    outer_dim = prob->count(0, axis);
    inner_dim = prob->count(axis + 1);
    Output(0)->ReshapeLike(Input(0));
    flags.Reshape({ outer_dim * inner_dim });

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
OPERATOR_SCHEMA(SparseSoftmaxCrossEntropyGradient).NumInputs(3).NumOutputs(1);

class GetSparseSoftmaxCrossEntropyGradient
    final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetSparseSoftmaxCrossEntropyGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(
    SparseSoftmaxCrossEntropy,
    GetSparseSoftmaxCrossEntropyGradient
);

}    // namespace dragon