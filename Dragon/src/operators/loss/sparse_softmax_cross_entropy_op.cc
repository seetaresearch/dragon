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

template <class Context>
void SparseSoftmaxCrossEntropyOp<Context>::SoftmaxRunFP16() {
    Tensor* XF32 = ws()->CreateTensor(
        "/mnt/" + anchor() + "/softmax/xf32");
    XF32->ReshapeLike(Input(0));
    auto* XdataF16 = Input(0).template data<float16, Context>();
    auto* XdataF32 = XF32->template mutable_data<float, Context>();
    kernel::TypeA2B<float16, float, Context>(
        Input(0).count(), XdataF16, XdataF32);
    OperatorDef softmax_def = MakeOperatorDef("Softmax", "",
        vector<string>({ XF32->name() }),
        vector<string>({ "/mnt/" + anchor() + "/softmax/prob" }));
    softmax_def.add_arg()->CopyFrom(this->arg("axis"));
    if (def().has_device_option())
        softmax_def.mutable_device_option()
            ->CopyFrom(def().device_option());
    if (!softmax_op) softmax_op.reset(
        CreateOperator(softmax_def, ws()));
    else softmax_op->MutableOp(softmax_def);
    softmax_op->Run();
}

template <class Context> template <typename Tx, typename Ty>
void SparseSoftmaxCrossEntropyOp<Context>::RunWithType() {
    auto* Pdata = prob->template data<Tx, Context>();
    auto* Tdata = Input(1).template data<Ty, Context>();
    auto* Idata = !ignores.count() ? nullptr :
        ignores.template data<int, Context>();
    auto* Ldata = losses.template mutable_data<Tx, Context>();
    auto* Fdata = flags.template mutable_data<Tx, Context>();

    kernel::SparseSoftmaxCrossEntropy<Tx, Ty, Context>(
        outer_dim, Input(0).dim(axis), inner_dim,
            Pdata, Tdata, Idata, ignores.count(),
                Ldata, Fdata, &ctx());

    if (normalization == "UNIT") {
        Output(0)->ReshapeLike(losses);
        Output(0)->template CopyFrom<Context>(losses);
        return;
    }

    Tx normalizer = 1;
    if (normalization == "VALID") {
        normalizer = std::max(
            math::ASum<Tx, Context>(
                flags.count(), Fdata), (Tx)1.f);
    } else if (normalization == "BATCH_SIZE") {
        normalizer = Input(0).dim(0);
    } else if (normalization == "FULL") {
        normalizer = outer_dim * inner_dim;
    }

    Tx loss = math::ASum<Tx, Context>(losses.count(), Ldata);
    Output(0)->Reshape({ 1 });
    auto* Ydata = Output(0)->template mutable_data<Tx, Context>();
    math::Set<Tx, Context>(1, loss / normalizer, Ydata);
}

template <class Context>
void SparseSoftmaxCrossEntropyOp<Context>::RunOnDevice() {
    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
    CHECK_EQ(outer_dim * inner_dim, Input(1).count())
        << "\nNumber of predictions must match the number of labels.";
    losses.Reshape({ outer_dim * inner_dim });
    flags.Reshape({ outer_dim * inner_dim });
    prob = ws()->CreateTensor("/mnt/" + anchor() + "/softmax/prob");

    if (XIsType(Input(0), float) ||
            XIsType(Input(0), float16)) {
        if (XIsType(Input(0), float16)) SoftmaxRunFP16();
        else SoftmaxRun();
        if (XIsType(Input(1), float)) RunWithType<float, float>();
        else if (XIsType(Input(1), int64_t)) RunWithType<float, int64_t>();
        else LOG(FATAL) << DTypeHelper(Input(1), { "float32", "int64" });
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
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
    auto* Fdata = flags.template mutable_data<Tx, Context>();
    ctx().template Copy<Tx, Context, Context>(
        prob->count(), dXdata, Pdata);

    kernel::SparseSoftmaxCrossEntropyGrad<Tx, Ty, Context>(
        outer_dim, Output(0)->dim(axis), inner_dim,
            Pdata, Tdata, Idata, ignores.count(),
                dXdata, Fdata, &ctx());

    if (normalization == "UNIT") {
        auto* dYdata = Input(-1).template data<Tx, Context>();
        kernel::SumGrad<Tx, Context>(
            Input(0).count() / Input(0).dim(axis),
                Input(0).dim(axis), inner_dim,
                    1.0, dYdata, Pdata);
        math::Mul<Tx, Context>(
            Output(0)->count(), Pdata, dXdata, dXdata);
        return;
    }

    Tx normalizer = 1;
    if (normalization == "VALID") {
        normalizer = std::max(
            math::ASum<Tx, Context>(
                flags.count(), Fdata), (Tx)1.f);
    } else if (normalization == "BATCH_SIZE") {
        normalizer = Input(0).dim(0);
    } else if (normalization == "FULL") {
        normalizer = outer_dim * inner_dim;
    }

    auto* dYdata = Input(-1).template data<Tx, Context>();
    Tx dYdata_host; ctx().template Copy<Tx, CPUContext, Context>(
        1, &dYdata_host, dYdata);
    math::Scal<Tx, Context>(Output(0)->count(),
        dYdata_host / normalizer, dXdata, &ctx());
}

template <class Context>
void SparseSoftmaxCrossEntropyGradientOp<Context>::RunOnDevice() {
    prob = ws()->GetTensor("/mnt/" + anchor() + "/softmax/prob");
    outer_dim = prob->count(0, axis);
    inner_dim = prob->count(axis + 1);
    Output(0)->ReshapeLike(Input(0));
    flags.Reshape({ outer_dim * inner_dim });

    if (XIsType(Input(0), float) || XIsType(Input(0), float16)) {
        if (XIsType(Input(1), float)) RunWithType<float, float>();
        else if (XIsType(Input(1), int64_t)) RunWithType<float, int64_t>();
        else LOG(FATAL) << DTypeHelper(Input(1), { "float32", "int64" });
        if (XIsType(Input(0), float16)) {
            auto* dXdataF32 = Output(0)->template data<float, Context>();
            auto* dXdataF16 = prob->template mutable_data<float16, Context>();
            kernel::TypeA2B<float, float16, Context>(Output(0)->count(), dXdataF32, dXdataF16);
            Output(0)->template CopyFrom<Context>(*prob);
        }
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