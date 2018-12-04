#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "utils/proto_utils.h"
#include "operators/activation/softmax_op.h"
#include "operators/loss/softmax_ce_loss_op.h"

namespace dragon {

template <class Context>
void SoftmaxCrossEntropyOp<Context>::SoftmaxRun() {
    OperatorDef softmax_def = MakeOperatorDef("Softmax", "",
        vector<string>({ Input(0).name() }),
        vector<string>({ "/mnt/" + anchor() + "/softmax/prob" }));
    softmax_def.add_arg()->CopyFrom(this->arg("axis"));
    if (def().has_device_option())
        softmax_def.mutable_device_option()
            ->CopyFrom(def().device_option());
    if (!softmax_op) softmax_op.reset(CreateOperator(softmax_def, ws()));
    else softmax_op->MutableOp(softmax_def);
    softmax_op->Run();
}

template <class Context> template <typename T>
void SoftmaxCrossEntropyOp<Context>::RunWithType() {
    auto* Pdata = prob->template data<T, Context>();
    auto* Tdata = Input(1).template data<T, Context>();
    auto* Ldata = losses.template mutable_data<T, Context>();
    kernel::SoftmaxCrossEntropy<T, Context>(Input(0).count(),
        Pdata, Tdata, Ldata, ctx());

    if (normalization == "UNIT") {
        Output(0)->Reshape({ outer_dim * inner_dim });
        auto* Ydata = Output(0)->template mutable_data<T, Context>();
        kernel::Sum<T, Context>(outer_dim * inner_dim,
            Input(0).dim(axis), inner_dim,
                Ldata, Ydata, ctx()); return;
    }

    T normalizer = 1;
    if (normalization == "BATCH_SIZE") {
        normalizer = Input(0).dim(0);
    } else if (normalization == "FULL") {
        normalizer = outer_dim * inner_dim;
    }

    T loss = math::ASum<T, Context>(losses.count(), Ldata);
    Output(0)->Reshape({ 1 });
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    math::Set<T, Context>(1, loss / normalizer, Ydata, ctx());
}

template <class Context>
void SoftmaxCrossEntropyOp<Context>::RunOnDevice() {
    ctx()->set_stream_id(0);  // Enforce SyncStream

    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
    CHECK_EQ(Input(0).count(), Input(1).count())
        << "\nNumber of predictions must match the number of labels.";
    losses.ReshapeLike(Input(0));
    ws()->CreateTensor("/mnt/" + anchor() + "/softmax/prob");
    SoftmaxRun();
    prob = ws()->GetTensor("/mnt/" + anchor() + "/softmax/prob");

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
        kernel::SumGrad<T, Context>(outer_dim * inner_dim,
            Input(0).dim(axis), inner_dim, 1.f, dYdata, Pdata, ctx());
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
    T dYdata_host; ctx()->template Copy
        <T, CPUContext, Context>(
            1, &dYdata_host, dYdata);
    math::Scal<T, Context>(Output(0)->count(),
        dYdata_host / normalizer, dXdata, ctx());
}

template <class Context>
void SoftmaxCrossEntropyGradientOp<Context>::RunOnDevice() {
    prob = ws()->GetTensor("/mnt/" + anchor() + "/softmax/prob");
    outer_dim = prob->count(0, axis);
    inner_dim = prob->count(axis + 1);
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(SoftmaxCrossEntropyGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SoftmaxCrossEntropyGradient);
#endif
OPERATOR_SCHEMA(SoftmaxCrossEntropyGradient).NumInputs(3).NumOutputs(1);

class GetSoftmaxCrossEntropyGradient
    final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetSoftmaxCrossEntropyGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(
    SoftmaxCrossEntropy,
    GetSoftmaxCrossEntropyGradient
);

}  // namespace dragon