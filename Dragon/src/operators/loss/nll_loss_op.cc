#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/loss/nll_loss_op.h"

namespace dragon {

template <class Context> template <typename Tx, typename Ty>
void NLLLossOp<Context>::RunWithType() {
    auto* LPdata = Input(0).template data<Tx, Context>();
    auto* Tdata = Input(1).template data<Ty, Context>();
    auto* Idata = !ignores.count() ? nullptr :
        ignores.template data<int, Context>();
    auto* Ldata = losses.template mutable_data<float, Context>();
    auto* Fdata = flags.template mutable_data<float, Context>();

    kernel::NLLLoss<Tx, Ty, Context>(
        outer_dim, Input(0).dim(axis), inner_dim,
            LPdata, Tdata, Idata, ignores.count(),
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
void NLLLossOp<Context>::RunOnDevice() {
    ctx()->set_stream_id(0);  //  enforce default stream

    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
    CHECK_EQ(outer_dim * inner_dim, Input(1).count())
        << "\nNumber of predictions must match the number of labels.";

    losses.Reshape({ outer_dim * inner_dim });
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

DEPLOY_CPU(NLLLoss);
#ifdef WITH_CUDA
DEPLOY_CUDA(NLLLoss);
#endif
OPERATOR_SCHEMA(NLLLoss).NumInputs(2).NumOutputs(1);

template <class Context> template <typename Tx, typename Ty>
void NLLLossGradientOp<Context>::RunWithType() {
    auto* LPdata = Input(0).template data<Tx, Context>();
    auto* Tdata = Input(1).template data<Ty, Context>();
    auto* Idata = !ignores.count() ? nullptr :
        ignores.template data<int, Context>();
    auto* dXdata = Output(0)->template mutable_data<Tx, Context>();
    auto* Fdata = flags.template mutable_data<float, Context>();

    math::Set<Tx, Context>(Output(0)->count(),
        dragon_cast<Tx, float>(0.) , dXdata, ctx());
   
    kernel::NLLLossGrad<Tx, Ty, Context>(
        outer_dim, Output(0)->dim(axis), inner_dim,
            LPdata, Tdata, Idata, ignores.count(),
                dXdata, Fdata, ctx());

    if (normalization == "UNIT") {
        auto* dYdata = Input(-1).template data<float, Context>();
        vector<void*> WSdata = ws()->template caches<Context>(
            { Input(0).count() * sizeof(float),
              Input(0).count() * sizeof(Tx) });
        kernel::SumGrad<float, Context>(
            Input(0).count() / Input(0).dim(axis),
                Input(0).dim(axis), inner_dim,
                    1.0, dYdata, (float*)WSdata[0], ctx());
        kernel::TypeA2B<float, Tx, Context>(Input(0).count(),
            (const float*)WSdata[0], (Tx*)WSdata[1], ctx());
        math::Mul<Tx, Context>(Output(0)->count(),
            (Tx*)WSdata[1], dXdata, dXdata, ctx());
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
void NLLLossGradientOp<Context>::RunOnDevice() {
    ctx()->set_stream_id(0);  //  enforce default stream

    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
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

DEPLOY_CPU(NLLLossGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(NLLLossGradient);
#endif
OPERATOR_SCHEMA(NLLLossGradient).NumInputs(3).NumOutputs(1);

class GetNLLLossGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetNLLLossGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(NLLLoss, GetNLLLossGradient);

}    // namespace dragon