#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/loss/nll_loss_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGUMENTS(X) \
    axis = OperatorBase::Arg<int64_t>("axis", 1); \
    axis = axis < 0 ? axis + X.ndim() : axis; \
    CHECK(axis >= 0 && axis < X.ndim()) \
       << "\nExcepted the axis in [-" << X.ndim() << ", " << X.ndim() \
       << "), got " << OperatorBase::Arg<int64_t>("axis", 1) << ".";

template <class Context> template <typename Tx, typename Ty>
void NLLLossOp<Context>::RunWithType() {
    auto* LPdata = Input(0).template data<Tx, Context>();
    auto* Tdata = Input(1).template data<Ty, Context>();
    auto* Idata = !ignores.count() ? nullptr :
        ignores.template data<int, Context>();
    auto* Ldata = losses.template mutable_data<float, Context>();
    auto* Fdata = flags.template mutable_data<float, Context>();

    kernel::NLLLoss(
        outer_dim, Input(0).dim(axis), inner_dim, ignores.count(),
            LPdata, Tdata, Idata, Ldata, Fdata, ctx());

    if (normalization == "UNIT") {
        auto output_dims = Input(0).dims();
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
void NLLLossOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGUMENTS(Input(0));

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

    math::Set(Output(0)->count(), cast::to<Tx>(0.f) , dXdata, ctx());
   
    kernel::NLLLossGrad(outer_dim, Output(0)->dim(axis), inner_dim,
        ignores.count(), LPdata, Tdata, Idata, dXdata, Fdata, ctx());

    if (normalization == "UNIT") {
        auto* dYdata = Input(-1).template data<float, Context>();
        vector<void*> WS = ws()->template caches<Context>(
            { Input(0).count() * sizeof(float),
                  Input(0).count() * sizeof(Tx) });
        kernel::Repeat(outer_dim, 1, inner_dim,
            Input(0).dim(axis), dYdata, (float*)WS[0], ctx());
        kernel::TypeA2B(Input(0).count(),
            (const float*)WS[0], (Tx*)WS[1], ctx());
        math::Mul(Output(0)->count(),
            (Tx*)WS[1], dXdata, dXdata, ctx());
        return;
    }

    float normalizer = 1;
    if (normalization == "VALID") {
        normalizer = std::max(
            math::Sum<float, Context>(flags.count(),
                1.f, Fdata, ctx()), 1.f);
    } else if (normalization == "BATCH_SIZE") {
        normalizer = (float)Input(0).dim(0);
    } else if (normalization == "FULL") {
        normalizer = (float)(outer_dim * inner_dim);
    }

    auto* dYdata = Input(-1).template data<float, Context>();
    float dYHost; ctx()->template Copy
        <float, CPUContext, Context>(1, &dYHost, dYdata);
    math::Scale(Output(0)->count(),
        dYHost / normalizer, dXdata, dXdata, ctx());
}

template <class Context>
void NLLLossGradientOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGUMENTS(Input(0));

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

OPERATOR_SCHEMA(NLLLossGradient)
    .NumInputs(3).NumOutputs(1);

class GetNLLLossGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetNLLLossGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), GO(0) }),
            vector<string>({ GI(0) }));
    }
};

REGISTER_GRADIENT(NLLLoss, GetNLLLossGradient);

#undef DETERMINE_RUNTIME_ARGUMENTS

}  // namespace dragon