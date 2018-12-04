#include "core/workspace.h"
#include "utils/math_functions.h"
#include "operators/norm/l2_norm_op.h"

namespace dragon {

template <class Context> template <typename T>
void L2NormOp<Context>::RunWithType() {
    DECLARE_MULTIPLIER(Dmult, dim);

    // Normalize by outer dim independently
    vector<TIndex> dims = Input(0).dims();
    for (int i = 0; i < axis; i++) dims[i] = 1;
    buffer.Reshape(dims);

    // Normalize by inner_dim independently if not across it
    norm = ws()->CreateTensor(
        "/mnt/" + anchor() + "/l2norm/normalizer");
    dims = Input(0).dims();
    for (int i = axis; i < end_axis; i++) dims[i] = 1;
    norm->Reshape(dims);

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    auto* Bdata = ws()->template caches<T, Context>({ buffer.count() })[0];
    auto* Ndata = norm->template mutable_data<T, Context>();
    math::Set<T, Context>(norm->count(),
        dragon_cast<T, float>(eps), Ndata, ctx());

    for (int n = 0; n < outer_dim; n++) {
        math::Square<T, Context>(buffer.count(),
            Xdata, Bdata, ctx());
        // Compute T1 = \sum_{i} x_{i,j}^{2}
        math::Gemv<T, Context>(
            CblasTrans, dim, inner_dim,
                mode == "MEAN" ? 1.f / dim : 1.f, Bdata, Dmult,
                    1.f, Ndata, ctx());
        // Compute T2 = \sqrt{T1}
        math::Sqrt<T, Context>(inner_dim, Ndata, Ndata, ctx());
        // Compute T3 = x / [(T2)]_{dim}
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                dim, inner_dim, 1,
                    1.f, Dmult, Ndata,
                        0.f, Bdata, ctx());
        math::Div<T, Context>(buffer.count(),
            Xdata, Bdata, Ydata, ctx());
        Ndata += inner_dim;
        Xdata += buffer.count();
        Ydata += buffer.count();
    }
}

template <class Context>
void L2NormOp<Context>::RunOnDevice() {
    if (num_axes >= 0) {
        if (num_axes == 0) num_axes += 1;
    } else num_axes = (int)Input(0).ndim() - axis;
    end_axis = axis + num_axes;
    CHECK_LE(end_axis, int(Input(0).ndim()));

    // Do statistics through [axis, end_axis)
    outer_dim = Input(0).count(0, axis);
    dim = Input(0).count(axis, axis + num_axes);
    inner_dim = Input(0).count(axis + num_axes);

    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(L2Norm);
#ifdef WITH_CUDA
DEPLOY_CUDA(L2Norm);
#endif
OPERATOR_SCHEMA(L2Norm).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void L2NormGradientOp<Context>::RunWithType() {
    DECLARE_MULTIPLIER(Dmult, dim);

    // Normalize by inner_dim independently if not across it
    norm = ws()->GetTensor("/mnt/" + anchor() + "/l2norm/normalizer");
    vector<TIndex> dims = Input(0).dims();
    for (int i = 0; i < axis; i++) dims[i] = 1;
    buffer.Reshape(dims);
    buffer_inner.Reshape({ inner_dim });
    vector<T*> BSdata = ws()->template caches<T, Context>(
        { buffer.count(), buffer_inner.count() });

    auto* Xdata = Input(0).template data<T, Context>();
    auto* dYdata = Input(-1).template data<T, Context>();
    auto* Ndata = norm->template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* Bdata = BSdata[0], *BInnerdata = BSdata[1];

    for (int n = 0; n < outer_dim; n++) {
        // Compute \sum_{i} x_{i, j}dy_{i, j}
        math::Mul<T, Context>(buffer.count(),
            Xdata, dYdata, Bdata, ctx());
        math::Gemv<T, Context>(
            CblasTrans, dim, inner_dim,
                mode == "MEAN" ? 1.f / dim : 1.f, Bdata, Dmult,
                    0.f, BInnerdata, ctx());
        // Compute T1 = x[(\sum_{i} x_{i, j}dy_{i, j})]_{dim}
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                dim, inner_dim, 1,
                    1.f, Dmult, BInnerdata,
                        0.f, Bdata, ctx());
        math::Mul<T, Context>(buffer.count(),
            Xdata, Bdata, dXdata, ctx());
        // Compute T2 = T1 / Normalizer^{2}
        math::Pow<T, Context>(inner_dim,
            2.f, Ndata, BInnerdata, ctx());
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                dim, inner_dim, 1,
                    1.f, Dmult, BInnerdata,
                        0.f, Bdata, ctx());
        math::Div<T, Context>(buffer.count(),
            dXdata, Bdata, dXdata, ctx());
        // Compute T3 = (dy - T2) / Normalizer
        math::Sub<T, Context>(buffer.count(),
            dYdata, dXdata, dXdata, ctx());
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                dim, inner_dim, 1,
                    1.f, Dmult, Ndata,
                        0.f, Bdata, ctx());
        math::Div<T, Context>(buffer.count(),
            dXdata, Bdata, dXdata, ctx());
        Ndata += inner_dim;
        Xdata += buffer.count();
        dYdata += buffer.count();
        dXdata += buffer.count();
    }
}

template <class Context>
void L2NormGradientOp<Context>::RunOnDevice() {
    if (num_axes >= 0) {
        if (num_axes == 0) num_axes += 1;
    } else { num_axes = (int)Input(0).ndim() - axis; }
    end_axis = axis + num_axes;
    CHECK_LE(end_axis, int(Input(0).ndim()));

    // Do statistics through [axis, end_axis)
    outer_dim = Input(0).count(0, axis);
    dim = Input(0).count(axis, axis + num_axes);
    inner_dim = Input(0).count(axis + num_axes);

    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(L2NormGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(L2NormGradient);
#endif
OPERATOR_SCHEMA(L2NormGradient).NumInputs(2).NumOutputs(1);

class GetL2NormGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetL2NormGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(L2Norm, GetL2NormGradient);

}  // namespace dragon