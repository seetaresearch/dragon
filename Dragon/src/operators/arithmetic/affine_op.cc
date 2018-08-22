#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/op_kernel.h"
#include "operators/arithmetic/affine_op.h"

namespace dragon {

template <class Context> template <typename T>
void AffineOp<Context>::RunWithType() {
    start_axis = axis;
    if (start_axis < 0) start_axis += (int)Input(0).ndim();
    if (num_axes == -1) num_axes = (int)Input(0).ndim() - start_axis;
    else if (num_axes == 0) num_axes = 1;

    CHECK_LT(start_axis, (int)Input(0).ndim());
    CHECK_LE(start_axis + num_axes, (int)Input(0).ndim());

    const auto& dim_start = Input(0).dims().begin() + start_axis;
    const auto& dim_end = dim_start + num_axes;
    vector<TIndex> param_dims(dim_start, dim_end);
    TENSOR_FILL(Input(1), param_dims);;
    outer_dim = Input(0).count(0, start_axis);
    inner_dim = Input(0).count(start_axis + num_axes);
    scale_dim = Input(1).count();
    if (InputSize() > 2) { TENSOR_FILL(Input(2), param_dims); }

    DECLARE_MULTIPLIER(bias_multiplier, inner_dim);

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Adata = Input(1).template data<T, Context>();
    auto* Bdata = InputSize() > 2 ?
        Input(2).template data<T, Context>() : nullptr;
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    kernel::Affine<T, Context>(
        Output(0)->count(), outer_dim, scale_dim, inner_dim,
            Xdata, Adata, Bdata, bias_multiplier, Ydata, ctx());
}

template <class Context>
void AffineOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(Affine);
#ifdef WITH_CUDA
DEPLOY_CUDA(Affine);
#endif
OPERATOR_SCHEMA(Affine).NumInputs(2, 3).NumOutputs(1).Inplace({ { 0, 0 } });

template <class Context> template <typename T>
void AffineGradientOp<Context>::BiasRunWithType() {
    Output(2)->ReshapeLike(Input(1));
    DECLARE_MULTIPLIER(multiplier, inner_dim);

    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dBias = Output(2)->template mutable_data<T, Context>(ctx());

    for (int n = 0; n < outer_dim; n++) {
        math::Gemv<T, Context>(
            CblasNoTrans, scale_dim, inner_dim,
                1.0, dYdata, multiplier,
                    1.0, dBias, ctx());
        dYdata += dim;
    }
}

template <class Context> template <typename T>
void AffineGradientOp<Context>::ScaleRunWithType() {
    Output(0)->ReshapeLike(Input(-1));
    Output(1)->ReshapeLike(Input(1));
    DECLARE_MULTIPLIER(multiplier, sum_dim);

    sum_result.Reshape({ outer_dim * scale_dim });
    bool is_eltwise = (Input(-1).count() == Input(1).count());
    auto* dYdata = Input(-1).template data<T, Context>();
    auto* Xdata = Input(0).template data<T, Context>();
    auto* dScale = Output(1)->template mutable_data<T, Context>(ctx());
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* dYxX = dXdata;

    math::Mul<T, Context>(Output(0)->count(), dYdata, Xdata, dYxX, ctx());

    if (!is_eltwise) {
        T* SRes_data = nullptr;
        //  reduce inner dimensions
        if (inner_dim == 1) {
            SRes_data = dYxX;
        } else {
            SRes_data = (outer_dim == 1) ?
                dScale : sum_result.template mutable_data<T, Context>();
            math::Gemv<T, Context>(
                CblasNoTrans, sum_result.count(), inner_dim,
                    1.0, dYxX, multiplier,
                        SRes_data == dScale ? 1.0 : 0.0,
                            SRes_data, ctx());
        } 
        //  reduce outer dimensions
        if (outer_dim != 1) {
            math::Gemv<T, Context>(
                CblasTrans, outer_dim, scale_dim,
                    1.0, SRes_data, multiplier,
                        1.0, dScale, ctx());
        }
    } else {
        math::Axpy<T, Context>(Output(1)->count(),
            1.f, dYxX, dScale, ctx());
    }
}

template <class Context> template <typename T>
void AffineGradientOp<Context>::RunWithType() {
    Output(0)->ReshapeLike(Input(-1));

    auto* dYdata = Input(-1).template data<T, Context>();
    auto* Adata = Input(1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();

    kernel::AffineGrad<T, Context>(
        Output(0)->count(), outer_dim, scale_dim, inner_dim,
            dYdata, Adata, dXdata, ctx());
}

template <class Context>
void AffineGradientOp<Context>::RunOnDevice() {
    start_axis = axis;
    if (start_axis < 0) start_axis += (int)Input(-1).ndim();
    if (num_axes == -1) num_axes = (int)Input(-1).ndim() - start_axis;
    else if (num_axes == 0) num_axes = 1;

    CHECK_LT(start_axis, (int)Input(-1).ndim());
    CHECK_LE(start_axis + num_axes, (int)Input(-1).ndim());

    outer_dim = Input(-1).count(0, start_axis);
    inner_dim = Input(-1).count(start_axis + num_axes);
    scale_dim = Input(1).count();
    sum_dim = std::max(outer_dim, inner_dim);
    dim = scale_dim * inner_dim;

    if (XIsType(Input(-1), float)) {
        if (Output(2)->name() != "ignore") BiasRunWithType<float>();
        if (Output(1)->name() != "ignore") ScaleRunWithType<float>();
        if (Output(0)->name() != "ignore") RunWithType<float>();
    } else {
        LOG(FATAL) << DTypeHelper(Input(-1), { "float32" });
    }
}

DEPLOY_CPU(AffineGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(AffineGradient);
#endif
OPERATOR_SCHEMA(AffineGradient).NumInputs(3).NumOutputs(3);

class GetAffineGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetAffineGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0), GI(1), GI(2)});
    }
};
REGISTER_GRADIENT(Affine, GetAffineGradient);

}    // namespace dragon