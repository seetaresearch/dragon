#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/op_kernel.h"
#include "operators/arithmetic/affine_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGUMENTS(X) \
    axis = OperatorBase::Arg<int64_t>("axis", 1); \
    num_axes = OperatorBase::Arg<int64_t>("num_axes", 1); \
    if (axis < 0) axis += X.ndim(); \
    if (num_axes < 0) num_axes = X.ndim() - axis; \
    else if (num_axes == 0) num_axes = 1; \
    CHECK(axis >= 0 && axis + num_axes <= X.ndim())

template <class Context> template <typename T>
void AffineOp<Context>::RunWithType() {
    const auto& dim_start = Input(0).dims().begin() + axis;
    const auto& dim_end = dim_start + num_axes;
    vector<int64_t> param_dims(dim_start, dim_end);
    TENSOR_FILL(Input(1), param_dims);;
    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + num_axes);
    scale_dim = Input(1).count();
    if (InputSize() > 2) { TENSOR_FILL(Input(2), param_dims); }

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Adata = Input(1).template data<T, Context>();
    auto* Bdata = InputSize() > 2 ?
        Input(2).template data<T, Context>() : nullptr;
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    kernel::Affine(outer_dim, inner_dim, scale_dim,
        Xdata, Adata, Bdata, Ydata, ctx());
}

template <class Context>
void AffineOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGUMENTS(Input(0));
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(Affine);
#ifdef WITH_CUDA
DEPLOY_CUDA(Affine);
#endif

OPERATOR_SCHEMA(Affine)
    .NumInputs(2, 3).NumOutputs(1)
    .Inplace({ { 0, 0 } });

template <class Context> template <typename T>
void AffineGradientOp<Context>::BiasRunWithType() {
    Output(2)->ReshapeLike(Input(1));
    DECLARE_MULTIPLIER(multiplier, inner_dim);

    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dBias = Output(2)->template mutable_data<T, Context>();

    for (int n = 0; n < outer_dim; n++) {
        math::Gemv(
            CblasNoTrans, scale_dim, inner_dim,
                1.f, dYdata, multiplier,
                    1.f, dBias, ctx());
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
    auto* dScale = Output(1)->template mutable_data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* dYxX = dXdata;

    math::Mul<T, Context>(Output(0)->count(), dYdata, Xdata, dYxX, ctx());

    if (!is_eltwise) {
        T* SRes_data = nullptr;
        // Reduce inner dimensions
        if (inner_dim == 1) {
            SRes_data = dYxX;
        } else {
            SRes_data = (outer_dim == 1) ?
                dScale : sum_result.template mutable_data<T, Context>();
            math::Gemv(
                CblasNoTrans, sum_result.count(), inner_dim,
                    1.f, dYxX, multiplier,
                        SRes_data == dScale ? 1.f : 0.f,
                            SRes_data, ctx());
        } 
        // Reduce outer dimensions
        if (outer_dim != 1) {
            math::Gemv(
                CblasTrans, outer_dim, scale_dim,
                    1.f, SRes_data, multiplier,
                        1.f, dScale, ctx());
        }
    } else {
        math::Axpy(Output(1)->count(), 1.f, dYxX, dScale, ctx());
    }
}

template <class Context> template <typename T>
void AffineGradientOp<Context>::RunWithType() {
    auto* dYdata = Input(-1).template data<T, Context>();
    auto* Adata = Input(1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();

    kernel::AffineGrad(outer_dim, inner_dim, scale_dim,
        dYdata, Adata, dXdata, ctx());
}

template <class Context>
void AffineGradientOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGUMENTS(Input(0));
    Output(0)->ReshapeLike(Input(-1));

    outer_dim = Input(-1).count(0, axis);
    inner_dim = Input(-1).count(axis + num_axes);
    scale_dim = Input(1).count();
    dim = scale_dim * inner_dim;
    sum_dim = std::max(outer_dim, inner_dim);

    if (XIsType(Input(-1), float)) {
        if (Output(2)->name() != "ignore") BiasRunWithType<float>();
        if (Output(1)->name() != "ignore") ScaleRunWithType<float>();
        if (Output(0)->name() != "ignore") RunWithType<float>();
    } else if (XIsType(Input(-1), float16)) {
        if (Output(2)->name() != "ignore") BiasRunWithType<float16>();
        if (Output(1)->name() != "ignore") ScaleRunWithType<float16>();
        if (Output(0)->name() != "ignore") RunWithType<float16>();
    } else {
        LOG(FATAL) << DTypeHelper(Input(-1), { "float32", "float16" });
    }
}

DEPLOY_CPU(AffineGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(AffineGradient);
#endif

OPERATOR_SCHEMA(AffineGradient)
    .NumInputs(3).NumOutputs(3);

class GetAffineGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetAffineGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), GO(0) }),
            vector<string>({ GI(0), GI(1), GI(2) }));
    }
};

REGISTER_GRADIENT(Affine, GetAffineGradient);

#undef DETERMINE_RUNTIME_ARGUMENTS

}  // namespace dragon