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
void AffineGradientOp<Context>::RunWithType() {
    auto* dYdata = Input(-1).template mutable_data<T, Context>();
    auto* Adata = Input(1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();

    // dA = X * dY
    if (Output(1)->name() != "ignore") {
        Output(1)->ReshapeLike(Input(1));
        auto* Xdata = Input(0).template data<T, Context>();
        auto* dAdata = Output(1)->template mutable_data<T, Context>();
        // Eltwise
        if (Input(0).count() == Input(1).count()) {
            math::Mul(Output(0)->count(), dYdata, Xdata, dAdata, ctx());
        } else {
            math::Mul(Output(0)->count(), dYdata, Xdata, dXdata, ctx());
            ComputeScaleGradient<T>(dXdata, dAdata);
        }
    }

    // dB = dY
    if (Output(2)->name() != "ignore") {
        Output(2)->ReshapeLike(Input(1));
        auto* dBdata = Output(2)->template mutable_data<T, Context>();
        // Eltwise
        if (Input(-1).count() == Input(1).count()) {
            ctx()->template Copy<T, Context, Context>(
                Output(2)->count(), dBdata, dYdata);
        } else {
            ComputeScaleGradient<T>(dYdata, dBdata);
        }
    }

    // dX = alpha * dY
    if (Output(0)->name() != "ignore") {
        kernel::AffineGrad(outer_dim, inner_dim, scale_dim,
            dYdata, Adata, dXdata, ctx());
    }
}

template <class Context> template <typename T>
void AffineGradientOp<Context>::ComputeScaleGradient(
    T*                      dYxX,
    T*                      dA) {
    DECLARE_MULTIPLIER(multiplier, sum_dim);
    T* SRes_data = nullptr;
    // Reduce inner dimensions
    if (inner_dim == 1) {
        SRes_data = dYxX;
    } else {
        SRes_data = (outer_dim == 1) ?
            dA : ws()->template caches<T, Context>(
                { outer_dim * scale_dim })[0];
        math::Gemv(
            CblasNoTrans, outer_dim * scale_dim, inner_dim,
                1.f, dYxX, multiplier,
                    0.f, SRes_data, ctx());
    }
    // Reduce outer dimensions
    if (outer_dim != 1) {
        math::Gemv(
            CblasTrans, outer_dim, scale_dim,
                1.f, SRes_data, multiplier,
                    0.f, dA, ctx());
    }
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

    if (XIsType(Input(-1), float)) RunWithType<float>();
    else if (XIsType(Input(-1), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(-1), { "float32", "float16" });
}

DEPLOY_CPU(AffineGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(AffineGradient);
#endif

OPERATOR_SCHEMA(AffineGradient)
    .NumInputs(3).NumOutputs(3)
    .Inplace({ { 2, 0 } });

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