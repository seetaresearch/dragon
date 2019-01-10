#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/ndarray/gather_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGUMENTS(X) \
    axis = OperatorBase::Arg<int64_t>("axis", 0); \
    axis = axis < 0 ? axis + X.ndim() : axis; \
    CHECK(axis >= 0 && axis < X.ndim())

template <class Context> template <typename T>
void GatherOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* indices = Input(1).template mutable_data<int, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    kernel::CanonicalAxis(Input(1).count(), x_slice_dim, indices, ctx());

    kernel::Gather(Output(0)->count(),
        outer_dim, inner_dim,
            x_slice_dim, y_slice_dim,
                indices, Xdata, Ydata, ctx());
}

template <class Context>
void GatherOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGUMENTS(Input(0));

    output_dims = Input(0).dims();
    x_slice_dim = Input(0).dim(axis);
    output_dims[axis] = y_slice_dim = Input(1).count();
    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);

    CHECK_GT(y_slice_dim, 0) << "\nLength of indices must > 0.";

    Output(0)->Reshape(output_dims);

    CHECK(Input(1).template IsType<int>())
        << "\nThe type of indices should be int32.";

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), int)) RunWithType<int>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "int32" });
}

DEPLOY_CPU(Gather);
#ifdef WITH_CUDA
DEPLOY_CUDA(Gather);
#endif
OPERATOR_SCHEMA(Gather).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void GatherGradientOp<Context>::RunWithType() {
    auto* indices = Input(1).template data<int, Context>();
    auto* dYdata = Input(-1).template data<T, Context>();

    T* dXdata = nullptr;
    if (!acc_grad) {
        dXdata = Output(0)->template mutable_data<T, Context>();
        math::Set(Output(0)->count(), cast::to<T>(0.f), dXdata, ctx());
    } else {
        dXdata = Output(0)->template mutable_data<T, Context>();
    }

    kernel::GatherGrad(Input(-1).count(),
        outer_dim, inner_dim,
            x_slice_dim, y_slice_dim,
                indices, dYdata, dXdata, ctx());
}

template <class Context>
void GatherGradientOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGUMENTS(Input(0));

    x_slice_dim = Input(0).dim(axis);
    y_slice_dim = Input(1).count();
    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);

    Output(0)->ReshapeLike(Input(0));

    CHECK(Input(1).template IsType<int>())
        << "\nThe type of indices should be int32.";

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), int)) RunWithType<int>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "int32" });
}

DEPLOY_CPU(GatherGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(GatherGradient);
#endif

OPERATOR_SCHEMA(GatherGradient)
    .NumInputs(3).NumOutputs(1);

class GetGatherGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetGatherGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "", 
            vector<string>({ I(0), I(1), GO(0) }),
            vector<string>({ GI(0)} ));
    }
};

REGISTER_GRADIENT(Gather, GetGatherGradient);

#undef DETERMINE_RUNTIME_ARGUMENTS

}  // namespace dragon