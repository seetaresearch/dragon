#include "operators/ndarray/at_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void AtOp<Context>::RunWithType() {
    auto* Xdata = input(0).template data<T, Context>();
    auto* indices = input(1).template mutable_data<int, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    kernel::CanonicalAxis<int, Context>(input(1).count(), x_slice_dim, indices);
    kernel::At<T, Context>(output(0)->count(), outer_dim, inner_dim,
                                           x_slice_dim, y_slice_dim,
                                                            indices,
                                                              Xdata,
                                                              Ydata,
                                                            &ctx());
}

template <class Context>
void AtOp<Context>::RunOnDevice() {
    output_dims = input(0).dims();
    x_slice_dim = input(0).dim(axis);
    output_dims[axis] = y_slice_dim = input(1).count();
    CHECK_GT(y_slice_dim, 0) << "\nLength of indices must > 0.";
    outer_dim = input(0).count(0, axis);
    inner_dim = input(0).count(axis + 1);
    output(0)->Reshape(output_dims);

    CHECK(input(1).template IsType<int>()) << "\nThe type of indices should be int32.";
    if (input(0).template IsType<float>()) RunWithType<float>();
    else if (input(0).template IsType<int>()) RunWithType<int>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(At);
#ifdef WITH_CUDA
DEPLOY_CUDA(At);
#endif
OPERATOR_SCHEMA(At).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void AtGradientOp<Context>::RunWithType() {
    auto* indices = input(1).template data<int, Context>();
    auto* dYdata = input(-1).template data<T, Context>();
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    if (!acc_grad) math::Set<T, Context>(output(0)->count(), 0, dXdata);
    kernel::AtGrad<T, Context>(input(-1).count(), outer_dim, inner_dim,
                                              x_slice_dim, y_slice_dim,
                                                               indices,
                                                                dYdata,
                                                                dXdata);
}

template <class Context>
void AtGradientOp<Context>::RunOnDevice() {
    x_slice_dim = input(0).dim(axis);
    y_slice_dim = input(1).count();
    outer_dim = input(0).count(0, axis);
    inner_dim = input(0).count(axis + 1);
    output(0)->ReshapeLike(input(0));

    CHECK(input(1).template IsType<int>()) << "\nThe type of indices should be int32.";
    if (input(0).template IsType<float>()) RunWithType<float>();
    else if (input(0).template IsType<int>()) RunWithType<int>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(AtGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(AtGradient);
#endif
OPERATOR_SCHEMA(AtGradient).NumInputs(3).NumOutputs(1);

class GetAtGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetAtGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "", 
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(At, GetAtGradient);

}    // namespace dragon