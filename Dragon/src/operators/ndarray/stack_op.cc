#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "operators/ndarray/stack_op.h"

namespace dragon {

template <class Context> template <typename T>
void StackOp<Context>::RunWithType() {
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    for (int i = 0; i < InputSize(); i++) {
        auto* Xdata = Input(i).template data<T, Context>();
        TIndex count = Input(i).count();
        x_concat_dim = 1;
        kernel::Concat<T, Context>(
            count, outer_dim, inner_dim,
                x_concat_dim, y_concat_dim,
                    concat_offset, Xdata, Ydata);
        concat_offset += x_concat_dim;
    }
}

template <class Context>
void StackOp<Context>::RunOnDevice() {
    while (axis < 0) axis += (Input(0).ndim() + 1);
    stack_dims = concat_dims =  Input(0).dims();
    concat_dims.insert(concat_dims.begin() + axis, InputSize());
    for (int i = 1; i < InputSize(); i++) {
        CHECK_EQ(stack_dims.size(), Input(i).ndim())
            << "\nAll inputs should have the same ndim.";
        for (int j = 0; j < stack_dims.size(); j++)
            CHECK_EQ(stack_dims[j], Input(i).dim(j))
                << "\nAll inputs should have the same dimensions.";
    }
    y_concat_dim = concat_dims[axis];
    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis);
    concat_offset = 0;
    Output(0)->Reshape(concat_dims);

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(Stack);
#ifdef WITH_CUDA
DEPLOY_CUDA(Stack);
#endif
OPERATOR_SCHEMA(Stack).NumInputs(1, INT_MAX).NumOutputs(1);

template <class Context> template <typename T>
void StackGradientOp<Context>::RunWithType() {
    auto* dYdata = Input(-1).template data<T, Context>();
    for (int i = 0; i < OutputSize(); i++) {
        x_concat_dim = 1;
        if (Output(i)->name() != "ignore") {
            auto* dXdata = Output(i)->template mutable_data<T, Context>();
            TIndex count = Output(i)->count();
            kernel::ConcatGrad<T, Context>(
                count, outer_dim, inner_dim,
                    x_concat_dim, y_concat_dim,
                        concat_offset, dYdata, dXdata);
        }
        concat_offset += x_concat_dim;
    }
}

template <class Context>
void StackGradientOp<Context>::RunOnDevice() {
    if (Input(-1).name() == "ignore") return;
    while (axis < 0) axis += Input(-1).ndim();
    concat_dims = Input(-1).dims();
    y_concat_dim = concat_dims[axis];
    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis);
    concat_offset = 0;
    for (int i = 0; i < OutputSize(); i++)
        Output(i)->ReshapeLike(Input(i));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(StackGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(StackGradient);
#endif
OPERATOR_SCHEMA(StackGradient).NumInputs(2, INT_MAX).NumOutputs(1, INT_MAX);

class GetStackGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetStackGradient);
    vector<OperatorDef> MakeDefs() override {
        vector<string> inputs, outputs;
        for (int i = 0; i < def.input_size(); i++) {
            inputs.push_back(def.input(i));
            outputs.push_back(GI(i));
        }
        inputs.push_back(GO(0));
        return SingleDef(def.type() + "Gradient", "", inputs, outputs);
    }
};
REGISTER_GRADIENT(Stack, GetStackGradient);

}    // namespace dragon