#include "operators/ndarray/concat_op.h"
#include "core/workspace.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void ConcatOp<Context>::RunWithType() {
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    for (int i = 0; i < InputSize(); i++) {
        auto* Xdata = Input(i).template data<T, Context>();
        TIndex count = Input(i).count();
        x_concat_dim = Input(i).dim(axis);
        kernel::Concat<T, Context>(count,
                               outer_dim,
                               inner_dim,
                            x_concat_dim,
                            y_concat_dim,
                           concat_offset,
                                   Xdata,
                                   Ydata,
                                 &ctx());
        concat_offset += x_concat_dim;
    }
}

template <class Context>
void ConcatOp<Context>::RunOnDevice() {
    concat_dims = Input(0).dims();
    for (int i = 1; i < InputSize(); i++) {
        CHECK_EQ(concat_dims.size(), Input(i).ndim())
            << "\nAll inputs should have the same ndim.";
        for (int j = 0; j < concat_dims.size(); j++) {
            if (j == axis) continue;
            CHECK_EQ(concat_dims[j], Input(i).dim(j))
                << "\nAll inputs should have the same dimensions"
                << ", except the concat axis.";
        }
        concat_dims[axis] += Input(i).dim(axis);
    }
    y_concat_dim = concat_dims[axis];
    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
    concat_offset = 0;
    Output(0)->Reshape(concat_dims);

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(Concat);
#ifdef WITH_CUDA
DEPLOY_CUDA(Concat);
#endif
OPERATOR_SCHEMA(Concat).NumInputs(1, INT_MAX).NumOutputs(1);

template <class Context> template    <typename T>
void ConcatGradientOp<Context>::RunWithType() {
    auto* dYdata = Input(-1).template data<T, Context>();
    for (int i = 0; i < OutputSize(); i++) {
        x_concat_dim = Input(i).dim(axis);
        if (Output(i)->name() != "ignore") {
            auto* dXdata = Output(i)->template mutable_data<T, Context>();
            TIndex count = Output(i)->count();
            kernel::ConcatGrad<T, Context>(count,
                                       outer_dim,
                                       inner_dim,
                                    x_concat_dim,
                                    y_concat_dim,
                                   concat_offset,
                                          dYdata,
                                          dXdata,
                                         &ctx());
        }
        concat_offset += x_concat_dim;
    }
}

template <class Context>
void ConcatGradientOp<Context>::RunOnDevice() {
    if (Input(-1).name() == "ignore") return;
    concat_dims = Input(-1).dims();
    y_concat_dim = concat_dims[axis];
    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
    concat_offset = 0;
    for (int i = 0; i < OutputSize(); i++) 
        Output(i)->ReshapeLike(Input(i));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(ConcatGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ConcatGradient);
#endif
OPERATOR_SCHEMA(ConcatGradient).NumInputs(2, INT_MAX).NumOutputs(1, INT_MAX);

class GetConcatGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetConcatGradient);
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
REGISTER_GRADIENT(Concat, GetConcatGradient);

}    // namespace dragon