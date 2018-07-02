#include "operators/ndarray/reduce_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void ReduceOp<Context>::SumRunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    if (axis == -1) {
        DECLARE_MULTIPLIER(multiplier, Input(0).count());
        auto* Ydata = Output(0)->template mutable_data<T, CPUContext>();
        Ydata[0] = math::Dot<T, Context>(
            Input(0).count(), multiplier, Xdata, &ctx());
    } else {
        auto* Ydata = Output(0)->template mutable_data<T, Context>();
        kernel::Sum<T, Context>(count, axis_dim, inner_dim, Xdata, Ydata);
    }
}

template <class Context> template <typename T>
void ReduceOp<Context>::MeanRunWithType() {
    SumRunWithType<T>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    T coeff = axis != -1 ? 1.0 / axis_dim : 1.0 / Input(0).count();
    math::Scal<T, Context>(Output(0)->count(), coeff, Ydata, &ctx());
}

template <class Context>
void ReduceOp<Context>::RunOnDevice() {
    if (axis != -1) {
        count = Input(0).count() / Input(0).dim(axis);
        axis_dim = Input(0).dim(axis);
    }
    inner_dim = Input(0).count(axis + 1);
    vector<TIndex> dims = Input(0).dims();
    if (!keep_dims) {
        if (axis != -1) dims.erase(dims.begin() + axis);
        else dims = vector<TIndex>();
    } else {
        if (axis != -1) dims[axis] = 1;
        else dims = vector<TIndex>(Input(0).ndim(), 1);
    }
    Output(0)->Reshape(dims);

    if (XIsType(Input(0), float)) {
        if (operation == "SUM") SumRunWithType<float>();
        else if (operation == "MEAN") MeanRunWithType<float>();
        else LOG(FATAL) << "Unknown operation: [" << operation << "].";
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(Reduce);
#ifdef WITH_CUDA
DEPLOY_CUDA(Reduce);
#endif
OPERATOR_SCHEMA(Reduce).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void ReduceGradientOp<Context>::SumRunWithType() {
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    if (axis == -1) {
        auto* dYdata = Input(-1).template data<T, CPUContext>();
        math::Set<T, Context>(Output(0)->count(), dYdata[0], dXdata);
    } else {
        auto* dYdata = Input(-1).template data<T, Context>();
        kernel::SumGrad<T, Context>(count,
            axis_dim, inner_dim, 1.0, dYdata, dXdata);
    }
}

template <class Context> template <typename T>
void ReduceGradientOp<Context>::MeanRunWithType() {
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    if (axis == -1) {
        auto* dYdata = Input(-1).template data<T, CPUContext>();
        math::Set<T, Context>(Output(0)->count(),
            dYdata[0] / Input(0).count(), dXdata);
    } else {
        auto* dYdata = Input(-1).template data<T, Context>();
        kernel::SumGrad<T, Context>(count,
            axis_dim, inner_dim, 1.0 / axis_dim, dYdata, dXdata);
    }
}

template <class Context>
void ReduceGradientOp<Context>::RunOnDevice() {
    if (axis != -1) {
        count = Input(0).count() / Input(0).dim(axis);
        axis_dim = Input(0).dim(axis);
    }
    inner_dim = Input(0).count(axis + 1);
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) {
        if (operation == "SUM") SumRunWithType<float>();
        else if (operation == "MEAN") MeanRunWithType<float>();
        else LOG(FATAL) << "Unknown operation: [" << operation << "].";
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(ReduceGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ReduceGradient);
#endif
OPERATOR_SCHEMA(ReduceGradient).NumInputs(2).NumOutputs(1);

class GetReduceGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetReduceGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(Reduce, GetReduceGradient);

}    // namespace dragon