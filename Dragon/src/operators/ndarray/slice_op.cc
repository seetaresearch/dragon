#include "operators/ndarray/slice_op.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void SliceOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    for (int i = 0; i < nout; i++) {
        auto* Ydata = Output(i)->template mutable_data<T, Context>();
        TIndex count = Output(i)->count();
        kernel::Slice<T, Context>(count, outer_dim, inner_dim,
            x_slice_dim, y_slice_dim, slice_offset, Xdata, Ydata, &ctx());
        slice_offset += y_slice_dim;
    }
}

template <class Context>
void SliceOp<Context>::RunOnDevice() {
    CHECK_EQ(Input(0).dim(axis) % nout, 0)
        << "\nSelected dim is " << Input(0).dim(axis)
        << ", can't be sliced by nout of " << nout;
    slice_dims = Input(0).dims();
    x_slice_dim = slice_dims[axis];
    slice_dims[axis] = y_slice_dim = x_slice_dim / nout;
    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
    slice_offset = 0;
    for (int i = 0; i < nout; i++) Output(i)->Reshape(slice_dims);

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(Slice);
#ifdef WITH_CUDA
DEPLOY_CUDA(Slice);
#endif
OPERATOR_SCHEMA(Slice).NumInputs(1).NumOutputs(1, INT_MAX);

template <class Context> template <typename T>
void SliceGradientOp<Context>::RunWithType() {
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    for (int i = 0; i < nout; i++) {
        if (Input(i + 1).name() == "ignore") continue;
        auto* dYdata = Input(i + 1).template data<T, Context>();
        TIndex count = Input(i + 1).count();
        kernel::SliceGrad<T, Context>(count, outer_dim, inner_dim,
            x_slice_dim, y_slice_dim, slice_offset, dYdata, dXdata, &ctx());
        slice_offset += y_slice_dim;
    }
}

template <class Context>
void SliceGradientOp<Context>::RunOnDevice() {
    slice_dims = Input(0).dims();
    x_slice_dim = slice_dims[axis];
    slice_dims[axis] = y_slice_dim = x_slice_dim / nout;
    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
    slice_offset = 0;
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(SliceGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SliceGradient);
#endif
OPERATOR_SCHEMA(SliceGradient).NumInputs(2, INT_MAX).NumOutputs(1);

class GetSliceGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetSliceGradient);
    vector<OperatorDef> MakeDefs() override {
        vector<string> inputs(1, I(0));
        for (int i = 0; i < def.output_size(); i++) inputs.push_back(GO(i));
        return SingleDef(def.type() + "Gradient", "", 
                         inputs, vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(Slice, GetSliceGradient);

}    // namespace dragon