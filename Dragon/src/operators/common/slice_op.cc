#include "operators/common/slice_op.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void SliceOp<Context>::RunWithType() {
    auto* Xdata = input(0).template data<T, Context>();
    for (int i = 0; i < nout; i++){
        auto* Ydata = output(i)->template mutable_data<T, Context>();
        TIndex count = output(i)->count();
        kernel::Slice<T, Context>(count, outer_dim, inner_dim,
            x_slice_dim, y_slice_dim, slice_offset, Xdata, Ydata, &ctx());
        slice_offset += y_slice_dim;
    }
}

template <class Context>
void SliceOp<Context>::RunOnDevice() {
    CHECK_EQ(input(0).dim(axis) % nout, 0)
        << "\nselected dim is " << input(0).dim(axis)
        << ", can't be sliced by nout of " << nout;
    slice_dims = input(0).dims();
    x_slice_dim = slice_dims[axis];
    slice_dims[axis] = y_slice_dim = x_slice_dim / nout;
    outer_dim = input(0).count(0, axis);
    inner_dim = input(0).count(axis + 1);
    slice_offset = 0;
    for (int i = 0; i < nout; i++) output(i)->Reshape(slice_dims);
    if (nout == 1) {
        output(0)->Share(input(0));
        return;
    }

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(Slice);
#ifdef WITH_CUDA
DEPLOY_CUDA(Slice);
#endif
OPERATOR_SCHEMA(Slice).NumInputs(1).NumOutputs(1, INT_MAX);

template <class Context> template <typename T>
void SliceGradientOp<Context>::RunWithType() {
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    for (int i = 0; i < nout; i++){
        if (input(i + 1).name() == "ignore") continue;
        auto* dYdata = input(i + 1).template data<T, Context>();
        TIndex count = input(i + 1).count();
        kernel::SliceGrad<T, Context>(count, outer_dim, inner_dim,
            x_slice_dim, y_slice_dim, slice_offset, dYdata, dXdata, &ctx());
        slice_offset += y_slice_dim;
    }
}

template <class Context>
void SliceGradientOp<Context>::RunOnDevice() {
    slice_dims = input(0).dims();
    x_slice_dim = slice_dims[axis];
    slice_dims[axis] = y_slice_dim = x_slice_dim / nout;
    outer_dim = input(0).count(0, axis);
    inner_dim = input(0).count(axis + 1);
    slice_offset = 0;
    output(0)->ReshapeLike(input(0));
    if (nout == 1) {
        output(0)->Share(input(-1));
        return;
    }

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
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