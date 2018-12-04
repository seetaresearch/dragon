#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/vision/bilinear_resize_op.h"

namespace dragon {

template <class Context> template <typename T>
void BilinearResizeOp<Context>::RunWithType() {
    if (data_format == "NCHW") {
        n = Input(0).dim(0);
        c = Input(0).dim(1);
        h = Input(0).dim(2);
        w = Input(0).dim(3);
        out_h = Output(0)->dim(2);
        out_w = Output(0)->dim(3);
    } else if (data_format == "NHWC") {
        n = Input(0).dim(0);
        h = Input(0).dim(1);
        w = Input(0).dim(2);
        c = Input(0).dim(3);
        out_h = Output(0)->dim(1);
        out_w = Output(0)->dim(2);
    }
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    kernel::BilinearResize<T, Context>(Output(0)->count(),
        n, c, h, w, out_h, out_w, data_format, Xdata, Ydata, ctx());
}

template <class Context>
void BilinearResizeOp<Context>::RunOnDevice() {
    vector<TIndex> dims = Input(0).dims();
    if (dsize_desc.size() > 0 || dsize_value.size() > 0) {
        for (int i = 0; i < 2; i++)
            dims[spatial_axis + i] = dsize(i);
    } else if (!shape_like_desc.empty()) {
        Tensor* shape_like_tensor = ws()->GetTensor(shape_like_desc);
        for (int i = 0; i < 2; i++)
            dims[spatial_axis + i] = shape_like_tensor->dim(spatial_axis + i);
    } else {
        CHECK(fy != -1.f && fx != -1.f)
            << "\nThe fx and fy should be set.";
        dims[spatial_axis] = int(dims[spatial_axis] * fy);
        dims[spatial_axis + 1] = int(dims[spatial_axis + 1] * fx);
    }
    Output(0)->Reshape(dims);
    
    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(BilinearResize);
#ifdef WITH_CUDA
DEPLOY_CUDA(BilinearResize);
#endif
OPERATOR_SCHEMA(BilinearResize).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void BilinearResizeGradientOp<Context>::RunWithType() {
    if (data_format == "NCHW") {
        n = Input(0).dim(0);
        c = Input(0).dim(1);
        h = Input(0).dim(2);
        w = Input(0).dim(3);
        out_h = Input(-1).dim(2);
        out_w = Input(-1).dim(3);
    } else if (data_format == "NHWC") {
        n = Input(0).dim(0);
        h = Input(0).dim(1);
        w = Input(0).dim(2);
        c = Input(0).dim(3);
        out_h = Input(-1).dim(1);
        out_w = Input(-1).dim(2);
    }
    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();

    math::Set<T, Context>(Output(0)->count(), 0, dXdata, ctx());

    kernel::BilinearResizeGrad<T, Context>(Input(-1).count(),
        n, c, h, w, out_h, out_w, data_format, dYdata, dXdata, ctx());
}

template <class Context>
void BilinearResizeGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));
    
    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(BilinearResizeGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(BilinearResizeGradient);
#endif
OPERATOR_SCHEMA(BilinearResizeGradient).NumInputs(2).NumOutputs(1);

class GetBilinearResizeGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetBilinearResizeGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(BilinearResize, GetBilinearResizeGradient);

}  // namespace dragon