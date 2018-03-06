#include "operators/vision/nn_resize_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void NNResizeOp<Context>::RunWithType() {
    if (data_format == "NCHW") {
        n = input(0).dim(0);
        c = input(0).dim(1);
        h = input(0).dim(2);
        w = input(0).dim(3);
        out_h = output(0)->dim(2);
        out_w = output(0)->dim(3);
    } else if (data_format == "NHWC") {
        n = input(0).dim(0);
        h = input(0).dim(1);
        w = input(0).dim(2);
        c = input(0).dim(3);
        out_h = output(0)->dim(1);
        out_w = output(0)->dim(2);
    }
    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    kernel::NNResize<T, Context>(output(0)->count(), n, c, h, w,
                                                   out_h, out_w,
                                                    data_format,
                                                          Xdata,
                                                          Ydata);
}

template <class Context>
void NNResizeOp<Context>::RunOnDevice() {
    vector<TIndex> dims = input(0).dims();
    if (dsize_desc.size() > 0 || dsize_value.size() > 0) {
        for (int i = 0; i < 2; i++)
            dims[spatial_axis + i] = dsize(i);
    } else if (!shape_like_desc.empty()) {
        Tensor* shape_like_tensor = ws()->GetTensor(shape_like_desc);
        for (int i = 0; i < 2; i++)
            dims[spatial_axis + i] = shape_like_tensor->dim(spatial_axis + i);
    } else {
        CHECK(fy != -1.0 && fx != -1.0)
            << "\nThe fx and fy should be set.";
        dims[spatial_axis] = int(dims[spatial_axis] * fy);
        dims[spatial_axis + 1] = int(dims[spatial_axis + 1] * fx);
    }
    output(0)->Reshape(dims);

    if (input(0).template IsType<float>()) return RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(NNResize);
#ifdef WITH_CUDA
DEPLOY_CUDA(NNResize);
#endif
OPERATOR_SCHEMA(NNResize).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void NNResizeGradientOp<Context>::RunWithType() {
    if (data_format == "NCHW") {
        n = input(0).dim(0);
        c = input(0).dim(1);
        h = input(0).dim(2);
        w = input(0).dim(3);
        out_h = input(-1).dim(2);
        out_w = input(-1).dim(3);
    } else if (data_format == "NHWC") {
        n = input(0).dim(0);
        h = input(0).dim(1);
        w = input(0).dim(2);
        c = input(0).dim(3);
        out_h = input(-1).dim(1);
        out_w = input(-1).dim(2);
    }
    auto* dYdata = input(-1).template data<T, Context>();
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    kernel::NNResizeGrad<T, Context>(input(-1).count(), n, c, h, w,
                                                      out_h, out_w,
                                                       data_format,
                                                            dYdata,
                                                            dXdata);
}

template <class Context>
void NNResizeGradientOp<Context>::RunOnDevice() {
    output(0)->ReshapeLike(input(0));
    
    if (input(0).template IsType<float>()) return RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(NNResizeGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(NNResizeGradient);
#endif
OPERATOR_SCHEMA(NNResizeGradient).NumInputs(2).NumOutputs(1);

class GetNNResizeGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetNNResizeGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(NNResize, GetNNResizeGradient);

}    // namespace dragon