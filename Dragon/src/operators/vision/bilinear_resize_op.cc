#include "operators/vision/bilinear_resize_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void BilinearResizeOp<Context>::RunWithType() {
    if (data_format == "NCHW") {
        n = dims[0];
        c = dims[1];
        h = input(0).dim(2);
        w = input(0).dim(3);
        out_h = dims[2];
        out_w = dims[3];
    } else if (data_format == "NHWC") {
        n = dims[0];
        h = input(0).dim(1);
        w = input(0).dim(2);
        out_h = dims[1];
        out_w = dims[2];
        c = dims[3];
    }
    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    kernel::BilinearResize<T, Context>(output(0)->count(), n, c, h, w,
                                                         out_h, out_w,
                                                          data_format,
                                                                Xdata,
                                                                Ydata);
}

template <class Context>
void BilinearResizeOp<Context>::RunOnDevice() {
    dims = input(0).dims();
    if (dsize_desc.size() > 0) {
        CHECK_EQ(dsize_desc.size(), 2) << "\nThe dsize should be a scalar with 2 elements.";
        for (int i = 0; i < 2; i++) {
            Tensor* dsize = ws()->GetTensor(dsize_desc[i]);
            CHECK(dsize->IsType<int>()) << "\nThe type of dsize should be int32.";
            dims[spatial_axis + i] = dsize->template data<int, CPUContext>()[0];
        }
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

DEPLOY_CPU(BilinearResize);
#ifdef WITH_CUDA
DEPLOY_CUDA(BilinearResize);
#endif
OPERATOR_SCHEMA(BilinearResize).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void BilinearResizeGradientOp<Context>::RunWithType() {
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
    kernel::BilinearResizeGrad<T, Context>(input(-1).count(), n, c, h, w,
                                                            out_h, out_w,
                                                             data_format,
                                                                  dYdata,
                                                                  dXdata);
}

template <class Context>
void BilinearResizeGradientOp<Context>::RunOnDevice() {
    output(0)->ReshapeLike(input(0));
    
    if (input(0).template IsType<float>()) return RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
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

}    // namespace dragon