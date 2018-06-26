#include "operators/vision/roi_align_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void ROIAlignOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Rdata = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    kernel::ROIAlign<T, Context>(
        Output(0)->count(), Input(0).dim(0), Input(0).dim(1),
            Input(0).dim(2), Input(0).dim(3), pool_h, pool_w,
                Input(1).dim(0), spatial_scale, sampling_ratio, Xdata, Rdata, Ydata);
}

template <class Context>
void ROIAlignOp<Context>::RunOnDevice() {
    Output(0)->Reshape(vector<TIndex>(
        { Input(1).dim(0), Input(0).dim(1), pool_h, pool_w }));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(ROIAlign);
#ifdef WITH_CUDA
DEPLOY_CUDA(ROIAlign);
#endif
OPERATOR_SCHEMA(ROIAlign).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void ROIAlignGradientOp<Context>::RunWithType() {
    auto* dYdata = Input(-1).template data<T, CUDAContext>();
    auto* Rdata = Input(1).template data<T, CUDAContext>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();

    math::Set<T, CUDAContext>(Output(0)->count(), 0, dXdata);

    kernel::ROIAlignGrad<T, Context>(
        Input(-1).count(), Output(0)->dim(0), Output(0)->dim(1),
            Output(0)->dim(2), Output(0)->dim(3), pool_h, pool_w,
                Input(1).dim(0), spatial_scale, sampling_ratio, dYdata, Rdata, dXdata);
}

template <class Context>
void ROIAlignGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(ROIAlignGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ROIAlignGradient);
#endif
OPERATOR_SCHEMA(ROIAlignGradient).NumInputs(3).NumOutputs(1);

class GetROIAlignGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetROIAlignGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(ROIAlign, GetROIAlignGradient);

}    // namespace dragon