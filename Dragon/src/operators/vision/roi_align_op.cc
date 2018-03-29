#include "operators/vision/roi_align_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void ROIAlignOp<Context>::RunWithType() {
    kernel::ROIAlign<T, Context>(spatial_scale,
                                pool_h, pool_w,
                                sampling_ratio,
                                     &Input(0),
                                     &Input(1),
                                    Output(0));
}

template <class Context>
void ROIAlignOp<Context>::RunOnDevice() {
    Output(0)->Reshape(vector<TIndex>({ Input(1).dim(0),
                                        Input(0).dim(1),
                                        pool_h, pool_w }));

    if (Input(0).template IsType<float>()) return RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(ROIAlign);
#ifdef WITH_CUDA
DEPLOY_CUDA(ROIAlign);
#endif
OPERATOR_SCHEMA(ROIAlign).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void ROIAlignGradientOp<Context>::RunWithType() {
    kernel::ROIAlignGrad<T, Context>(spatial_scale,
                                    pool_h, pool_w,
                                    sampling_ratio,
                                        &Input(-1),
                                         &Input(1),
                                        Output(0));
}

template <class Context>
void ROIAlignGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (Input(0).template IsType<float>()) return RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
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