#include "operators/vision/roi_pooling_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void ROIPoolingOp<Context>::RunWithType() {
    kernel::ROIPooling<T, Context>(spatial_scale, 
                                  pool_h, pool_w,
                                       &Input(0),
                                       &Input(1),
                                            mask,
                                      Output(0));
}

template <class Context>
void ROIPoolingOp<Context>::RunOnDevice() {
    mask = ws()->CreateTensor("/mnt/" + Anchor() + "/roi_pool/mask");

    vector<TIndex> dims({Input(1).dim(0), Input(0).dim(1), pool_h, pool_w});
    Output(0)->Reshape(dims);
    mask->Reshape(dims);

    if (Input(0).template IsType<float>()) return RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(ROIPooling);
#ifdef WITH_CUDA
DEPLOY_CUDA(ROIPooling);
#endif
OPERATOR_SCHEMA(ROIPooling).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void ROIPoolingGradientOp<Context>::RunWithType() {
    kernel::ROIPoolingGrad<T, Context>(spatial_scale,
                                      pool_h, pool_w,
                                          &Input(-1),
                                           &Input(1),
                                                mask,
                                          Output(0));
}

template <class Context>
void ROIPoolingGradientOp<Context>::RunOnDevice() {
    mask = ws()->GetTensor("/mnt/" + Anchor() + "/roi_pool/mask");

    Output(0)->ReshapeLike(Input(0));

    if (Input(0).template IsType<float>()) return RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(ROIPoolingGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ROIPoolingGradient);
#endif
OPERATOR_SCHEMA(ROIPoolingGradient).NumInputs(3).NumOutputs(1);

class GetROIPoolingGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetROIPoolingGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(ROIPooling, GetROIPoolingGradient);

}    // namespace dragon