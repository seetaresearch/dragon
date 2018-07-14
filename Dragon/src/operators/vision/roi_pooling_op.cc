#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/vision/roi_pooling_op.h"

namespace dragon {

template <class Context> template <typename T>
void ROIPoolingOp<Context>::RunWithType() {
    Tensor* mask = ws()->CreateTensor(
        "/mnt/" + anchor() + "/roi_pool/mask");
    mask->ReshapeLike(*Output(0));

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Rdata = Input(1).template data<T, Context>();
    auto* Mdata = mask->template mutable_data<int, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    kernel::ROIPooling<T, Context>(
        Output(0)->count(), Input(0).dim(0), Input(0).dim(1),
            Input(0).dim(2), Input(0).dim(3), pool_h, pool_w,
                Input(1).dim(0), spatial_scale, Xdata, Rdata, Mdata, Ydata);
}

template <class Context>
void ROIPoolingOp<Context>::RunOnDevice() {
    Output(0)->Reshape(vector<TIndex>(
        { Input(1).dim(0), Input(0).dim(1), pool_h, pool_w }));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(ROIPooling);
#ifdef WITH_CUDA
DEPLOY_CUDA(ROIPooling);
#endif
OPERATOR_SCHEMA(ROIPooling).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void ROIPoolingGradientOp<Context>::RunWithType() {
    Tensor* mask = ws()->GetTensor(
        "/mnt/" + anchor() + "/roi_pool/mask");

    auto* dYdata = Input(-1).template data<T, CUDAContext>();
    auto* Rdata = Input(1).template data<T, CUDAContext>();
    auto* Mdata = mask->template data<int, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();

    kernel::ROIPoolingGrad<T, Context>(
        Output(0)->count(), Output(0)->dim(0), Output(0)->dim(1),
            Output(0)->dim(2), Output(0)->dim(3), pool_h, pool_w,
                Input(1).dim(0), spatial_scale, dYdata, Rdata, Mdata, dXdata);
}

template <class Context>
void ROIPoolingGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
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