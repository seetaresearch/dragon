#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/vision/roi_pool_op.h"

namespace dragon {

template <class Context> template <typename T>
void ROIPoolOp<Context>::RunWithType() {
    Tensor* mask = ws()->CreateTensor(mount_name(
        "roi_pool/mask"))->ReshapeLike(*Output(0));

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Rdata = Input(1).template data<float, Context>();
    auto* Mdata = mask->template mutable_data<int, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    kernel::ROIPool(
        Input(0).dim(1), Input(0).dim(2), Input(0).dim(3),
            pool_h, pool_w, Input(1).dim(0), spatial_scale,
                Xdata, Rdata, Mdata, Ydata, ctx());
}

template <class Context>
void ROIPoolOp<Context>::RunOnDevice() {
    Output(0)->Reshape({
        Input(1).dim(0),    /*!      Number of RoIs    */
        Input(0).dim(1),    /*!      Channels          */
        pool_h,             /*!      Pooled height     */
        pool_w              /*!      Pooled width      */
    });

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(ROIPool);
#ifdef WITH_CUDA
DEPLOY_CUDA(ROIPool);
#endif
OPERATOR_SCHEMA(ROIPool).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void ROIPoolGradientOp<Context>::RunWithType() {
    Tensor* mask = ws()->GetTensor(mount_name("roi_pool/mask"));

    auto* dYdata = Input(-1).template data<T, Context>();
    auto* Rdata = Input(1).template data<float, Context>();
    auto* Mdata = mask->template data<int, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();

    kernel::ROIPoolGrad(
        Output(0)->dim(0), Output(0)->dim(1),
            Output(0)->dim(2), Output(0)->dim(3),
                pool_h, pool_w, Input(1).dim(0), spatial_scale,
                    dYdata, Rdata, Mdata, dXdata, ctx());
}

template <class Context>
void ROIPoolGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(ROIPoolGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ROIPoolGradient);
#endif

OPERATOR_SCHEMA(ROIPoolGradient)
    .NumInputs(3).NumOutputs(1);

class GetROIPoolGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetROIPoolGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), GO(0) }),
            vector<string>({ GI(0) }));
    }
};

REGISTER_GRADIENT(ROIPool, GetROIPoolGradient);

}  // namespace dragon