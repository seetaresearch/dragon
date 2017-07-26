#include "operators/vision/pooling_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void PoolingOp<Context>::MaxRunWithType() {
    mask = ws()->CreateTensor("_t_" + anchor() + "_pool_mask");
    mask->ReshapeLike(*output(0));

    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    auto* Mdata = mask->template mutable_data<int, Context>();

    kernel::MAXPooling<T, Context>(output(0)->count(), 
                         num, channels, height, width, 
                              pool_height, pool_width,
                       kernel_size[0], kernel_size[1], 
                                 stride[0], stride[1], 
                                       pad[0], pad[1],
                                                Xdata, 
                                                Mdata, 
                                               Ydata);
}

template <class Context> template <typename T>
void PoolingOp<Context>::AvgRunWithType() {
    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();

    kernel::AVEPooling<T, Context>(output(0)->count(), 
                         num, channels, height, width,
                              pool_height, pool_width, 
                       kernel_size[0], kernel_size[1], 
                                 stride[0], stride[1], 
                                       pad[0], pad[1],
                                                Xdata, 
                                               Ydata);
}

template <class Context>
void PoolingOp<Context>::Reshape() {
    CHECK_GE(input(0).ndim(), 3) 
        << "input ndim must >= 3 (channels, height, width).";

    num = input(0).ndim() == 3 ? 1 : input(0).dim(0);
    channels = input(0).ndim() == 3 ? input(0).dim(0) : input(0).dim(1);
    height = input(0).ndim() == 3 ? input(0).dim(1) : input(0).dim(2);
    width = input(0).ndim() == 3 ? input(0).dim(2) : input(0).dim(3);

    pool_height = ceil((height + 2 * pad[0] - kernel_size[0]) / (float)stride[0]) + 1;
    pool_width = ceil((width + 2 * pad[1] - kernel_size[1]) / (float)stride[1]) + 1;
    if (pad.size()) {
        if ((pool_height - 1) * stride[0] >= (height + pad[0])) pool_height--;
        if ((pool_width - 1) * stride[1] >= (width + pad[1])) pool_width--;
    }

    vector<TIndex> top_shape({ num, channels, pool_height, pool_width });
    if (input(0).ndim() == 3) top_shape.erase(top_shape.begin());
    output(0)->Reshape(top_shape);
}

template <class Context>
void PoolingOp<Context>::RunOnDevice() {
    Reshape();

    if (mode == MAX_POOLING) {
        if (input(0).template IsType<float>()) MaxRunWithType<float>();
        else LOG(FATAL) << "unsupported input types.";
    } 
    else if (mode == AVG_POOLING) {
        if (input(0).template IsType<float>()) AvgRunWithType<float>();
        else LOG(FATAL) << "unsupported input types.";
    }
    else { 
        LOG(FATAL) << "unsupported pooling mode."; 
    }
}

DEPLOY_CPU(Pooling);
#ifdef WITH_CUDA
DEPLOY_CUDA(Pooling);
#endif
OPERATOR_SCHEMA(Pooling).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void PoolingGradientOp<Context>::MaxRunWithType() {
    mask = ws()->GetTensor("_t_" + anchor() + "_pool_mask");

    auto* dYdata = input(-1).template data<T, Context>();
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    auto* Mdata = mask->template data<int, Context>();

    kernel::MAXPoolingGrad<T, Context>(output(0)->count(),
                             num, channels, height, width,
                                  pool_height, pool_width, 
                           kernel_size[0], kernel_size[1],
                                     stride[0], stride[1],
                                           pad[0], pad[1], 
                                                   dYdata, 
                                                    Mdata, 
                                                  dXdata);
}

template <class Context> template <typename T>
void PoolingGradientOp<Context>::AvgRunWithType() {
    auto* dYdata = input(-1).template data<T, Context>();
    auto* dXdata = output(0)->template mutable_data<T, Context>();

    kernel::AVEPoolingGrad<T, Context>(output(0)->count(),
                             num, channels, height, width,
                                  pool_height, pool_width, 
                           kernel_size[0], kernel_size[1],
                                     stride[0], stride[1],
                                           pad[0], pad[1],
                                                   dYdata, 
                                                  dXdata);
}

template <class Context>
void PoolingGradientOp<Context>::Reshape() {
    num = input(0).ndim() == 3 ? 1 : input(0).dim(0);
    channels = input(0).ndim() == 3 ? input(0).dim(0) : input(0).dim(1);
    height = input(0).ndim() == 3 ? input(0).dim(1) : input(0).dim(2);
    width = input(0).ndim() == 3 ? input(0).dim(2) : input(0).dim(3);
    pool_height = ceil((height + 2 * pad[0] - kernel_size[0]) / (float)stride[0]) + 1;
    pool_width = ceil((width + 2 * pad[1] - kernel_size[1]) / (float)stride[1]) + 1;
    if (pad.size()) {
        if ((pool_height - 1) * stride[0] >= (height + pad[0])) pool_height--;
        if ((pool_width - 1)* stride[1] >= (width + pad[1])) pool_width--;
    }
    output(0)->ReshapeLike(input(0));
}

template <class Context>
void PoolingGradientOp<Context>::RunOnDevice() {
    Reshape();

    if (mode == MAX_POOLING) {
        if (input(0).template IsType<float>()) MaxRunWithType<float>();
        else LOG(FATAL) << "unsupported input types.";
    }
    else if (mode == AVG_POOLING) {
        if (input(0).template IsType<float>()) AvgRunWithType<float>();
        else LOG(FATAL) << "unsupported input types.";
    } 
    else { 
        LOG(FATAL) << "unsupported pooling mode."; 
    }
}

template <class Context>
void PoolingGradientOp<Context>::ShareBeforeRun() {
    Tensor* dX = ws()->GetBuffer();
    if (dX != nullptr) output(0)->Replace(*dX);
}

template <class Context>
void PoolingGradientOp<Context>::ClearAfterRun() {
    Tensor* dY = &input(-1);
    ws()->ReleaseBuffer(dY);
}

DEPLOY_CPU(PoolingGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(PoolingGradient);
#endif
OPERATOR_SCHEMA(PoolingGradient).NumInputs(3).NumOutputs(1);

class GetPoolingGradient final : public GradientMakerBase {
public:
    GRADIENT_MAKER_CTOR(GetPoolingGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), O(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(Pooling, GetPoolingGradient);

}    // namespace dragon