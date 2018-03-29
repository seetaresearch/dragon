#include "operators/vision/pooling_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void Pooling2dOp<Context>::MAXRunWithType() {
    mask = ws()->CreateTensor("/mnt/" + anchor() + "/max_pool_mask");
    mask->ReshapeLike(*Output(0));

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    auto* Mdata = mask->template mutable_data<int, Context>();

    kernel::MAXPooling2d<T, Context>(Output(0)->count(),
                                             n, c, h, w,
                                         pool_h, pool_w,
                         kernel_size[0], kernel_size[1],
                                   stride[0], stride[1],
                                         pad[0], pad[1],
                                            data_format,
                                                  Xdata,
                                                  Mdata,
                                                 Ydata);
}

template <class Context> template <typename T>
void Pooling2dOp<Context>::AVGRunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    kernel::AVGPooling2d<T, Context>(Output(0)->count(),
                                             n, c, h, w,
                                         pool_h, pool_w,
                         kernel_size[0], kernel_size[1],
                                   stride[0], stride[1],
                                         pad[0], pad[1],
                                            data_format,
                                                  Xdata,
                                                 Ydata);
}

template <class Context>
void Pooling2dOp<Context>::Reshape() {
    if (data_format == "NCHW") {
        n = Input(0).dim(0);
        c = Input(0).dim(1);
        h = Input(0).dim(2);
        w = Input(0).dim(3);
        if (global_pooling) {
            for (int i = 0; i < 2; i++)
                kernel_size[i] = Input(0).dim(i + 2);
        }
        if (padding == "SAME") {
            for (int i = 0; i < 2; i++) {
                TIndex input_size = Input(0).dim(i + 2);
                TIndex output_size = (input_size + stride[i] - 1) / (float)stride[i];
                TIndex padding_needed = std::max(TIndex(0), (output_size - 1) * stride[i] + kernel_size[i] - input_size);
                TIndex pad_l = padding_needed / 2;
                TIndex pad_r = padding_needed - pad_l;
                pad[i] = pad_l;
            }
        }
    } else if (data_format == "NHWC") {
        n = Input(0).dim(0);
        h = Input(0).dim(1);
        w = Input(0).dim(2);
        c = Input(0).dim(3);
        if (global_pooling) {
            for (int i = 0; i < 2; i++)
                kernel_size[i] = Input(0).dim(i + 1);
        }
        if (padding == "SAME") {
            for (int i = 0; i < 2; i++) {
                TIndex input_size = Input(0).dim(i + 1);
                TIndex output_size = (input_size + stride[i] - 1) / (float)stride[i];
                TIndex padding_needed = std::max(TIndex(0), (output_size - 1) * stride[i] + kernel_size[i] - input_size);
                TIndex pad_l = padding_needed / 2;
                TIndex pad_r = padding_needed - pad_l;
                pad[i] = pad_l;
            }
        }
    } else LOG(FATAL) << "Unknown data format: " << data_format;

    if (padding != "SAME") {
        //  case 1: infer output shape with symmetry pad size
        pool_h = ceil((h + 2 * pad[0] - kernel_size[0]) / (float)stride[0]) + 1;
        pool_w = ceil((w + 2 * pad[1] - kernel_size[1]) / (float)stride[1]) + 1;
        if ((pool_h - 1) * stride[0] >= (h + pad[0])) pool_h--;
        if ((pool_w - 1) * stride[1] >= (w + pad[1])) pool_w--;
    } else {
        //  case 2: infer output shape with adaptive pad size
        pool_h = (h + stride[0] - 1) / (float)stride[0];
        pool_w = (w + stride[1] - 1) / (float)stride[1];
    }
    if (data_format == "NCHW") Output(0)->Reshape(vector<TIndex>({ n, c, pool_h, pool_w }));
    else if (data_format == "NHWC") Output(0)->Reshape(vector<TIndex>({ n, pool_h, pool_w, c }));
}

template <class Context>
void Pooling2dOp<Context>::RunOnDevice() {
    Reshape();

    if (mode == "MAX") {
        if (Input(0).template IsType<float>()) MAXRunWithType<float>();
        else LOG(FATAL) << "Unsupported input types.";
    }  else if (mode == "AVG") {
        if (Input(0).template IsType<float>()) AVGRunWithType<float>();
        else LOG(FATAL) << "Unsupported input types.";
    } else { 
        LOG(FATAL) << "Unsupported pooling mode: " << mode;
    }
}

DEPLOY_CPU(Pooling2d);
#ifdef WITH_CUDA
DEPLOY_CUDA(Pooling2d);
#endif
OPERATOR_SCHEMA(Pooling2d).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void Pooling2dGradientOp<Context>::MAXRunWithType() {
    mask = ws()->GetTensor("/mnt/" + anchor() + "/max_pool_mask");

    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* Mdata = mask->template data<int, Context>();

    kernel::MAXPooling2dGrad<T, Context>(Output(0)->count(),
                                                 n, c, h, w,
                                             pool_h, pool_w,
                             kernel_size[0], kernel_size[1],
                                       stride[0], stride[1],
                                             pad[0], pad[1],
                                                data_format,
                                                     dYdata,
                                                      Mdata,
                                                    dXdata);
}

template <class Context> template <typename T>
void Pooling2dGradientOp<Context>::AVGRunWithType() {
    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();

    kernel::AVGPooling2dGrad<T, Context>(Output(0)->count(),
                                                 n, c, h, w,
                                             pool_h, pool_w,
                             kernel_size[0], kernel_size[1],
                                       stride[0], stride[1],
                                             pad[0], pad[1],
                                                data_format,
                                                     dYdata,
                                                    dXdata);
}

template <class Context>
void Pooling2dGradientOp<Context>::Reshape() {
   if (data_format == "NCHW") {
        n = Input(0).dim(0);
        c = Input(0).dim(1);
        h = Input(0).dim(2);
        w = Input(0).dim(3);
        if (global_pooling) {
            for (int i = 0; i < 2; i++)
                kernel_size[i] = Input(0).dim(i + 2);
        }
        if (padding == "SAME") {
            for (int i = 0; i < 2; i++) {
                TIndex input_size = Input(0).dim(i + 2);
                TIndex output_size = (input_size + stride[i] - 1) / (float)stride[i];
                TIndex padding_needed = std::max(TIndex(0), (output_size - 1) * stride[i] + kernel_size[i] - input_size);
                TIndex pad_l = padding_needed / 2;
                TIndex pad_r = padding_needed - pad_l;
                pad[i] = pad_l;
            }
        }
    } else if (data_format == "NHWC") {
        n = Input(0).dim(0);
        h = Input(0).dim(1);
        w = Input(0).dim(2);
        c = Input(0).dim(3);
        if (global_pooling) {
            for (int i = 0; i < 2; i++)
                kernel_size[i] = Input(0).dim(i + 1);
        }
        if (padding == "SAME") {
            for (int i = 0; i < 2; i++) {
                TIndex input_size = Input(0).dim(i + 1);
                TIndex output_size = (input_size + stride[i] - 1) / (float)stride[i];
                TIndex padding_needed = std::max(TIndex(0), (output_size - 1) * stride[i] + kernel_size[i] - input_size);
                TIndex pad_l = padding_needed / 2;
                TIndex pad_r = padding_needed - pad_l;
                pad[i] = pad_l;
            }
        }
    } else LOG(FATAL) << "Unknown data format: " << data_format;

    if (padding != "SAME") {
        //  case 1: infer output shape with symmetry pad size
        pool_h = ceil((h + 2 * pad[0] - kernel_size[0]) / (float)stride[0]) + 1;
        pool_w = ceil((w + 2 * pad[1] - kernel_size[1]) / (float)stride[1]) + 1;
        if ((pool_h - 1) * stride[0] >= (h + pad[0])) pool_h--;
        if ((pool_w - 1) * stride[1] >= (w + pad[1])) pool_w--;
    } else {
        //  case 2: infer output shape with adaptive pad size
        pool_h = (h + stride[0] - 1) / (float)stride[0];
        pool_w = (w + stride[1] - 1) / (float)stride[1];
    }
    Output(0)->ReshapeLike(Input(0));
}

template <class Context>
void Pooling2dGradientOp<Context>::RunOnDevice() {
    Reshape();

   if (mode == "MAX") {
        if (Input(0).template IsType<float>()) MAXRunWithType<float>();
        else LOG(FATAL) << "Unsupported input types.";
    }  else if (mode == "AVG") {
        if (Input(0).template IsType<float>()) AVGRunWithType<float>();
        else LOG(FATAL) << "Unsupported input types.";
    } else { 
        LOG(FATAL) << "Unsupported pooling mode: " << mode;
    }
}

DEPLOY_CPU(Pooling2dGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(Pooling2dGradient);
#endif
OPERATOR_SCHEMA(Pooling2dGradient).NumInputs(3).NumOutputs(1);

class GetPooling2dGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetPooling2dGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), O(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(Pooling2d, GetPooling2dGradient);

}    // namespace dragon