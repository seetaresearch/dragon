#include "operators/misc/image_data_op.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename Tx, typename Ty>
void ImageDataOp<Context>::RunWithType() {
    auto* Xdata = input(0).template data<Tx, Context>();
    auto* Mdata = mean.count() > 0 ? mean.template data<float, Context>() : nullptr;
    auto* Sdata = std.count() > 0 ? std.template data<float, Context>() : nullptr;
    auto* Ydata = output(0)->template mutable_data<Ty, Context>();
    kernel::ImageData<Tx, Ty, Context>(output(0)->count(),
                                               n, c, h, w,
                                             Mdata, Sdata,
                                              data_format,
                                                    Xdata,
                                                    Ydata);
}

template <class Context>
void ImageDataOp<Context>::RunOnDevice() {
    n = input(0).dim(0);
    c = input(0).dim(3);
    h = input(0).dim(1);
    w = input(0).dim(2);

    if (data_format == "NCHW") {
        output(0)->Reshape(vector<TIndex>({ n, c, h, w }));
    } else if (data_format == "NHWC") {
        output(0)->ReshapeLike(input(0));
    } else LOG(FATAL) << "Unknown data format: " << data_format;

    if (input(0).template IsType<float>()) {
        if (dtype == "FLOAT32") RunWithType<float, float>();
#ifdef WITH_CUDA_FP16
        else if (dtype == "FLOAT16") RunWithType<float, float16>();
#endif
        else LOG(FATAL) << "Unsupported output type: " << dtype;
    } else if (input(0).template IsType<uint8_t>()) {
        if (dtype == "FLOAT32") RunWithType<uint8_t, float>();
#ifdef WITH_CUDA_FP16
        else if (dtype == "FLOAT16") RunWithType<uint8_t, float16>();
#endif
        else LOG(FATAL) << "Unsupported output type: " << dtype;
    } 
    else { LOG(FATAL) << "Unsupported input types."; }
}

DEPLOY_CPU(ImageData);
#ifdef WITH_CUDA
DEPLOY_CUDA(ImageData);
#endif
OPERATOR_SCHEMA(ImageData).NumInputs(1).NumOutputs(1);

NO_GRADIENT(ImageData);

}    // namespace dragon