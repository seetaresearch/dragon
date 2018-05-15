#include "operators/misc/image_data_op.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename Tx, typename Ty>
void ImageDataOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<Tx, Context>();
    auto* Mdata = mean.count() > 0 ? mean.template data<float, Context>() : nullptr;
    auto* Sdata = std.count() > 0 ? std.template data<float, Context>() : nullptr;
    auto* Ydata = Output(0)->template mutable_data<Ty, Context>();
    kernel::ImageData<Tx, Ty, Context>(Output(0)->count(),
                                               n, c, h, w,
                                             Mdata, Sdata,
                                              data_format,
                                                    Xdata,
                                                    Ydata);
}

template <class Context>
void ImageDataOp<Context>::RunOnDevice() {
    n = Input(0).dim(0);
    c = Input(0).dim(3);
    h = Input(0).dim(1);
    w = Input(0).dim(2);

    if (data_format == "NCHW") {
        Output(0)->Reshape(vector<TIndex>({ n, c, h, w }));
    } else if (data_format == "NHWC") {
        Output(0)->ReshapeLike(Input(0));
    } else LOG(FATAL) << "Unknown data format: " << data_format;

    if (XIsType(Input(0), float)) {
        if (dtype == "FLOAT32") RunWithType<float, float>();
        else if (dtype == "FLOAT16") RunWithType<float, float16>();
        else LOG(FATAL) << "Unsupported output type: " << dtype;
    } else if (XIsType(Input(0), uint8_t)) {
        if (dtype == "FLOAT32") RunWithType<uint8_t, float>();
        else if (dtype == "FLOAT16") RunWithType<uint8_t, float16>();
        else LOG(FATAL) << "Unsupported output type: " << dtype;
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "uint8" });
}

DEPLOY_CPU(ImageData);
#ifdef WITH_CUDA
DEPLOY_CUDA(ImageData);
#endif
OPERATOR_SCHEMA(ImageData).NumInputs(1).NumOutputs(1);

NO_GRADIENT(ImageData);

}    // namespace dragon