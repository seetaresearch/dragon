#include "utils/op_kernel.h"
#include "operators/misc/image_data_op.h"

namespace dragon {

template <class Context>
template <typename Tx, typename Ty>
void ImageDataOp<Context>::RunImpl() {
    auto* mean = mean_.count() == 0 ? nullptr :
                 mean_.template data<float, Context>();
    auto* std  = std_.count() == 0 ? nullptr :
                 std_.template data<float, Context>();

    auto* x = X(0).template data<Tx, Context>();
    auto* y = Y(0)->template mutable_data<Ty, Context>();

    kernel::ImageData(
        n_, c_, h_, w_, data_format(),
        mean, std, x, y, ctx()
    );
}

template <class Context>
void ImageDataOp<Context>::RunOnDevice() {
    n_ = X(0).dim(0), c_ = X(0).dim(3);
    h_ = X(0).dim(1), w_ = X(0).dim(2);

    if (data_format() == "NCHW") {
        Y(0)->Reshape({ n_, c_, h_, w_ });
    } else if (data_format() == "NHWC") {
        Y(0)->ReshapeLike(X(0));
    } else {
        LOG(FATAL) << "Unknown DataFormat: " << data_format();
    }

    if (XIsType(X(0), float)) {
        if (dtype() == "float32") {
            RunImpl<float, float>();
        } else if (dtype() == "float16") {
            RunImpl<float, float16>();
        } else {
            LOG(FATAL) << DTypeString(dtype(),
                { "float32", "float16" }
            );
        }
    } else if (XIsType(X(0), uint8_t)) {
        if (dtype() == "float32") {
            RunImpl<uint8_t, float>();
        } else if (dtype() == "float16") {
            RunImpl<uint8_t, float16>();
        } else {
            LOG(FATAL) << DTypeString(dtype(),
                { "float32", "float16" }
            );
        }
    } else {
        LOG(FATAL) << DTypeString(X(0),
            { "float32", "uint8" }
        );
    }
}

DEPLOY_CPU(ImageData);
#ifdef WITH_CUDA
DEPLOY_CUDA(ImageData);
#endif

OPERATOR_SCHEMA(ImageData)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

NO_GRADIENT(ImageData);

}  // namespace dragon