#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/* <Tx = ?, Ty = ?, Device = CPU> */

template <typename Tx, typename Ty>
void _ImageDataNCHW(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean,
    const float*            std,
    const Tx*               x,
    Ty*                     y) {
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                const int NH = n * H + h;
                for (int w = 0; w < W; ++w) {
                    Ty raw_value = x[(NH * W + w) * C + c];
                    if (mean) raw_value -= mean[c];
                    if (std) raw_value /= std[c];
                    *(y++) = raw_value;
                }
            }
        }
    }
}

template <typename Tx, typename Ty>
void _ImageDataNHWC(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean,
    const float*            std,
    const Tx*               x,
    Ty*                     y) {
    const auto count = N * H * W * C;
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        const int c = i % C;
        Ty raw_value = x[i];
        if (mean) raw_value -= mean[c];
        if (std) raw_value /= std[c];
        y[i] = raw_value;
    }
}

/* <Tx = float32, Ty = float32, Device = CPU> */

template <> void ImageData<float, float, CPUContext>(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const string&           data_format,
    const float*            mean,
    const float*            std,
    const float*            x,
    float*                  y,
    CPUContext*             ctx) {
    if (data_format == "NCHW") {
        _ImageDataNCHW(N, C, H, W, mean, std, x, y);
    } else if (data_format == "NHWC") {
        _ImageDataNHWC(N, C, H, W, mean, std, x, y);
    } else {
        LOG(FATAL) << "Unknown DataFormat: " << data_format;
    }
}

/* <Tx = uint8, Ty = float32, Device = CPU> */

template <> void ImageData<uint8_t, float, CPUContext>(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const string&           data_format,
    const float*            mean,
    const float*            std,
    const uint8_t*          x,
    float*                  y,
    CPUContext*             ctx) {
    if (data_format == "NCHW") {
        _ImageDataNCHW(N, C, H, W, mean, std, x, y);
    } else if (data_format == "NHWC") {
        _ImageDataNHWC(N, C, H, W, mean, std, x, y);
    } else {
        LOG(FATAL) << "Unknown DataFormat: " << data_format;
    }
}

/* <Tx = float32, Ty = float16, Device = CPU> */

template <> void ImageData<float, float16, CPUContext>(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const string&           data_format,
    const float*            mean,
    const float*            std,
    const float*            x,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

/* <Tx = uint8, Ty = float16, Device = CPU> */

template <> void ImageData<uint8_t, float16, CPUContext>(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const string&           data_format,
    const float*            mean,
    const float*            std,
    const uint8_t*          x,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

}  // namespace kernel

}  // namepsace dragon