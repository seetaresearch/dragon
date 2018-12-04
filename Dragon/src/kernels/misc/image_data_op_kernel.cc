#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! ImageData <Tx = ?, Ty = ?, Device = CPU> */

template <typename Tx, typename Ty>
void _ImageData_NCHW(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean_values,
    const float*            std_values,
    const Tx*               x,
    Ty*                     y) {
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                const int NH = n * H + h;
                for (int w = 0; w < W; ++w) {
                    Ty raw_value = x[(NH * W + w) * C + c];
                    if (mean_values) raw_value -= mean_values[c];
                    if (std_values) raw_value /= std_values[c];
                    *(y++) = raw_value;
                }
            }
        }
    }
}

template <typename Tx, typename Ty>
void _ImageData_NHWC(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean_values,
    const float*            std_values,
    const Tx*               x,
    Ty*                     y) {
    const auto count = N * H * W * C;
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        const int c = i % C;
        Ty raw_value = x[i];
        if (mean_values) raw_value -= mean_values[c];
        if (std_values) raw_value /= std_values[c];
        y[i] = raw_value;
    }
}

/*! ImageData <Tx = float32, Ty = float32, Device = CPU> */

template <> void ImageData<float, float, CPUContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean_values,
    const float*            std_values,
    const string&           data_format,
    const float*            x,
    float*                  y,
    CPUContext*             ctx) {
    if (data_format == "NCHW") {
        _ImageData_NCHW<float, float>(
            N, C, H, W, mean_values, std_values, x, y);
    } else if (data_format == "NHWC") {
        _ImageData_NHWC<float, float>(
            N, C, H, W, mean_values, std_values, x, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/*! ImageData <Tx = uint8, Ty = float32, Device = CPU> */

template <> void ImageData<uint8_t, float, CPUContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean_values,
    const float*            std_values,
    const string&           data_format,
    const uint8_t*          x,
    float*                  y,
    CPUContext*             ctx) {
    if (data_format == "NCHW") {
        _ImageData_NCHW<uint8_t, float>(
            N, C, H, W, mean_values, std_values, x, y);
    } else if (data_format == "NHWC") {
        _ImageData_NHWC<uint8_t, float>(
            N, C, H, W, mean_values, std_values, x, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/*! ImageData <Tx = float32, Ty = float16, Device = CPU> */

template <> void ImageData<float, float16, CPUContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean_values,
    const float*            std_values,
    const string&           data_format,
    const float*            x,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

/*! ImageData <Tx = uint8, Ty = float16, Device = CPU> */

template <> void ImageData<uint8_t, float16, CPUContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean_values,
    const float*            std_values,
    const string&           data_format,
    const uint8_t*          x,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

}  // namespace kernel

}  // namepsace dragon