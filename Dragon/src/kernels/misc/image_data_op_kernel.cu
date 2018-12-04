#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! ImageData <Tx = ?, Ty = ?, Device = CUDA> */

template <typename Tx, typename Ty>
__global__ void _ImageData_NCHW(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean_values,
    const float*            std_values,
    const Tx*               x,
    Ty*                     y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int w = idx % W;
        const int h = (idx / W) % H;
        const int c = (idx / W / H) % C;
        const int n = idx / W / H / C;
        Ty raw_value = x[((n * H + h) * W + w) * C + c];
        if (mean_values) raw_value -= mean_values[c];
        if (std_values) raw_value /= std_values[c];
        y[idx] = raw_value;
    }
}

template <typename Tx, typename Ty>
__global__ void _ImageData_NHWC(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean_values,
    const float*            std_values,
    const Tx*               x,
    Ty*                     y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int c = idx % C;
        Ty raw_value = x[idx];
        if (mean_values) raw_value -= mean_values[c];
        if (std_values) raw_value /= std_values[c];
        y[idx] = raw_value;
    }
}

/*! ImageData <Tx = float32, Ty = float32, Device = CUDA> */

template <> void ImageData<float, float, CUDAContext>(
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
    CUDAContext*            ctx) {
    if (data_format == "NCHW") {
        _ImageData_NCHW<float, float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, N, C, H, W, mean_values, std_values, x, y);
    } else if (data_format == "NHWC") {
        _ImageData_NHWC<float, float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, N, C, H, W, mean_values, std_values, x, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/*! ImageData <Tx = uint8, Ty = float32, Device = CUDA> */

template <> void ImageData<uint8_t, float, CUDAContext>(
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
    CUDAContext*            ctx) {
    if (data_format == "NCHW") {
        _ImageData_NCHW<uint8_t, float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, N, C, H, W, mean_values, std_values, x, y);
    } else if (data_format == "NHWC") {
        _ImageData_NHWC<uint8_t, float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, N, C, H, W, mean_values, std_values, x, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/*! ImageData <Tx = ?, Ty = float16, Device = CUDA> */

template <typename Tx, typename Ty>
__global__ void _ImageDataHalf_NCHW(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean_values,
    const float*            std_values,
    const Tx*               x,
    Ty*                     y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int w = idx % W;
        const int h = (idx / W) % H;
        const int c = (idx / W / H) % C;
        const int n = idx / W / H / C;
        float raw_value = x[((n * H + h) * W + w) * C + c];
        if (mean_values) raw_value -= mean_values[c];
        if (std_values) raw_value /= std_values[c];
        y[idx] = __float2half(raw_value);
    }
}

template <typename Tx, typename Ty>
__global__ void _ImageDataHalf_NHWC(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean_values,
    const float*            std_values,
    const Tx*               x,
    Ty*                     y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int c = idx % C;
        float raw_value = x[idx];
        if (mean_values) raw_value -= mean_values[c];
        if (std_values) raw_value /= std_values[c];
        y[idx] = __float2half(raw_value);
    }
}

/*! ImageData <Tx = float32, Ty = float16, Device = CUDA> */

template <> void ImageData<float, float16, CUDAContext>(
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
    CUDAContext*            ctx) {
    if (data_format == "NCHW") {
        _ImageDataHalf_NCHW<float, half>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, N, C, H, W, mean_values, std_values,
                x, reinterpret_cast<half*>(y));
    } else if (data_format == "NHWC") {
        _ImageDataHalf_NHWC<float, half> 
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, N, C, H, W, mean_values, std_values,
                x, reinterpret_cast<half*>(y));
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template <> void ImageData<uint8_t, float16, CUDAContext>(
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
    CUDAContext*            ctx) {
    if (data_format == "NCHW") {
        _ImageDataHalf_NCHW<uint8_t, half>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, N, C, H, W, mean_values, std_values,
                x, reinterpret_cast<half*>(y));
    } else if (data_format == "NHWC") {
        _ImageDataHalf_NHWC<uint8_t, half>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, N, C, H, W, mean_values, std_values,
                x, reinterpret_cast<half*>(y));
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA