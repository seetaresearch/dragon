#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! ImageData <Tx = ?, Ty = ?, Device = CUDA> */

template <typename Tx, typename Ty>
__global__ void _ImageDataNCHW(
    const int               nthreads,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean,
    const float*            std,
    const Tx*               x,
    Ty*                     y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const int w = i % W;
        const int h = (i / W) % H;
        const int c = (i / W / H) % C;
        const int n = i / W / H / C;
        Ty raw_value = x[((n * H + h) * W + w) * C + c];
#if __CUDA_ARCH__ >= 350
        if (mean) raw_value -= __ldg(mean + c);
        if (std) raw_value /= __ldg(std + c);
#else
        if (mean) raw_value -= mean[c];
        if (std) raw_value /= std[c];
#endif
        y[i] = raw_value;
    }
}

template <typename Tx, typename Ty>
__global__ void _ImageDataNHWC(
    const int               nthreads,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean,
    const float*            std,
    const Tx*               x,
    Ty*                     y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const int c = i % C;
        Ty raw_value = x[i];
#if __CUDA_ARCH__ >= 350
        if (mean) raw_value -= __ldg(mean + c);
        if (std) raw_value /= __ldg(std + c);
#else
        if (mean) raw_value -= mean[c];
        if (std) raw_value /= std[c];
#endif
        y[i] = raw_value;
    }
}

/*! ImageData <Tx = float32, Ty = float32, Device = CUDA> */

template <> void ImageData<float, float, CUDAContext>(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const string&           data_format,
    const float*            mean,
    const float*            std,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    auto nthreads = N * C * H * W;
    if (data_format == "NCHW") {
        _ImageDataNCHW
            <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
                0, ctx->cuda_stream() >>>(
            nthreads, C, H, W, mean, std, x, y
        );
    } else if (data_format == "NHWC") {
        _ImageDataNHWC
            <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
                0, ctx->cuda_stream() >>>(
            nthreads, C, H, W, mean, std, x, y
       );
    } else {
        LOG(FATAL) << "Unknown data format: " << data_format;
    }
}

/*! ImageData <Tx = uint8, Ty = float32, Device = CUDA> */

template <> void ImageData<uint8_t, float, CUDAContext>(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const string&           data_format,
    const float*            mean,
    const float*            std,
    const uint8_t*          x,
    float*                  y,
    CUDAContext*            ctx) {
    auto nthreads = N * C * H * W;
    if (data_format == "NCHW") {
        _ImageDataNCHW
            <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
                0, ctx->cuda_stream() >>>(
            nthreads, C, H, W, mean, std, x, y
        );
    } else if (data_format == "NHWC") {
        _ImageDataNHWC
            <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
                0, ctx->cuda_stream() >>>(
            nthreads, C, H, W, mean, std, x, y
       );
    } else {
        LOG(FATAL) << "Unknown data format: " << data_format;
    }
}

/*! ImageData <Tx = ?, Ty = float16, Device = CUDA> */

template <typename Tx, typename Ty>
__global__ void _ImageDataHalfNCHW(
    const int               nthreads,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean,
    const float*            std,
    const Tx*               x,
    Ty*                     y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const int w = i % W;
        const int h = (i / W) % H;
        const int c = (i / W / H) % C;
        const int n = i / W / H / C;
        float raw_value = x[((n * H + h) * W + w) * C + c];
#if __CUDA_ARCH__ >= 350
        if (mean) raw_value -= __ldg(mean + c);
        if (std) raw_value /= __ldg(std + c);
#else
        if (mean) raw_value -= mean[c];
        if (std) raw_value /= std[c];
#endif
        y[i] = __float2half(raw_value);
    }
}

template <typename Tx, typename Ty>
__global__ void _ImageDataHalfNHWC(
    const int               nthreads,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean,
    const float*            std,
    const Tx*               x,
    Ty*                     y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const int c = i % C;
        float raw_value = x[i];
#if __CUDA_ARCH__ >= 350
        if (mean) raw_value -= __ldg(mean + c);
        if (std) raw_value /= __ldg(std + c);
#else
        if (mean) raw_value -= mean[c];
        if (std) raw_value /= std[c];
#endif
        y[i] = __float2half(raw_value);
    }
}

/*! ImageData <Tx = float32, Ty = float16, Device = CUDA> */

template <> void ImageData<float, float16, CUDAContext>(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const string&           data_format,
    const float*            mean,
    const float*            std,
    const float*            x,
    float16*                y,
    CUDAContext*            ctx) {
    auto nthreads = N * C * H * W;
    if (data_format == "NCHW") {
        _ImageDataHalfNCHW
            <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
                0, ctx->cuda_stream() >>>(
            nthreads, C, H, W, mean, std,
            x, reinterpret_cast<half*>(y)
        );
    } else if (data_format == "NHWC") {
        _ImageDataHalfNHWC
            <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
                0, ctx->cuda_stream() >>>(
           nthreads, C, H, W, mean, std,
           x, reinterpret_cast<half*>(y)
        );
    } else {
        LOG(FATAL) << "Unknown data format: " << data_format;
    }
}

template <> void ImageData<uint8_t, float16, CUDAContext>(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const string&           data_format,
    const float*            mean,
    const float*            std,
    const uint8_t*          x,
    float16*                y,
    CUDAContext*            ctx) {
    auto nthreads = N * C * H * W;
    if (data_format == "NCHW") {
        _ImageDataHalfNCHW
            <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
                0, ctx->cuda_stream() >>>(
            nthreads, C, H, W, mean, std,
            x, reinterpret_cast<half*>(y)
        );
    } else if (data_format == "NHWC") {
        _ImageDataHalfNHWC
            <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
                0, ctx->cuda_stream() >>>(
            nthreads, C, H, W, mean, std,
            x, reinterpret_cast<half*>(y)
        );
    } else {
        LOG(FATAL) << "Unknown data format: " << data_format;
    }
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA