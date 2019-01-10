#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! NNResize <T = ?, Device = CUDA> */

template <typename T>
__global__ void _NNResize_NCHW(
    const int               nthreads,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const float             scale_h,
    const float             scale_w,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(y_idx, nthreads) {
        const int w = y_idx % out_w;
        const int h = (y_idx / out_w) % out_h;
        const int c = (y_idx / out_w / out_h) % C;
        const int n = y_idx / out_w / out_h / C;
        const int h_in = min(int(floorf(h * scale_h)), H - 1);
        const int w_in = min(int(floorf(w * scale_w)), W - 1);
#if __CUDA_ARCH__ >= 350
        y[y_idx] = __ldg(x + (((n * C + c) * H + h_in) * W + w_in));
#else
        y[y_idx] = x[((n * C + c) * H + h_in) * W + w_in];
#endif
    }
}

template <typename T>
__global__ void _NNResize_NHWC(
    const int               nthreads,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const float             scale_h,
    const float             scale_w,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(y_idx, nthreads) {
        const int c = y_idx % C;
        const int w = (y_idx / C) % out_w;
        const int h = (y_idx / C / out_w) % out_h;
        const int n = y_idx / C / out_w / out_h;
        const int h_in = min(int(floorf(h * scale_h)), H - 1);
        const int w_in = min(int(floorf(w * scale_w)), W - 1);
#if __CUDA_ARCH__ >= 350
        y[y_idx] = __ldg(x + (((n * H + h_in) * W + w_in) * C + c));
#else
        y[y_idx] = x[((n * H + h_in) * W + w_in) * C + c];
#endif
    }
}

/*! NNResize <T = float32, Device = CUDA> */

template <> void NNResize<float, CUDAContext>(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const string&           data_format,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    auto nthreads = N * C * out_h * out_w;
    const float scale_h = (float)H / out_h;
    const float scale_w = (float)W / out_w;
    if (data_format == "NCHW") {
        _NNResize_NCHW<float>
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (nthreads, C, H, W, out_h, out_w,
                scale_h, scale_w, x, y);
    } else if(data_format == "NHWC") {
        _NNResize_NHWC<float>
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (nthreads, C, H, W, out_h, out_w,
                scale_h, scale_w, x, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/*! NNResize <T = float16, Device = CUDA> */

template <> void NNResize<float16, CUDAContext>(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const string&           data_format,
    const float16*          x,
    float16*                y,
    CUDAContext*            ctx) {
    auto nthreads = N * C * out_h * out_w;
    const float scale_h = (float)H / out_h;
    const float scale_w = (float)W / out_w;
    if (data_format == "NCHW") {
        _NNResize_NCHW<half>
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (nthreads, C, H, W, out_h, out_w, scale_h, scale_w,
                reinterpret_cast<const half*>(x),
                    reinterpret_cast<half*>(y));
    } else if(data_format == "NHWC") {
        _NNResize_NHWC<half>
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (nthreads, C, H, W, out_h, out_w, scale_h, scale_w,
                reinterpret_cast<const half*>(x),
                    reinterpret_cast<half*>(y));
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/*! NNResizeGrad <T = float32, Device = CUDA> */

template <typename T>
 __global__ void _NNResizeGrad_NCHW(
     const int              nthreads,
     const int              C,
     const int              H,
     const int              W,
     const int              out_h,
     const int              out_w,
     const float            scale_h,
     const float            scale_w,
     const T*               dy,
     T*                     dx) {
    CUDA_1D_KERNEL_LOOP(y_idx, nthreads) {
        const int w = y_idx % out_w;
        const int h = (y_idx / out_w) % out_h;
        const int c = (y_idx / out_w / out_h) % C;
        const int n = y_idx / out_w / out_h / C;
        const int h_in = min(int(floorf(h * scale_h)), H - 1);
        const int w_in = min(int(floorf(w * scale_w)), W - 1);
#if __CUDA_ARCH__ >= 350
        atomicAdd(&dx[((n * C + c) * H + h_in) * W + w_in], __ldg(dy + y_idx));
#else
        atomicAdd(&dx[((n * C + c) * H + h_in) * W + w_in], dy[y_idx]);
#endif
    }
}

template <typename T>
__global__ void _NNResizeGrad_NHWC(
    const int               nthreads,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const float             scale_h,
    const float             scale_w,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(y_idx, nthreads) {
        const int c = y_idx % C;
        const int w = (y_idx / C) % out_w;
        const int h = (y_idx / C / out_w) % out_h;
        const int n = y_idx / C / out_w / out_h;
        const int h_in = min(int(floorf(h * scale_h)), H - 1);
        const int w_in = min(int(floorf(w * scale_w)), W - 1);
#if __CUDA_ARCH__ >= 350
        atomicAdd(&dx[((n * H + h_in) * W + w_in) * C + c], __ldg(dy + y_idx));
#else
        atomicAdd(&dx[((n * H + h_in) * W + w_in) * C + c], dy[y_idx]);
#endif
    }
}

template <> void NNResizeGrad<float, CUDAContext>(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const string&           data_format,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    auto nthreads = N * C * out_h * out_w;
    const float scale_h = (float)H / out_h;
    const float scale_w = (float)W / out_w;
    if (data_format == "NCHW") {
        _NNResizeGrad_NCHW<float>
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (nthreads, C, H, W, out_h, out_w,
                scale_h, scale_w, dy, dx);
    } else if(data_format == "NHWC") {
        _NNResizeGrad_NHWC<float> 
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (nthreads, C, H, W, out_h, out_w,
                scale_h, scale_w, dy, dx);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA