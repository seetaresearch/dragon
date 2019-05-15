#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _NNResizeNCHW(
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
    CUDA_1D_KERNEL_LOOP(yi, nthreads) {
        const int w = yi % out_w;
        const int h = (yi / out_w) % out_h;
        const int c = (yi / out_w / out_h) % C;
        const int n = yi / out_w / out_h / C;
        const int h_in = min(int(floorf(h * scale_h)), H - 1);
        const int w_in = min(int(floorf(w * scale_w)), W - 1);
#if __CUDA_ARCH__ >= 350
        y[yi] = __ldg(x + (((n * C + c) * H + h_in) * W + w_in));
#else
        y[yi] = x[((n * C + c) * H + h_in) * W + w_in];
#endif
    }
}

template <typename T>
__global__ void _NNResizeNHWC(
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
    CUDA_1D_KERNEL_LOOP(yi, nthreads) {
        const int c = yi % C;
        const int w = (yi / C) % out_w;
        const int h = (yi / C / out_w) % out_h;
        const int n = yi / C / out_w / out_h;
        const int h_in = min(int(floorf(h * scale_h)), H - 1);
        const int w_in = min(int(floorf(w * scale_w)), W - 1);
#if __CUDA_ARCH__ >= 350
        y[yi] = __ldg(x + (((n * H + h_in) * W + w_in) * C + c));
#else
        y[yi] = x[((n * H + h_in) * W + w_in) * C + c];
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
    auto scale_h = (float)H / (float)out_h;
    auto scale_w = (float)W / (float)out_w;
    if (data_format == "NCHW") {
        _NNResizeNCHW
            <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
                0, ctx->cuda_stream() >>>(
            nthreads, C, H, W, out_h, out_w,
            scale_h, scale_w, x, y
        );
    } else if(data_format == "NHWC") {
        _NNResizeNHWC
            <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
                0, ctx->cuda_stream() >>>(
            nthreads, C, H, W, out_h, out_w,
            scale_h, scale_w, x, y
        );
    } else {
        LOG(FATAL) << "Unknown DataFormat: " << data_format;
    }
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
    auto scale_h = (float)H / (float)out_h;
    auto scale_w = (float)W / (float)out_w;
    if (data_format == "NCHW") {
        _NNResizeNCHW
            <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
                0, ctx->cuda_stream() >>>(
            nthreads, C, H, W,
            out_h, out_w, scale_h, scale_w,
            reinterpret_cast<const half*>(x),
            reinterpret_cast<half*>(y)
        );
    } else if(data_format == "NHWC") {
        _NNResizeNHWC
            <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
                0, ctx->cuda_stream() >>>(
            nthreads, C, H, W,
            out_h, out_w, scale_h, scale_w,
            reinterpret_cast<const half*>(x),
            reinterpret_cast<half*>(y)
        );
    } else {
        LOG(FATAL) << "Unknown DataFormat: " << data_format;
    }
}

/* <T = float32, Device = CUDA> */

template <typename T>
__global__ void _NNResizeGradNCHW(
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
    CUDA_1D_KERNEL_LOOP(yi, nthreads) {
        const int w = yi % out_w;
        const int h = (yi / out_w) % out_h;
        const int c = (yi / out_w / out_h) % C;
        const int n = yi / out_w / out_h / C;
        const int h_in = min(int(floorf(h * scale_h)), H - 1);
        const int w_in = min(int(floorf(w * scale_w)), W - 1);
#if __CUDA_ARCH__ >= 350
        atomicAdd(&dx[((n * C + c) * H + h_in) * W + w_in], __ldg(dy + yi));
#else
        atomicAdd(&dx[((n * C + c) * H + h_in) * W + w_in], dy[yi]);
#endif
    }
}

template <typename T>
__global__ void _NNResizeGradNHWC(
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
    CUDA_1D_KERNEL_LOOP(yi, nthreads) {
        const int c = yi % C;
        const int w = (yi / C) % out_w;
        const int h = (yi / C / out_w) % out_h;
        const int n = yi / C / out_w / out_h;
        const int h_in = min(int(floorf(h * scale_h)), H - 1);
        const int w_in = min(int(floorf(w * scale_w)), W - 1);
#if __CUDA_ARCH__ >= 350
        atomicAdd(&dx[((n * H + h_in) * W + w_in) * C + c], __ldg(dy + yi));
#else
        atomicAdd(&dx[((n * H + h_in) * W + w_in) * C + c], dy[yi]);
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
    auto scale_h = (float)H / (float)out_h;
    auto scale_w = (float)W / (float)out_w;
    if (data_format == "NCHW") {
        _NNResizeGradNCHW
            <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
                0, ctx->cuda_stream() >>>(
            nthreads, C, H, W, out_h, out_w,
            scale_h, scale_w, dy, dx
        );
    } else if(data_format == "NHWC") {
        _NNResizeGradNHWC 
            <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
                0, ctx->cuda_stream() >>>(
            nthreads, C, H, W, out_h, out_w,
            scale_h, scale_w, dy, dx
        );
    } else {
        LOG(FATAL) << "Unknown DataFormat: " << data_format;
    }
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA