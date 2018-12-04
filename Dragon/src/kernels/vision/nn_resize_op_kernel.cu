#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! NNResize <T = ?, Device = CUDA> */

template <typename T>
__global__ void _NNResize_NCHW(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const float             scale_h,
    const float             scale_w,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int w = idx % out_w;
        const int h = (idx / out_w) % out_h;
        const int c = (idx / out_w / out_h) % C;
        const int n = idx / out_w / out_h / C;
        const int h_in = min(int(floorf(h * scale_h)), H - 1);
        const int w_in = min(int(floorf(w * scale_w)), W - 1);
        y[idx] = x[((n * C + c) * H + h_in) * W + w_in];
    }
}

template <typename T>
__global__ void _NNResize_NHWC(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const float             scale_h,
    const float             scale_w,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int c = idx % C;
        const int w = (idx / C) % out_w;
        const int h = (idx / C / out_w) % out_h;
        const int n = idx / C / out_w / out_h;
        const int h_in = min(int(floorf(h * scale_h)), H - 1);
        const int w_in = min(int(floorf(w * scale_w)), W - 1);
        y[idx] = x[((n * H + h_in) * W + w_in) * C + c];
    }
}

/*! NNResize <T = float32, Device = CUDA> */

template <> void NNResize<float, CUDAContext>(
    const int               count,
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
    const float scale_h = (float)H / out_h;
    const float scale_w = (float)W / out_w;
    if (data_format == "NCHW") {
        _NNResize_NCHW<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, N, C, H, W, out_h, out_w,
                scale_h, scale_w, x, y);
    } else if(data_format == "NHWC") {
        _NNResize_NHWC<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, N, C, H, W, out_h, out_w,
                scale_h, scale_w, x, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/*! NNResize <T = float16, Device = CUDA> */

template <> void NNResize<float16, CUDAContext>(
    const int               count,
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
    const float scale_h = (float)H / out_h;
    const float scale_w = (float)W / out_w;
    if (data_format == "NCHW") {
        _NNResize_NCHW<half>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, N, C, H, W, out_h, out_w, scale_h, scale_w,
                reinterpret_cast<const half*>(x),
                    reinterpret_cast<half*>(y));
    } else if(data_format == "NHWC") {
        _NNResize_NHWC<half>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, N, C, H, W, out_h, out_w, scale_h, scale_w, 
                reinterpret_cast<const half*>(x),
                    reinterpret_cast<half*>(y));
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/*! NNResizeGrad <T = float32, Device = CUDA> */

template <typename T>
 __global__ void _NNResizeGrad_NCHW(
     const int              count,
     const int              N,
     const int              C,
     const int              H,
     const int              W,
     const int              out_h,
     const int              out_w,
     const float            scale_h,
     const float            scale_w,
     const T*               dy,
     T*                     dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int w = idx % out_w;
        const int h = (idx / out_w) % out_h;
        const int c = (idx / out_w / out_h) % C;
        const int n = idx / out_w / out_h / C;
        const int h_in = min(int(floorf(h * scale_h)), H - 1);
        const int w_in = min(int(floorf(w * scale_w)), W - 1);
        atomicAdd(&dx[((n * C + c) * H + h_in) * W + w_in], dy[idx]);
    }
}

template <typename T>
__global__ void _NNResizeGrad_NHWC(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const float             scale_h,
    const float             scale_w,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int c = idx % C;
        const int w = (idx / C) % out_w;
        const int h = (idx / C / out_w) % out_h;
        const int n = idx / C / out_w / out_h;
        const int h_in = min(int(floorf(h * scale_h)), H - 1);
        const int w_in = min(int(floorf(w * scale_w)), W - 1);
        atomicAdd(&dx[((n * H + h_in) * W + w_in) * C + c], dy[idx]);
    }
}

template <> void NNResizeGrad<float, CUDAContext>(
    const int               count,
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
    const float scale_h = (float)H / out_h;
    const float scale_w = (float)W / out_w;
    if (data_format == "NCHW") {
        _NNResizeGrad_NCHW<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, N, C, H, W, out_h, out_w,
                scale_h, scale_w, dy, dx);
    } else if(data_format == "NHWC") {
        _NNResizeGrad_NHWC<float> 
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, N, C, H, W, out_h, out_w,
                scale_h, scale_w, dy, dx);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA