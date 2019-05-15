#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = float32, Device = CUDA> */

template <typename T>
__global__ void _BilinearResizeNCHW(
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
        const int n = yi / out_w / out_w / C;

        const float h_in = h * scale_h;
        const int tyi = floorf(h_in);
        const int byi = h_in < H - 1 ? ceilf(h_in) : H - 1;
        const T ylerp = h_in - tyi;

        const float w_in = w * scale_w;
        const int lxi = floorf(w_in);
        const int rxi = (w_in < W - 1) ? ceilf(w_in) : W - 1;
        const T xlerp = w_in - lxi;

        const int ncht = (n * C + c) * H + tyi;
        const int nchb = (n * C + c) * H + byi;

        const T tl(x[ncht * W + lxi]);
        const T tr(x[ncht * W + rxi]);
        const T bl(x[nchb * W + lxi]);
        const T br(x[nchb * W + rxi]);

        const T t = tl + (tr - tl) * xlerp;
        const T b = bl + (br - bl) * xlerp;
        y[yi] = t + (b - t) * ylerp;
    }
}

template <typename T>
__global__ void _BilinearResizeNHWC(
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

        const float h_in = h * scale_h;
        const int tyi = floorf(h_in);
        const int byi = (h_in < H - 1) ? ceilf(h_in) : H - 1;
        const T ylerp = h_in - tyi;

        const float w_in = w * scale_w;
        const int lxi = floorf(w_in);
        const int rxi = (w_in < W - 1) ? ceilf(w_in) : W - 1;
        const T xlerp = w_in - lxi;

        const int nht = n * H + tyi;
        const int nhb = n * H + byi;

        const T tl(x[(nht * W + lxi) * C + c]);
        const T tr(x[(nht * W + rxi) * C + c]);
        const T bl(x[(nhb * W + lxi) * C + c]);
        const T br(x[(nhb * W + rxi) * C + c]);

        const T t = tl + (tr - tl) * xlerp;
        const T b = bl + (br - bl) * xlerp;
        y[yi] = t + (b - t) * ylerp;
    }
}

template <> void BilinearResize<float, CUDAContext>(
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
         _BilinearResizeNCHW
             <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >>>(
             nthreads, C, H, W, out_h, out_w,
             scale_h, scale_w, x, y
        );
    } else if(data_format == "NHWC") {
         _BilinearResizeNHWC
             <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >>>(
             nthreads, C, H, W, out_h, out_w,
             scale_h, scale_w, x, y
        );
    } else {
        LOG(FATAL) << "Unknown DataFormat: " << data_format;
    }
}

/* <T = float32, Device = CUDA> */

template <typename T>
__global__ void _BilinearResizeGradNCHW(
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
        const int w = yi % out_w;
        const int h = (yi / out_w) % out_h;
        const int c = (yi / out_w / out_h) % C;
        const int n = yi / out_w / out_w / C;

        const float h_in = h * scale_h;
        const int tyi = floorf(h_in);
        const int byi = (h_in < H - 1) ? ceilf(h_in) : H - 1;
        const T ylerp = h_in - tyi;

        const float w_in = w * scale_w;
        const int lxi = floorf(w_in);
        const int rxi = (w_in < W - 1) ? ceilf(w_in) : W - 1;
        const T xlerp = w_in - lxi;

        const int ncht = (n * C + c) * H + tyi;
        const int nchb = (n * C + c) * H + byi;
        const T dt = (T(1) - ylerp) * dy[yi];
        const T db = ylerp * dy[yi];

        atomicAdd(&dx[ncht * W + lxi], (T(1) - xlerp) * dt);
        atomicAdd(&dx[ncht * W + rxi], xlerp * dt);
        atomicAdd(&dx[nchb * W + lxi], (T(1) - xlerp) * db);
        atomicAdd(&dx[nchb * W + rxi], xlerp * db);
    }
}

template <typename T>
__global__ void _BilinearResizeGradNHWC(
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

        const float h_in = h * scale_h;
        const int tyi = floorf(h_in);
        const int byi = (h_in < H - 1) ? ceilf(h_in) : H - 1;
        const T ylerp = h_in - tyi;

        const float w_in = w * scale_w;
        const int lxi = floorf(w_in);
        const int rxi = (w_in < W - 1) ? ceilf(w_in) : W - 1;
        const T xlerp = w_in - lxi;

        const int nht = n * H + tyi;
        const int nhb = n * H + byi;
        const T dt = (T(1) - ylerp) * dy[yi];
        const T db = ylerp * dy[yi];

        atomicAdd(&dx[(nht * W + lxi) * C + c], (T(1) - xlerp) * dt);
        atomicAdd(&dx[(nht * W + rxi) * C + c], xlerp * dt);
        atomicAdd(&dx[(nhb * W + lxi) * C + c], (T(1) - xlerp) * db);
        atomicAdd(&dx[(nhb * W + rxi) * C + c], xlerp * db);
    }
}

template <> void BilinearResizeGrad<float, CUDAContext>(
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
         _BilinearResizeGradNCHW
             <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >>>(
             nthreads, C, H, W, out_h, out_w,
             scale_h, scale_w, dy, dx
        );
    } else if(data_format == "NHWC") {
         _BilinearResizeGradNHWC
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