#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! BilinearResize <T = float32, Device = CUDA> */

template <typename T>
__global__ void _BilinearResize_NCHW(
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
        const int n = idx / out_w / out_w / C;

        const float h_in = h * scale_h;
        const int top_y_idx = floorf(h_in);
        const int bottom_y_idx = (h_in < H - 1) ? ceilf(h_in) : H - 1;
        const float y_lerp = h_in - top_y_idx;

        const float w_in = w * scale_w;
        const int left_x_idx = floorf(w_in);
        const int right_x_idx = (w_in < W - 1) ? ceilf(w_in) : W - 1;
        const float x_lerp = w_in - left_x_idx;

        const int NCHT = (n * C + c) * H + top_y_idx;
        const int NCHB = (n * C + c) * H + bottom_y_idx;

        const float top_left(x[NCHT * W + left_x_idx]);
        const float top_right(x[NCHT * W + right_x_idx]);
        const float bottom_left(x[NCHB * W + left_x_idx]);
        const float bottom_right(x[NCHB * W + right_x_idx]);

        const float top = top_left + (top_right - top_left) * x_lerp;
        const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
        y[idx] = top + (bottom - top) * y_lerp;
    }
}

template <typename T>
__global__ void _BilinearResize_NHWC(
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

        const float h_in = h * scale_h;
        const int top_y_idx = floorf(h_in);
        const int bottom_y_idx = (h_in < H - 1) ? ceilf(h_in) : H - 1;
        const float y_lerp = h_in - top_y_idx;

        const float w_in = w * scale_w;
        const int left_x_idx = floorf(w_in);
        const int right_x_idx = (w_in < W - 1) ? ceilf(w_in) : W - 1;
        const float x_lerp = w_in - left_x_idx;

        const int NHT = n * H + top_y_idx;
        const int NHB = n * H + bottom_y_idx;

        const float top_left(x[(NHT * W + left_x_idx) * C + c]);
        const float top_right(x[(NHT * W + right_x_idx) * C + c]);
        const float bottom_left(x[(NHB * W + left_x_idx) * C + c]);
        const float bottom_right(x[(NHB * W + right_x_idx) * C + c]);

        const float top = top_left + (top_right - top_left) * x_lerp;
        const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
        y[idx] = top + (bottom - top) * y_lerp;
    }
}

template <> void BilinearResize<float, CUDAContext>(
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
         _BilinearResize_NCHW<float>
             << < CUDA_BLOCKS(count), CUDA_THREADS,
                  0, ctx->cuda_stream() >> >
             (count, N, C, H, W, out_h, out_w,
                 scale_h, scale_w, x, y);
    } else if(data_format == "NHWC") {
         _BilinearResize_NHWC<float>
             << < CUDA_BLOCKS(count), CUDA_THREADS,
                  0, ctx->cuda_stream() >> >
             (count, N, C, H, W, out_h, out_w,
                 scale_h, scale_w, x, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/*! BilinearResizeGrad <T = float32, Device = CUDA> */

template <typename T>
__global__ void _BilinearResizeGrad_NCHW(
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
        const int w = idx % out_w;
        const int h = (idx / out_w) % out_h;
        const int c = (idx / out_w / out_h) % C;
        const int n = idx / out_w / out_w / C;

        const float h_in = h * scale_h;
        const int top_y_idx = floorf(h_in);
        const int bottom_y_idx = (h_in < H - 1) ? ceilf(h_in) : H - 1;
        const float y_lerp = h_in - top_y_idx;

        const float w_in = w * scale_w;
        const int left_x_idx = floorf(w_in);
        const int right_x_idx = (w_in < W - 1) ? ceilf(w_in) : W - 1;
        const float x_lerp = w_in - left_x_idx;

        const int NCHT = (n * C + c) * H + top_y_idx;
        const int NCHB = (n * C + c) * H + bottom_y_idx;
        const float dtop = (1 - y_lerp) * dy[idx];
        const float dbottom = y_lerp * dy[idx];

        atomicAdd(&dx[NCHT * W + left_x_idx], static_cast<T>((1 - x_lerp) * dtop));
        atomicAdd(&dx[NCHT * W + right_x_idx], static_cast<T>(x_lerp * dtop));
        atomicAdd(&dx[NCHB * W + left_x_idx], static_cast<T>((1 - x_lerp) * dbottom));
        atomicAdd(&dx[NCHB * W + right_x_idx], static_cast<T>(x_lerp * dbottom));
    }
}

template <typename T>
__global__ void _BilinearResizeGrad_NHWC(
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

        const float h_in = h * scale_h;
        const int top_y_idx = floorf(h_in);
        const int bottom_y_idx = (h_in < H - 1) ? ceilf(h_in) : H - 1;
        const float y_lerp = h_in - top_y_idx;

        const float w_in = w * scale_w;
        const int left_x_idx = floorf(w_in);
        const int right_x_idx = (w_in < W - 1) ? ceilf(w_in) : W - 1;
        const float x_lerp = w_in - left_x_idx;

        const int NHT = n * H + top_y_idx;
        const int NHB = n * H + bottom_y_idx;
        const float dtop = (1 - y_lerp) * dy[idx];
        const float dbottom = y_lerp * dy[idx];

        atomicAdd(&dx[(NHT * W + left_x_idx) * C + c], static_cast<T>((1 - x_lerp) * dtop));
        atomicAdd(&dx[(NHT * W + right_x_idx) * C + c], static_cast<T>(x_lerp * dtop));
        atomicAdd(&dx[(NHB * W + left_x_idx) * C + c], static_cast<T>((1 - x_lerp) * dbottom));
        atomicAdd(&dx[(NHB * W + right_x_idx) * C + c], static_cast<T>(x_lerp * dbottom));
    }
}

template <> void BilinearResizeGrad<float, CUDAContext>(
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
         _BilinearResizeGrad_NCHW<float>
             << < CUDA_BLOCKS(count), CUDA_THREADS,
                  0, ctx->cuda_stream() >> >
             (count, N, C, H, W, out_h, out_w,
                 scale_h, scale_w, dy, dx);
    } else if(data_format == "NHWC") {
         _BilinearResizeGrad_NHWC<float>
             << < CUDA_BLOCKS(count), CUDA_THREADS,
                  0, ctx->cuda_stream() >> >
             (count, N, C, H, W, out_h, out_w,
                 scale_h, scale_w, dy, dx);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA