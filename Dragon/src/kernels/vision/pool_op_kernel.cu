#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = float32, Device = CUDA> */

template<typename T>
__global__ void _MaxPool2dNCHW(
    const int               nthreads,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const T*                x,
    int*                    mask,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(yi, nthreads) {
        const int pw = yi % pool_w;
        const int ph = (yi / pool_w) % pool_h;
        const int c = (yi / pool_w / pool_h) % C;
        const int n = yi / pool_w / pool_h / C;

        int h_start = ph * stride_h - pad_h;
        int w_start = pw * stride_w - pad_w;
        const int h_end = min(h_start + kernel_h, H);
        const int w_end = min(w_start + kernel_w, W);
        h_start = max(h_start, 0);
        w_start = max(w_start, 0);

        int maxi = -1; T max_val = -FLT_MAX;
        const T* X = x + (n * C + c) * H * W;
        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                if (X[h * W + w] > max_val) {
                    maxi = h * W + w;
                    max_val = X[maxi];
                }
            }
        }
        y[yi] = max_val;
        mask[yi] = maxi;
    }
}

template<typename T>
__global__ void _MaxPool2dNHWC(
    const int               nthreads,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const T*                x,
    int*                    mask,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(yi, nthreads) {
        const int c = yi % C;
        const int pw = (yi / C) % pool_w;
        const int ph = (yi / C / pool_w) % pool_h;
        const int n = yi / C / pool_w / pool_h;

        int h_start = ph * stride_h - pad_h;
        int w_start = pw * stride_w - pad_w;
        const int h_end = min(h_start + kernel_h, H);
        const int w_end = min(w_start + kernel_w, W);
        h_start = max(h_start, 0);
        w_start = max(w_start, 0);

        int maxi = -1; T max_val = -FLT_MAX;
        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                const int xi = ((n * H + h) * W + w) * C + c;
                if (x[xi] > max_val) {
                    maxi = xi;
                    max_val = x[xi];
                }
            }
        }
        y[yi] = max_val;
        mask[yi] = maxi;
    }
}

template<> void MaxPool2d<float, CUDAContext>(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const string&           data_format,
    const float*            x,
    int*                    mask,
    float*                  y,
    CUDAContext*            ctx) {
    auto nthreads = N * C * pool_h * pool_w;
    if (data_format == "NCHW") {
        _MaxPool2dNCHW
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(
            nthreads,
            C, H, W,
            pool_h, pool_w,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            x, mask, y
        );
    } else if (data_format == "NHWC") {
        _MaxPool2dNHWC
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(
            nthreads,
            C, H, W,
            pool_h, pool_w,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            x, mask, y
        );
    } else {
        LOG(FATAL) << "Unknown DataFormat: " << data_format;
    }
}

/* <T = float32, Device = CUDA> */

template<typename T>
__global__ void _AvgPool2dNCHW(
    const int               nthreads,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(yi, nthreads) {
        const int pw = yi % pool_w;
        const int ph = (yi / pool_w) % pool_h;
        const int c = (yi / pool_w / pool_h) % C;
        const int n = yi / pool_w / pool_h / C;

        int h_start = ph * stride_h - pad_h;
        int w_start = pw * stride_w - pad_w;
        int h_end = min(h_start + kernel_h, H + pad_h);
        int w_end = min(w_start + kernel_w, W + pad_w);

        h_start = max(h_start, 0);
        w_start = max(w_start, 0);
        h_end = min(h_end, H);
        w_end = min(w_end, W);

        T sum_val = T(0);
        const T* X = x + (n * C + c) * H * W;
        const T area = (h_end - h_start) * (w_end - w_start);
        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                sum_val += X[h * W + w];
            }
        }
        y[yi] = sum_val / area;
    }
}

template<typename T>
__global__ void _AvgPool2dNHWC(
    const int               nthreads,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(yi, nthreads) {
        const int c = yi % C;
        const int pw = (yi / C) % pool_w;
        const int ph = (yi / C / pool_w) % pool_h;
        const int n = yi / C / pool_w / pool_h;

        int h_start = ph * stride_h - pad_h;
        int w_start = pw * stride_w - pad_w;
        int h_end = min(h_start + kernel_h, H + pad_h);
        int w_end = min(w_start + kernel_w, W + pad_w);

        h_start = max(h_start, 0);
        w_start = max(w_start, 0);
        h_end = min(h_end, H);
        w_end = min(w_end, W);

        T sum_val = 0;
        const T area = (h_end - h_start) * (w_end - w_start);
        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                sum_val += x[((n * H + h) * W + w) * C + c];
            }
        }
        y[yi] = sum_val / area;
    }
}

template<> void AvgPool2d<float, CUDAContext>(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const string&           data_format,
    const float*            x,
    float*                  y,
    CUDAContext*            ctx) {
    auto nthreads = N * C * pool_h * pool_w;
    if (data_format == "NCHW") {
        _AvgPool2dNCHW
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(
            nthreads,
            C, H, W,
            pool_h, pool_w,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            x, y
        );
    } else if (data_format == "NHWC") {
        _AvgPool2dNHWC
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(
            nthreads,
            C, H, W,
            pool_h, pool_w,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            x, y
        );
    } else {
        LOG(FATAL) << "Unknown DataFormat: " << data_format;
    }
}

/* <T = float32, Device = CUDA> */

template<typename T>
__global__ void _MaxPool2dGrad_NCHW(
    const int               nthreads,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const T*                dy,
    const int*              mask,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(xi, nthreads) {
        const int w = xi % W;
        const int h = (xi / W) % H;
        const int c = (xi / W / H) % C;
        const int n = xi / W / H / C;

        const int ph_start = (h + pad_h < kernel_h) ?
            0 : (h + pad_h - kernel_h) / stride_h + 1;
        const int pw_start = (w + pad_w < kernel_w) ? 
            0 : (w + pad_w - kernel_w) / stride_w + 1;
        const int ph_end = min((h + pad_h) / stride_h + 1, pool_h);
        const int pw_end = min((w + pad_w) / stride_w + 1, pool_w);

        T grad = T(0);
        const int offset = (n * C + c) * pool_h * pool_w;
        const T* dY = dy + offset;
        const int* M = mask + offset;
        for (int ph = ph_start; ph < ph_end; ++ph) {
            for (int pw = pw_start; pw < pw_end; ++pw) {
                if (M[ph * pool_w + pw] == (h * W + w)) {
                    grad += dY[ph * pool_w + pw];
                }
            }
        }
        dx[xi] = grad;
    }
}

template<typename T>
__global__ void _MaxPool2dGradNHWC(
    const int               nthreads,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const T*                dy,
    const int*              mask,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(xi, nthreads) {
        const int c = xi % C;
        const int w = (xi / C) % W;
        const int h = (xi / C / W) % H;
        const int n = xi / C / W / H;

        const int ph_start = (h + pad_h < kernel_h) ?
            0 : (h + pad_h - kernel_h) / stride_h + 1;
        const int pw_start = (w + pad_w < kernel_w) ?
            0 : (w + pad_w - kernel_w) / stride_w + 1;
        const int ph_end = min((h + pad_h) / stride_h + 1, pool_h);
        const int pw_end = min((w + pad_w) / stride_w + 1, pool_w);

        T grad = T(0);
        for (int ph = ph_start; ph < ph_end; ++ph) {
            for (int pw = pw_start; pw < pw_end; ++pw) {
                const int yi = ((n * pool_h + ph) * pool_w + pw) * C + c;
                if (mask[yi] == xi) grad += dy[yi];
            }
        }
        dx[xi] = grad;
    }
}

template<> void MaxPool2dGrad<float, CUDAContext>(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const string&           data_format,
    const float*            dy,
    const int*              mask,
    float*                  dx,
    CUDAContext*            ctx) {
    auto nthreads = N * C * H * W;
    if (data_format == "NCHW") {
        _MaxPool2dGrad_NCHW
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(
            nthreads,
            C, H, W,
            pool_h, pool_w,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            dy,mask, dx
        );
    } else if (data_format == "NHWC") {
        _MaxPool2dGradNHWC
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(
            nthreads,
            C, H, W,
            pool_h, pool_w,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            dy, mask, dx
        );
    } else {
        LOG(FATAL) << "Unknown DataFormat: " << data_format;
    }
}

/* <T = float32, Device = CUDA> */

template<typename T>
__global__ void _AvgPool2dGradNCHW(
    const int               nthreads,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(xi, nthreads) {
        const int w = xi % W;
        const int h = (xi / W) % H;
        const int c = (xi / W / H) % C;
        const int n = xi / W / H / C;

        const int ph_start = (h + pad_h < kernel_h) ?
            0 : (h + pad_h - kernel_h) / stride_h + 1;
        const int pw_start = (w + pad_w < kernel_w) ?
            0 : (w + pad_w - kernel_w) / stride_w + 1;
        const int ph_end = min(h / stride_h + 1, pool_h);
        const int pw_end = min(w / stride_w + 1, pool_w);

        T grad = T(0);
        const T* dY = dy + (n * C + c) * pool_h * pool_w;
        for (int ph = ph_start; ph < ph_end; ++ph) {
            for (int pw = pw_start; pw < pw_end; ++pw) {
                const int h_start = ph * stride_h - pad_h;
                const int w_start = pw * stride_w - pad_w;
                const int h_end = min(h_start + kernel_h, H + pad_h);
                const int w_end = min(w_start + kernel_w, W + pad_w);
                const T area = (h_end - h_start) * (w_end - w_start);
                grad += dY[ph * pool_w + pw] / area;
            }
        }
        dx[xi] = grad;
    }
}

template<typename T>
__global__ void _AvgPool2dGradNHWC(
    const int               nthreads,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(xi, nthreads) {
        const int c = xi % C;
        const int w = (xi / C) % W;
        const int h = (xi / C / W) % H;
        const int n = xi / C / W / H;

        const int ph_start = (h + pad_h < kernel_h) ?
            0 : (h + pad_h - kernel_h) / stride_h + 1;
        const int pw_start = (w + pad_w < kernel_w) ?
            0 : (w + pad_w - kernel_w) / stride_w + 1;
        const int ph_end = min(h / stride_h + 1, pool_h);
        const int pw_end = min(w / stride_w + 1, pool_w);

        T grad = 0;
        for (int ph = ph_start; ph < ph_end; ++ph) {
            for (int pw = pw_start; pw < pw_end; ++pw) {
                const int h_start = ph * stride_h - pad_h;
                const int w_start = pw * stride_w - pad_w;
                const int h_end = min(h_start + kernel_h, H + pad_h);
                const int w_end = min(w_start + kernel_w, W + pad_w);
                const T area = (h_end - h_start) * (w_end - w_start);
                const int yi = ((n * pool_h + ph) * pool_w + pw) * C + c;
                grad += dy[yi] / area;
            }
        }
        dx[xi] = grad;
    }
}

template<> void AvgPool2dGrad<float, CUDAContext>(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const string&           data_format,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    auto nthreads = N * C * H * W;
    if (data_format == "NCHW") {
        _AvgPool2dGradNCHW
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(
            nthreads,
            C, H, W,
            pool_h, pool_w,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            dy, dx
        );
    } else if (data_format == "NHWC") {
        _AvgPool2dGradNHWC
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(
            nthreads,
            C, H, W,
            pool_h, pool_w,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            dy, dx
        );
    } else {
        LOG(FATAL) << "Unknown DataFormat: " << data_format;
    }
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA