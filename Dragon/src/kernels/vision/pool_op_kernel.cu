#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! MAXPool2d <T = float32, Device = CUDA> */

template<typename T>
__global__ void _MAXPool2d_NCHW(
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
    CUDA_1D_KERNEL_LOOP(y_idx, nthreads) {
        const int pw = y_idx % pool_w;
        const int ph = (y_idx / pool_w) % pool_h;
        const int pc = (y_idx / pool_w / pool_h) % C;
        const int pn = y_idx / pool_w / pool_h / C;

        int start_h = ph * stride_h - pad_h;
        int start_w = pw * stride_w - pad_w;
        const int end_h = min(start_h + kernel_h, H);
        const int end_w = min(start_w + kernel_w, W);

        start_h = max(start_h, 0);
        start_w = max(start_w, 0);

        T max_val = -FLT_MAX;
        int max_idx = -1;
        const T* x_ptr = x + (pn * C + pc) * H * W;

        for (int h = start_h; h < end_h; ++h) {
            for (int w = start_w; w < end_w; ++w) {
                if (x_ptr[h * W + w] > max_val) {
                    max_idx = h * W + w;
                    max_val = x_ptr[max_idx];
                }
            }
        }
        y[y_idx] = max_val;
        mask[y_idx] = max_idx;
    }
}

template<typename T>
__global__ void _MAXPool2d_NHWC(
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
    CUDA_1D_KERNEL_LOOP(y_idx, nthreads) {
        const int pc = y_idx % C;
        const int pw = (y_idx / C) % pool_w;
        const int ph = (y_idx / C / pool_w) % pool_h;
        const int pn = y_idx / C / pool_w / pool_h;

        int start_h = ph * stride_h - pad_h;
        int start_w = pw * stride_w - pad_w;
        const int end_h = min(start_h + kernel_h, H);
        const int end_w = min(start_w + kernel_w, W);

        start_h = max(start_h, 0);
        start_w = max(start_w, 0);

        T max_val = -FLT_MAX;
        int max_idx = -1;
        for (int h = start_h; h < end_h; ++h) {
            for (int w = start_w; w < end_w; ++w) {
                const int x_idx = ((pn * H + h) * W + w) * C + pc;
                if (x[x_idx] > max_val) {
                    max_idx = x_idx;
                    max_val = x[max_idx];
                }
            }
        }
        y[y_idx] = max_val;
        mask[y_idx] = max_idx;
    }
}

template<> void MAXPool2d<float, CUDAContext>(
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
        _MAXPool2d_NCHW<float>
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (nthreads, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, x, mask, y);
    } else if (data_format == "NHWC") {
        _MAXPool2d_NHWC<float>
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (nthreads, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, x, mask, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/*! AVGPool2d <T = float32, Device = CUDA> */

template<typename T>
__global__ void _AVGPool2d_NCHW(
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
    CUDA_1D_KERNEL_LOOP(y_idx, nthreads) {
        const int pw = y_idx % pool_w;
        const int ph = (y_idx / pool_w) % pool_h;
        const int pc = (y_idx / pool_w / pool_h) % C;
        const int pn = y_idx / pool_w / pool_h / C;

        int start_h = ph * stride_h - pad_h;
        int start_w = pw * stride_w - pad_w;
        int end_h = min(start_h + kernel_h, H + pad_h);
        int end_w = min(start_w + kernel_w, W + pad_w);

        start_h = max(start_h, 0);
        start_w = max(start_w, 0);
        end_h = min(end_h, H);
        end_w = min(end_w, W);

        const T* x_ptr = x + (pn * C + pc) * H * W;
        const int pool_area = (end_h - start_h) * (end_w - start_w);
        T avg_val = 0;

        for (int h = start_h; h < end_h; ++h) {
            for (int w = start_w; w < end_w; ++w) {
                avg_val += x_ptr[h * W + w];
            }
        }
        y[y_idx] = avg_val / pool_area;
    }
}

template<typename T>
__global__ void _AVGPool2d_NHWC(
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
    CUDA_1D_KERNEL_LOOP(y_idx, nthreads) {
        const int pc = y_idx % C;
        const int pw = (y_idx / C) % pool_w;
        const int ph = (y_idx / C / pool_w) % pool_h;
        const int pn = y_idx / C / pool_w / pool_h;

        int start_h = ph * stride_h - pad_h;
        int start_w = pw * stride_w - pad_w;
        int end_h = min(start_h + kernel_h, H + pad_h);
        int end_w = min(start_w + kernel_w, W + pad_w);

        start_h = max(start_h, 0);
        start_w = max(start_w, 0);
        end_h = min(end_h, H);
        end_w = min(end_w, W);

        const int pool_area = (end_h - start_h) * (end_w - start_w);
        T avg_val = 0;

        for (int h = start_h; h < end_h; ++h) 
            for (int w = start_w; w < end_w; ++w)
                avg_val += x[((pn * H + h) * W + w) * C + pc];

        y[y_idx] = avg_val / pool_area;
    }
}

template<> void AVGPool2d<float, CUDAContext>(
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
        _AVGPool2d_NCHW<float>
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (nthreads, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, x, y);
    } else if (data_format == "NHWC") {
        _AVGPool2d_NHWC<float>
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (nthreads, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, x, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/*! MAXPool2dGrad <T = float32, Device = CUDA> */

template<typename T>
__global__ void _MAXPool2dGrad_NCHW(
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
    CUDA_1D_KERNEL_LOOP(x_idx, nthreads) {
        const int w = x_idx % W;
        const int h = (x_idx / W) % H;
        const int c = (x_idx / W / H) % C;
        const int n = x_idx / W / H / C;

        // Allow overlapping
        const int start_ph = (h + pad_h < kernel_h) ?
            0 : (h + pad_h - kernel_h) / stride_h + 1;
        const int start_pw = (w + pad_w < kernel_w) ? 
            0 : (w + pad_w - kernel_w) / stride_w + 1;
        // Allow clip
        const int end_ph = min((h + pad_h) / stride_h + 1, pool_h);
        const int end_pw = min((w + pad_w) / stride_w + 1, pool_w);

        T grad = 0;
        const int offset = (n * C + c) * pool_h * pool_w;
        const T* dy_ptr = dy + offset;
        const int* mask_ptr = mask + offset;

        for (int ph = start_ph; ph < end_ph; ++ph) {
            for (int pw = start_pw; pw < end_pw; ++pw) {
                if (mask_ptr[ph * pool_w + pw] == (h * W + w)) {
                    grad += dy_ptr[ph * pool_w + pw];
                }
            }
        }
        dx[x_idx] = grad;
    }
}

template<typename T>
__global__ void _MAXPool2dGrad_NHWC(
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
    CUDA_1D_KERNEL_LOOP(x_idx, nthreads) {
        const int c = x_idx % C;
        const int w = (x_idx / C) % W;
        const int h = (x_idx / C / W) % H;
        const int n = x_idx / C / W / H;

        // Allow overlapping
        const int start_ph = (h + pad_h < kernel_h) ?
            0 : (h + pad_h - kernel_h) / stride_h + 1;
        const int start_pw = (w + pad_w < kernel_w) ?
            0 : (w + pad_w - kernel_w) / stride_w + 1;
        // Allow clip
        const int end_ph = min((h + pad_h) / stride_h + 1, pool_h);
        const int end_pw = min((w + pad_w) / stride_w + 1, pool_w);

        T grad = 0;
        for (int ph = start_ph; ph < end_ph; ++ph) {
            for (int pw = start_pw; pw < end_pw; ++pw) {
                const int x_idx = ((n * H + h) * W + w) * C + c;
                const int y_idx = ((n * pool_h + ph) * pool_w + pw) * C + c;
                if (mask[y_idx] == x_idx) grad += dy[y_idx];
            }
        }
        dx[x_idx] = grad;
    }
}

template<> void MAXPool2dGrad<float, CUDAContext>(
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
        _MAXPool2dGrad_NCHW<float>
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (nthreads, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, dy,mask, dx);
    } else if (data_format == "NHWC") {
        _MAXPool2dGrad_NHWC<float>
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (nthreads, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, dy, mask, dx);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/*! AVGPool2dGrad <T = float32, Device = CUDA> */

template<typename T>
__global__ void _AVGPool2dGrad_NCHW(
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
    CUDA_1D_KERNEL_LOOP(x_idx, nthreads) {
        const int w = x_idx % W;
        const int h = (x_idx / W) % H;
        const int c = (x_idx / W / H) % C;
        const int n = x_idx / W / H / C;

        const int start_ph = (h + pad_h < kernel_h) ?
            0 : (h + pad_h - kernel_h) / stride_h + 1;
        const int start_pw = (w + pad_w < kernel_w) ?
            0 : (w + pad_w - kernel_w) / stride_w + 1;
        const int end_ph = min(h / stride_h + 1, pool_h);
        const int end_pw = min(w / stride_w + 1, pool_w);

        T grad = 0;
        const T* dy_ptr = dy + (n * C + c) * pool_h * pool_w;
        for (int ph = start_ph; ph < end_ph; ++ph) {
            for (int pw = start_pw; pw < end_pw; ++pw) {
                int start_h = ph * stride_h - pad_h;
                int start_w = pw * stride_w - pad_w;
                int end_h = min(start_h + kernel_h, H + pad_h);
                int end_w = min(start_w + kernel_w, W + pad_w);
                int pool_area = (end_h - start_h) * (end_w - start_w);
                grad += (dy_ptr[ph * pool_w + pw] / pool_area);
            }
        }
        dx[x_idx] = grad;
    }
}

template<typename T>
__global__ void _AVGPool2dGrad_NHWC(
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
    CUDA_1D_KERNEL_LOOP(x_idx, nthreads) {
        const int c = x_idx % C;
        const int w = (x_idx / C) % W;
        const int h = (x_idx / C / W) % H;
        const int n = x_idx / C / W / H;

        const int start_ph = (h + pad_h < kernel_h) ?
            0 : (h + pad_h - kernel_h) / stride_h + 1;
        const int start_pw = (w + pad_w < kernel_w) ?
            0 : (w + pad_w - kernel_w) / stride_w + 1;
        const int end_ph = min(h / stride_h + 1, pool_h);
        const int end_pw = min(w / stride_w + 1, pool_w);

        T grad = 0;
        for (int ph = start_ph; ph < end_ph; ++ph) {
            for (int pw = start_pw; pw < end_pw; ++pw) {
                int start_h = ph * stride_h - pad_h;
                int start_w = pw * stride_w - pad_w;
                int end_h = min(start_h + kernel_h, H + pad_h);
                int end_w = min(start_w + kernel_w, W + pad_w);
                int pool_area = (end_h - start_h) * (end_w - start_w);
                const int y_idx = ((n * pool_h + ph) * pool_w + pw) * C + c;
                grad += (dy[y_idx] / pool_area);
            }
        }
        dx[x_idx] = grad;
    }
}

template<> void AVGPool2dGrad<float, CUDAContext>(
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
        _AVGPool2dGrad_NCHW<float>
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (nthreads, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, dy, dx);
    } else if (data_format == "NHWC") {
        _AVGPool2dGrad_NHWC<float>
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (nthreads, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, dy, dx);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA