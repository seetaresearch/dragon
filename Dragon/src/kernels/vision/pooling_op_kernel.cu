#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! MAXPooling2d <T = float32, Device = CUDA> */

template<typename T>
__global__ void _MAXPooling2d_NCHW(
    const int               count,
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
    const T*                x,
    int*                    mask,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int pw = idx % pool_w;
        const int ph = (idx / pool_w) % pool_h;
        const int pc = (idx / pool_w / pool_h) % C;
        const int pn = idx / pool_w / pool_h / C;

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
        y[idx] = max_val;
        mask[idx] = max_idx;
    }
}

template<typename T>
__global__ void _MAXPooling2d_NHWC(
    const int               count,
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
    const T*                x,
    int*                    mask,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int pc = idx % C;
        const int pw = (idx / C) % pool_w;
        const int ph = (idx / C / pool_w) % pool_h;
        const int pn = idx / C / pool_w / pool_h;

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
        y[idx] = max_val;
        mask[idx] = max_idx;
    }
}

template<> void MAXPooling2d<float, CUDAContext>(
    const int               count,
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
    if (data_format == "NCHW") {
        _MAXPooling2d_NCHW<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, x, mask, y);
    } else if (data_format == "NHWC") {
        _MAXPooling2d_NHWC<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, x, mask, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/*! AVGPooling2d <T = float32, Device = CUDA> */

template<typename T>
__global__ void _AVGPooling2d_NCHW(
    const int               count,
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
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int pw = idx % pool_w;
        const int ph = (idx / pool_w) % pool_h;
        const int pc = (idx / pool_w / pool_h) % C;
        const int pn = idx / pool_w / pool_h / C;

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
        y[idx] = avg_val / pool_area;
    }
}

template<typename T>
__global__ void _AVGPooling2d_NHWC(
    const int               count,
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
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int pc = idx % C;
        const int pw = (idx / C) % pool_w;
        const int ph = (idx / C / pool_w) % pool_h;
        const int pn = idx / C / pool_w / pool_h;

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

        y[idx] = avg_val / pool_area;
    }
}

template<> void AVGPooling2d<float, CUDAContext>(
    const int               count,
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
    if (data_format == "NCHW") {
        _AVGPooling2d_NCHW<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, x, y);
    } else if (data_format == "NHWC") {
        _AVGPooling2d_NHWC<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, x, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/*! MAXPooling2dGrad <T = float32, Device = CUDA> */

template<typename T>
__global__ void _MAXPooling2dGrad_NCHW(
    const int               count,
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
    const T*                dy,
    const int*              mask,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int w = idx % W;
        const int h = (idx / W) % H;
        const int c = (idx / W / H) % C;
        const int n = idx / W / H / C;

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
        dx[idx] = grad;
    }
}

template<typename T>
__global__ void _MAXPooling2dGrad_NHWC(
    const int               count,
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
    const T*                dy,
    const int*              mask,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int c = idx % C;
        const int w = (idx / C) % W;
        const int h = (idx / C / W) % H;
        const int n = idx / C / W / H;

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
        dx[idx] = grad;
    }
}

template<> void MAXPooling2dGrad<float, CUDAContext>(
    const int               count,
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
    if (data_format == "NCHW") {
        _MAXPooling2dGrad_NCHW<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, dy,mask, dx);
    } else if (data_format == "NHWC") {
        _MAXPooling2dGrad_NHWC<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, dy, mask, dx);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/*! AVGPooling2dGrad <T = float32, Device = CUDA> */

template<typename T>
__global__ void _AVGPooling2dGrad_NCHW(
    const int               count,
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
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int w = idx % W;
        const int h = (idx / W) % H;
        const int c = (idx / W / H) % C;
        const int n = idx / W / H / C;

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
        dx[idx] = grad;
    }
}

template<typename T>
__global__ void _AVGPooling2dGrad_NHWC(
    const int               count,
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
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int c = idx % C;
        const int w = (idx / C) % W;
        const int h = (idx / C / W) % H;
        const int n = idx / C / W / H;

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
        dx[idx] = grad;
    }
}

template<> void AVGPooling2dGrad<float, CUDAContext>(
    const int               count,
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
    if (data_format == "NCHW") {
        _AVGPooling2dGrad_NCHW<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, dy, dx);
    } else if (data_format == "NHWC") {
        _AVGPooling2dGrad_NHWC<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, dy, dx);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA