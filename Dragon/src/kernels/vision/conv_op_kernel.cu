#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! Im2Col2d <T = float32, Device = CUDA> */

template<typename T>
__global__ void _Im2Col2d_NCHW(
    const int               count,
    const int               H,
    const int               W,
    const int               col_h,
    const int               col_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const T*                im,
    T*                      col) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int w = idx % col_w;
        const int h_idx = idx / col_w;
        const int h = h_idx % col_h;
        const int im_c = h_idx / col_h;
        const int c = im_c * kernel_h * kernel_w;

        const int im_h_off = h * stride_h - pad_h;
        const int im_w_off = w * stride_w - pad_w;

        T* col_ptr = col;
        col_ptr += ((c * col_h + h) * col_w + w);

        const T* im_ptr = im;
        im_ptr += ((im_c * H + im_h_off) * W + im_w_off);

        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                const int im_h = kh * dilation_h + im_h_off;
                const int im_w = kw * dilation_w + im_w_off;
                *col_ptr = (im_h >= 0 && im_w >= 0 && im_h < H && im_w < W) ? 
                    im_ptr[kh * dilation_h * W + kw * dilation_w] : 0;
                col_ptr += (col_h * col_w);
            }
        }
    }
}

template<typename T>
__global__ void _Im2Col2d_NHWC(
    const int               count,
    const int               C,
    const int               H,
    const int               W,
    const int               col_h,
    const int               col_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const T*                im,
    T*                      col) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int c = idx % C;
        const int w = (idx / C) % col_w;
        const int h = idx / C / col_w;
      
        const int im_h_off = h * stride_h - pad_h;
        const int im_w_off = w * stride_w - pad_w;
        const int base_col_idx = (h * col_w) + w;

        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                const int im_h = kh * dilation_h + im_h_off;
                const int im_w = kw * dilation_w + im_w_off;
                const int col_idx = (
                    ((base_col_idx * kernel_h + kh) * kernel_w + kw) * C + c
                );
                col[col_idx] = (im_h >= 0 && im_w >= 0 &&
                    im_h < H && im_w < W) ? im[(im_h * W + im_w) * C + c] : 0;
            }
        }
    }
}

template <> void Im2Col2d<float, CUDAContext>(
    const int               C,
    const int               H,
    const int               W,
    const int               col_h,
    const int               col_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const string&           data_format,
    const float*            im,
    float*                  col,
    CUDAContext*            ctx) {
    if (data_format == "NCHW") {
         const int count = (C * col_h * col_w);
         _Im2Col2d_NCHW<float> 
             << < CUDA_BLOCKS(count), CUDA_THREADS,
                  0, ctx->cuda_stream() >> >
             (count, H, W, col_h, col_w, kernel_h, kernel_w,
                 stride_h, stride_w, pad_h, pad_w,
                    dilation_h, dilation_w, im, col);
    } else if (data_format == "NHWC") {
         const int count = (col_h * col_w * C);
         _Im2Col2d_NHWC<float> 
             << < CUDA_BLOCKS(count), CUDA_THREADS,
                  0, ctx->cuda_stream() >> >
             (count, C, H, W, col_h, col_w, kernel_h, kernel_w,
                 stride_h, stride_w, pad_h, pad_w,
                    dilation_h, dilation_w, im, col);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/*! Col2Im2d <T = float32, Device = CUDA> */

template<typename T>
__global__ void _Col2Im2d_NCHW(
    const int               count,
    const int               H,
    const int               W,
    const int               col_h,
    const int               col_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const T*                col,
    T*                      im) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        T val = 0;
        const int im_w = idx % W + pad_w;
        const int im_h = (idx / W) % H + pad_h;
        const int im_c = idx / W / H;
        const int ex_kernel_h = (kernel_h - 1) * dilation_h + 1;
        const int ex_kernel_w = (kernel_w - 1) * dilation_w + 1;

        // Redundant pixels will be ignored when conv
        // Note to clip them by min(x,col_w)
        const int w_start = (im_w < ex_kernel_w) ?
            0 : (im_w - ex_kernel_w) / stride_w + 1;
        const int w_end = min(im_w / stride_w + 1, col_w);
        const int h_start = (im_h < ex_kernel_h) ? 
            0 : (im_h - ex_kernel_h) / stride_h + 1;
        const int h_end = min(im_h / stride_h + 1, col_h);

        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                int kh_off = (im_h - h * stride_h);
                int kw_off = (im_w - w * stride_w);
                // Only the serval im pixels used in dilated-conv
                // Ignore the corresponding col pixels
                if (kh_off % dilation_h == 0 && kw_off % dilation_w == 0) {
                    kh_off /= dilation_h;
                    kw_off /= dilation_w;
                    const int col_idx = ((
                        (im_c * kernel_h + kh_off) * kernel_w + kw_off) * col_h + h
                    ) * col_w + w;
                    val += col[col_idx];
                }
            }
        }
        im[idx] = val;
    }
}

template<typename T>
__global__ void _Col2Im2d_NHWC(
    const int               count,
    const int               C,
    const int               H,
    const int               W,
    const int               col_h,
    const int               col_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const T*                col,
    T*                      im) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        T val = 0;
        const int im_c = idx % C;
        const int im_w = (idx / C) % W + pad_w;
        const int im_h = (idx / C / W) + pad_h;
        const int ex_kernel_h = (kernel_h - 1) * dilation_h + 1;
        const int ex_kernel_w = (kernel_w - 1) * dilation_w + 1;

        // Redundant pixels will be ignored when conv
        // Note to clip them by min(x,col_w)
        const int w_start = (im_w < ex_kernel_w) ?
            0 : (im_w - ex_kernel_w) / stride_w + 1;
        const int w_end = min(im_w / stride_w + 1, col_w);
        const int h_start = (im_h < ex_kernel_h) ?
            0 : (im_h - ex_kernel_h) / stride_h + 1;
        const int h_end = min(im_h / stride_h + 1, col_h);

        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                int kh_off = (im_h - h * stride_h);
                int kw_off = (im_w - w * stride_w);
                // Only the serval im pixels used in dilated-conv
                // Ignore the corresponding col pixels
                if (kh_off % dilation_h == 0 && kw_off % dilation_w == 0) {
                    kh_off /= dilation_h;
                    kw_off /= dilation_w;
                    const int col_idx = (
                        ((h * col_w + w) * kernel_h + kh_off) * kernel_w + kw_off
                    ) * C + im_c;
                    val += col[col_idx];
                }
            }
        }
        im[idx] = val;
    }
}

template <> void Col2Im2d<float, CUDAContext>(
    const int               C,
    const int               H,
    const int               W,
    const int               col_h,
    const int               col_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const string&           data_format,
    const float*            col,
    float*                  im,
    CUDAContext*            ctx) {
    if (data_format == "NCHW") {
         const int count = (C * H * W);
         _Col2Im2d_NCHW<float>
             << < CUDA_BLOCKS(count), CUDA_THREADS,
                  0, ctx->cuda_stream() >> >
             (count, H, W, col_h, col_w, kernel_h, kernel_w,
                 stride_h, stride_w, pad_h, pad_w,
                    dilation_h, dilation_w, col, im);
    } else if (data_format == "NHWC") {
         const int count = (H * W * C);
         _Col2Im2d_NHWC<float>
             << < CUDA_BLOCKS(count), CUDA_THREADS,
                  0, ctx->cuda_stream() >> >
             (count, C, H, W, col_h, col_w, kernel_h, kernel_w,
                 stride_h, stride_w, pad_h, pad_w,
                    dilation_h, dilation_w, col, im);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA