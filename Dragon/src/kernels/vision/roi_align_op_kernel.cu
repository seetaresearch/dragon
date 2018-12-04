#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! ROIAlign <T = float32, Device = CUDA> */

template <typename T>
__device__ T _ROIAlignInterpolate(
    const T*                Xdata,
    const int               height,
    const int               width,
    T                       y,
    T                       x) {
    if (y < -1.0 || y > height || x < -1.0 || x > width) return 0;
    if (y <= 0) y = 0;
    if (x <= 0) x = 0;

    int y_low = (int)y;
    int x_low = (int)x;
    int y_high;
    int x_high;

    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (T)y_low;
    } else {
        y_high = y_low + 1;
    }

    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (T)x_low;
    } else {
        x_high = x_low + 1;
    }

    T ly = y - y_low;
    T lx = x - x_low;
    T hy = 1. - ly, hx = 1. - lx;
    T v1 = Xdata[y_low * width + x_low];
    T v2 = Xdata[y_low * width + x_high];
    T v3 = Xdata[y_high * width + x_low];
    T v4 = Xdata[y_high * width + x_high];
    T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

template <typename T>
__global__ void _ROIAlign(
    const int               count,
    const float             spatial_scale,
    const int               channels,
    const int               height,
    const int               width,
    const int               pool_h,
    const int               pool_w,
    const int               sampling_ratio,
    const T*                Xdata,
    const float*            rois,
    T*                      Ydata) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        int pw = idx % pool_w;
        int ph = (idx / pool_w) % pool_h;
        int c = (idx / pool_w / pool_h) % channels;
        int n = idx / pool_w / pool_h / channels;

        const T* offset_rois = rois + n * 5;
        int roi_batch_ind = offset_rois[0];

        if (roi_batch_ind < 0) {
            Ydata[idx] = 0;
            continue;
        }

        T roi_start_w = offset_rois[1] * spatial_scale;
        T roi_start_h = offset_rois[2] * spatial_scale;
        T roi_end_w = offset_rois[3] * spatial_scale;
        T roi_end_h = offset_rois[4] * spatial_scale;

        T roi_width = max(roi_end_w - roi_start_w, (T)1.);
        T roi_height = max(roi_end_h - roi_start_h, (T)1.);
        T bin_size_h = (T)roi_height / (T)pool_h;
        T bin_size_w = (T)roi_width / (T)pool_w;

        const T* offset_Xdata = Xdata +(roi_batch_ind * channels + c) * height * width;

        int roi_bin_grid_h = (sampling_ratio > 0) ?
            sampling_ratio : ceil(roi_height / pool_h);
        int roi_bin_grid_w = (sampling_ratio > 0) ? 
            sampling_ratio : ceil(roi_width / pool_w);
        
        T output_val = 0.;
        const T num_bin_grids = roi_bin_grid_h * roi_bin_grid_w;

        for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            const T y = roi_start_h + ph * bin_size_h +
                static_cast<T>(iy + .5f) * bin_size_h /
                    static_cast<T>(roi_bin_grid_h);
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                const T x = roi_start_w + pw * bin_size_w + 
                    static_cast<T>(ix + .5f) * bin_size_w /
                        static_cast<T>(roi_bin_grid_w);
                output_val += _ROIAlignInterpolate(
                    offset_Xdata, height, width, y, x);
            }
        }
        output_val /= num_bin_grids;
        Ydata[idx] = output_val;
    }
}

template<> void ROIAlign<float, CUDAContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const int               sampling_ratio,
    const float*            x,
    const float*            rois,
    float*                  y,
    CUDAContext*            ctx) {
    _ROIAlign<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, spatial_scale, C, H, W, pool_h, pool_w,
            sampling_ratio, x, rois, y);
}

/*! ROIAlignGrad <T = float32, Device = CUDA> */

template <typename T>
__device__ void _ROIAlignInterpolateGrad(
    const int               height,
    const int               width,
    T                       y,
    T                       x,
    T&                      w1,
    T&                      w2,
    T&                      w3,
    T&                      w4,
    int&                    x_low,
    int&                    x_high,
    int&                    y_low,
    int&                    y_high) {
    if (y < -1.0 || y > height ||
            x < -1.0 || x > width) {
        w1 = w2 = w3 = w4 = 0.;
        x_low = x_high = y_low = y_high = -1;
        return;
    }

    if (y <= 0) y = 0;
    if (x <= 0) x = 0;

    y_low = (int)y;
    x_low = (int)x;

    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (T)y_low;
    } else {
        y_high = y_low + 1;
    }

    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (T)x_low;
    } else {
        x_high = x_low + 1;
    }

    T ly = y - y_low;
    T lx = x - x_low;
    T hy = 1. - ly, hx = 1. - lx;
    w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    return;
}

template <typename T>
__global__ void _ROIAlignGrad(
    const int               count,
    const int               num_rois,
    const T                 spatial_scale,
    const int               channels,
    const int               height,
    const int               width,
    const int               pool_h,
    const int               pool_w,
    const int               sampling_ratio,
    const T*                dYdata,
    const T*                rois,
    T*                      dXdata) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        int pw = idx % pool_w;
        int ph = (idx / pool_w) % pool_h;
        int c = (idx / pool_w / pool_h) % channels;
        int n = idx / pool_w / pool_h / channels;

        const T* offset_rois = rois + n * 5;
        int roi_batch_ind = offset_rois[0];

        if (roi_batch_ind < 0) continue;

        T roi_start_w = offset_rois[1] * spatial_scale;
        T roi_start_h = offset_rois[2] * spatial_scale;
        T roi_end_w = offset_rois[3] * spatial_scale;
        T roi_end_h = offset_rois[4] * spatial_scale;

        T roi_width = max(roi_end_w - roi_start_w, (T)1.);
        T roi_height = max(roi_end_h - roi_start_h, (T)1.);
        T bin_size_h = (T)roi_height / (T)pool_h;
        T bin_size_w = (T)roi_width / (T)pool_w;

        T* offset_dXdata = dXdata + 
            (roi_batch_ind * channels + c) * height * width;

        int y_offset = (n * channels + c) * pool_h * pool_w;
        const T* offset_dYdata = dYdata + y_offset;
        const T dYdata_this_bin = offset_dYdata[ph * pool_w + pw];

        int roi_bin_grid_h = (sampling_ratio > 0) ?
            sampling_ratio : ceil(roi_height / pool_h);
        int roi_bin_grid_w = (sampling_ratio > 0) ?
            sampling_ratio : ceil(roi_width / pool_w);

        const T num_bin_grids = roi_bin_grid_h * roi_bin_grid_w;

        for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            const T y = roi_start_h + ph * bin_size_h +
                static_cast<T>(iy + .5f) * bin_size_h /
                    static_cast<T>(roi_bin_grid_h);
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                const T x = roi_start_w + pw * bin_size_w +
                    static_cast<T>(ix + .5f) * bin_size_w /
                        static_cast<T>(roi_bin_grid_w);

                T w1, w2, w3, w4;
                int x_low, x_high, y_low, y_high;

                _ROIAlignInterpolateGrad(
                    height, width, y, x, w1, w2, w3, w4,
                        x_low, x_high, y_low, y_high);

                T g1 = dYdata_this_bin * w1 / num_bin_grids;
                T g2 = dYdata_this_bin * w2 / num_bin_grids;
                T g3 = dYdata_this_bin * w3 / num_bin_grids;
                T g4 = dYdata_this_bin * w4 / num_bin_grids;

                if (x_low >= 0 && x_high >= 0 
                        && y_low >= 0 && y_high >= 0) {
                    atomicAdd(
                        offset_dXdata + y_low * width + x_low,
                        static_cast<T>(g1));
                    atomicAdd(
                        offset_dXdata + y_low * width + x_high,
                        static_cast<T>(g2));
                    atomicAdd(
                        offset_dXdata + y_high * width + x_low,
                        static_cast<T>(g3));
                    atomicAdd(
                        offset_dXdata + y_high * width + x_high,
                        static_cast<T>(g4));
                }
            }
        }
    }
}

template<> void ROIAlignGrad<float, CUDAContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const int               sampling_ratio,
    const float*            dy,
    const float*            rois,
    float*                  dx,
    CUDAContext*            ctx) {
    _ROIAlignGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, num_rois, spatial_scale, C, H, W,
            pool_h, pool_w, sampling_ratio, dy, rois, dx);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA