#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! ROIAlign <T = float32, Device = CPU> */

template <typename T>
T _ROIAlignIntp(
    const T*                X,
    const int               H,
    const int               W,
    T                       y,
    T                       x) {
    if (y < -1.0 || y > H || x < -1.0 || x > W) return 0;
    if (y <= 0) y = 0;
    if (x <= 0) x = 0;

    int y_low = (int)y;
    int x_low = (int)x;
    int y_high;
    int x_high;

    if (y_low >= H - 1) {
        y_high = y_low = H - 1;
        y = (T)y_low;
    } else {
        y_high = y_low + 1;
    }

    if (x_low >= W - 1) {
        x_high = x_low = W - 1;
        x = (T)x_low;
    } else {
        x_high = x_low + 1;
    }

    T ly = y - y_low;
    T lx = x - x_low;
    T hy = (T)1 - ly, hx = (T)1 - lx;
    T v1 = X[y_low * W + x_low];
    T v2 = X[y_low * W + x_high];
    T v3 = X[y_high * H + x_low];
    T v4 = X[y_high * H + x_high];
    T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

template<> void ROIAlign<float, CPUContext>(
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const int               sampling_ratio,
    const float*            xdata,
    const float*            rois,
    float*                  ydata,
    CPUContext*             ctx) {
    auto X_ofs = H * W, Y_ofs = pool_h * pool_w;
    auto x_ofs = C * X_ofs, y_ofs = C * Y_ofs;

    for (int n = 0; n < num_rois; ++n) {
        auto* roi = rois + n * 5;
        int batch_ind = (int)roi[0];
        auto* Y = ydata + n * y_ofs;

        if (batch_ind < 0) {
            memset(Y, 0, sizeof(float) * y_ofs);
            continue;
        }

        const float* X = xdata + batch_ind * x_ofs;
        float roi_wstart = roi[1] * spatial_scale;
        float roi_hstart = roi[2] * spatial_scale;
        float roi_wend = roi[3] * spatial_scale;
        float roi_hend = roi[4] * spatial_scale;

        float roi_w = std::max(roi_wend - roi_wstart, 1.f);
        float roi_h = std::max(roi_hend - roi_hstart, 1.f);
        float bin_h = roi_h / (float)pool_h;
        float bin_w = roi_w / (float)pool_w;

        const int grid_h = (sampling_ratio > 0) ?
            sampling_ratio : (int)ceil(roi_h / pool_h);
        const int grid_w = (sampling_ratio > 0) ?
            sampling_ratio : (int)ceil(roi_w / pool_w);

        float intp_val; int yi;

        for (int c = 0; c < C; ++c) {
            for (int ph = 0; ph < pool_h; ++ph) {
            for (int pw = 0; pw < pool_w; ++pw) {
                intp_val = 0.f; yi = ph * pool_w + pw;
                for (int iy = 0; iy < grid_h; iy++) {
                    const float y = roi_hstart + ph * bin_h +
                        (float)(iy + .5f) * bin_h / (float)(grid_h);
                    for (int ix = 0; ix < grid_w; ix++) {
                        const float x = roi_wstart + pw * bin_w +
                            (float)(ix + .5f) * bin_w / (float)(grid_w);
                        intp_val += _ROIAlignIntp(X, H, W, y, x);
                    }  // End ix
                }  // End iy
                Y[yi] = intp_val / float(grid_h * grid_w);
            }}  // End ph && pw
            // Offset according to C
            X += X_ofs;
            Y += Y_ofs;
        }  // End c
    }  // End n
}

/*! ROIAlign <T = float16, Device = CPU> */

template<> void ROIAlign<float16, CPUContext>(
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const int               sampling_ratio,
    const float16*          x,
    const float*            rois,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

/*! ROIAlignGrad <T = float32, Device = CPU> */

template<> void ROIAlignGrad<float, CPUContext>(
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
    CPUContext*             ctx) {
    NOT_IMPLEMENTED;
}

}  // namespace kernel

}  // namepsace dragon