#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! ROIPool <T = float32, Device = CPU> */

template<> void ROIPool<float, CPUContext>(
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const float*            x,
    const float*            rois,
    int*                    mask,
    float*                  y,
    CPUContext*             ctx) {
    auto X_ofs = H * W, Y_ofs = pool_h * pool_w;
    auto x_ofs = C * X_ofs, y_ofs = C * Y_ofs;

    for (int n = 0; n < num_rois; ++n) {
        auto* roi = rois + n * 5;
        int batch_ind = (int)roi[0];
        auto* Y = y + n * y_ofs;
        auto* M = mask + n * y_ofs;

        if (batch_ind < 0) {
            memset(Y, 0, sizeof(float) * y_ofs);
            memset(M, -1, sizeof(int) * y_ofs);
            continue;
        }

        const float* X = x + batch_ind * x_ofs;
        const int roi_wstart = (int)round(roi[1] * spatial_scale);
        const int roi_hstart = (int)round(roi[2] * spatial_scale);
        const int roi_wend = (int)round(roi[3] * spatial_scale);
        const int roi_hend = (int)round(roi[4] * spatial_scale);

        const int roi_h = std::max(roi_hend - roi_hstart + 1, 1);
        const int roi_w = std::max(roi_wend - roi_wstart + 1, 1);
        const float bin_h = (float)roi_h / (float)pool_h;
        const float bin_w = (float)roi_w / (float)pool_w;

        int xi, yi, hstart, wstart, hend, wend;

        for (int c = 0; c < C; ++c) {
            for (int ph = 0; ph < pool_h; ++ph) {
                for (int pw = 0; pw < pool_w; ++pw) {
                    hstart = (int)floor(bin_h * ph);
                    wstart = (int)floor(bin_w * pw);
                    hend = (int)ceil(bin_h * (ph + 1));
                    wend = (int)ceil(bin_w * (pw + 1));
                    hstart = std::min(std::max(hstart + roi_hstart, 0), H);
                    wstart = std::min(std::max(wstart + roi_wstart, 0), W);
                    hend = std::min(std::max(hend + roi_hstart, 0), H);
                    wend = std::min(std::max(wend + roi_wstart, 0), W);
                    bool empty = (hend == hstart) || (wend == wstart);
                    yi = ph * pool_w + pw;
                    M[yi] = -1;
                    if (empty) Y[yi] = 0;
                    else Y[yi] = -FLT_MAX;
                    for (int h = hstart; h < hend; ++h) {
                        for (int w = wstart; w < wend; ++w) {
                            xi = h * W + w;
                            if (X[xi] > Y[yi]) {
                                M[yi] = xi;
                                Y[yi] = X[xi];
                            }
                        }  // End w
                    }  // End h
                }  // End pw
            }  // End ph
            // Offset according to C
            X += X_ofs;
            Y += Y_ofs;
            M += Y_ofs;
        }  // End c
    }  // End n
}

/*! ROIPool <T = float16, Device = CPU> */

template<> void ROIPool<float16, CPUContext>(
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const float16*          x,
    const float*            rois,
    int*                    mask,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

/*! ROIPoolGrad <T = float32, Device = CPU> */

template<> void ROIPoolGrad<float, CPUContext>(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const float*            dy,
    const float*            rois,
    const int*              mask,
    float*                  dx,
    CPUContext*             ctx) {
    NOT_IMPLEMENTED;
}

}  // namespace kernel

}  // namepsace dragon