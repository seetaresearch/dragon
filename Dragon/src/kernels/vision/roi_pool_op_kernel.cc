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
    const int64_t X_offset = H * W, Y_offset = pool_h * pool_w;
    const int64_t x_offset = C * X_offset, y_offset = C * Y_offset;

    for (int n = 0; n < num_rois; ++n) {
        auto* R = rois + n * 5;
        int roi_batch_ind = (int)R[0];
        auto* Y = y + n * y_offset;
        auto* M = mask + n * y_offset;

        if (roi_batch_ind < 0) {
            memset(Y, 0, sizeof(float) * y_offset);
            memset(M, -1, sizeof(int) * y_offset);
            continue;
        }

        int x1 = (int)round(R[1] * spatial_scale);
        int y1 = (int)round(R[2] * spatial_scale);
        int x2 = (int)round(R[3] * spatial_scale);
        int y2 = (int)round(R[4] * spatial_scale);
        int roi_height = std::max(y2 - y1 + 1, 1);
        int roi_width = std::max(x2 - x1 + 1, 1);
        const float unit_h = (float)roi_height / (float)pool_h;
        const float unit_w = (float)roi_width / (float)pool_w;
        const float* X = x + roi_batch_ind * x_offset;
        
        for (int c = 0; c < C; ++c) {
            for (int ph = 0; ph < pool_h; ++ph) {
                for (int pw = 0; pw < pool_w; ++pw) {
                    int start_h = (int)floor(unit_h * ph);
                    int start_w = (int)floor(unit_w * pw);
                    int end_h = (int)ceil(unit_h*(ph + 1));
                    int end_w = (int)ceil(unit_w*(pw + 1));
                    start_h = std::max(start_h + y1, 0);
                    start_w = std::max(start_w + x1, 0);
                    end_h = std::max(end_h + y1, 0);
                    end_w = std::max(end_w + x1, 0);
                    start_h = std::min(start_h, H);
                    start_w = std::min(start_w, W);
                    end_h = std::min(end_h, H);
                    end_w = std::min(end_w, W);
                    bool is_empty = (end_h == start_h) || (end_w == start_w);
                    const int pool_idx = ph * pool_w + pw;
                    M[pool_idx] = -1;
                    if (is_empty) Y[pool_idx] = 0;
                    else Y[pool_idx] = -FLT_MAX;
                    for (int h = start_h; h < end_h; ++h) {
                        for (int w = start_w; w < end_w; ++w) {
                            const int idx = h * W + w;
                            if (X[idx] > Y[pool_idx]) {
                                M[pool_idx] = idx;
                                Y[pool_idx] = X[idx];
                            }
                        }  // End w
                    }  // End h
                }  // End pw
            }  // End ph
            // Offset according to C
            X += X_offset;
            Y += Y_offset;
            M += Y_offset;
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