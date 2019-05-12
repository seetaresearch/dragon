#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/* <T = float32, Device = CPU> */

template<> void PRelu<float, CPUContext>(
    const int               count,
    const int               channels,
    const int               dim,
    const bool              channel_shared,
    const string&           data_format,
    const float*            x,
    const float*            w,
    float*                  y,
    CPUContext*             ctx) {
    if (channel_shared) {
#ifdef WITH_OMP
        #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
        for (int i = 0; i < count; ++i) {
            y[i] = std::max(x[i], 0.f) +
                w[0] * std::min(x[i], 0.f);
        }
    } else {
        if (data_format == "NCHW") {
#ifdef WITH_OMP
            #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
            for (int i = 0; i < count; ++i) {
                int c = (i / dim) % channels;
                y[i] = std::max(x[i], 0.f) +
                    w[c] * std::min(x[i], 0.f);
            }
        } else if (data_format == "NHWC") {
#ifdef WITH_OMP
            #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
            for (int i = 0; i < count; ++i) {
                int c = i % channels;
                y[i] = std::max(x[i], 0.f) +
                    w[c] * std::min(x[i], 0.f);
            }
        } else LOG(FATAL) << "Unknown data format: " << data_format;
    }
}

/* <T = float32, Device = CPU> */

template<> void PReluGrad<float, CPUContext>(
    const int               count,
    const int               channels,
    const int               dim,
    const bool              channel_shared,
    const string&           data_format,
    const float*            dy,
    const float*            x,
    const float*            w,
    float*                  dx,
    CPUContext*             ctx) {
    if (channel_shared) {
#ifdef WITH_OMP
        #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
        for (int i = 0; i < count; ++i) {
            dx[i] = dy[i] * ((x[i] > 0) + w[0] * (x[i] <= 0));
        }
    } else {
        if (data_format == "NCHW") {
#ifdef WITH_OMP
            #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
            for (int i = 0; i < count; ++i) {
                int c = (i / dim) % channels;
                dx[i] = dy[i] * ((x[i] > 0) + w[c] * (x[i] <= 0));
            }
        } else if (data_format == "NHWC") {
#ifdef WITH_OMP
            #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
            for (int i = 0; i < count; ++i) {
                int c = i % channels;
                dx[i] = dy[i] * ((x[i] > 0) + w[c] * (x[i] <= 0));
            }
        } else LOG(FATAL) << "Unknown data format: " << data_format;
    }
}

/* <T = float32, Device = CPU> */

template<> void PReluWGrad<float, CPUContext>(
    const int               rows,
    const int               row_offset,
    const int               channels,
    const int               dim,
    const bool              channel_shared,
    const string&           data_format,
    const float*            dy,
    const float*            x,
    const float*            multiplier,
    float*                  bcast_dw,
    float*                  dw,
    CPUContext*             ctx) {
    const int cdim = channels * dim;
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(cdim))
#endif
    for (int i = 0; i < cdim; ++i) {
        bcast_dw[i] = dy[i] * x[i] * (x[i] <= 0);
        for (int n = 1; n < rows; n++) {
            const int cur_idx = i + n * row_offset;
            bcast_dw[i] += dy[cur_idx] * x[cur_idx] * (x[cur_idx] <= 0);
        }
    }
    if (channel_shared) {
        math::Dot(
            channels * dim,
            bcast_dw, multiplier,
            dw, ctx
        );
    } else {
        if (data_format == "NCHW") {
            math::Gemv(
                CblasNoTrans,
                channels, dim,
                1.f, bcast_dw, multiplier,
                0.f, dw, ctx
            );
        } else if (data_format == "NHWC") {
            math::Gemv(
                CblasTrans,
                dim, channels,
                1.f, bcast_dw, multiplier,
                0.f, dw, ctx
            );
        } else {
            LOG(FATAL) << "Unknown data format: " << data_format;
        }
    }
}

}  // namespace kernel

}  // namepsace dragon