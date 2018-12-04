#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! LSTMCell <T = float32, Device = CPU> */

template <typename T>
T _SigmoidUnit(T x) { return T(1) / (T(1) + exp(-x)); }

template <> void LSTMCell<float, CPUContext>(
    const int               count,
    const int               N,
    const int               C,
    const float*            cx,
    float*                  xact,
    float*                  c,
    float*                  h,
    CPUContext*             ctx) {
    float i, f, o, c_;
    int f_offset = C, o_offset = 2 * C, c_offset = 3 * C, x_offset = 4 * C;
    for (int n = 0; n < N; ++n) {
        for (int idx = 0; idx < C; ++idx) {
            xact[idx] = i = _SigmoidUnit<float>(xact[idx]);
            xact[idx + f_offset] = f = _SigmoidUnit<float>(xact[idx + f_offset]);
            xact[idx + o_offset] = o = _SigmoidUnit<float>(xact[idx + o_offset]);
            xact[idx + c_offset] = c_ = tanh(xact[idx + c_offset]);
            c_ = c[idx] = f * cx[idx] + i * c_;
            h[idx] = o * tanh(c_);
        }
        cx += C; xact += x_offset; c += C; h += C;
    }
}

/*! LSTMCellGrad <T = float32, Device = CPU> */

template <> void LSTMCellGrad<float, CPUContext>(
    const int               count,
    const int               N,
    const int               C,
    const float*            cx,
    const float*            xact,
    const float*            c,
    const float*            dc,
    const float*            dh,
    float*                  dcx,
    float*                  dx,
    CPUContext*             ctx) {
    float i, f, o, g, tanh_c, dcx_sum_term;
    int f_offset = C,
            o_offset = 2 * C,
                c_offset = 3 * C,
                    x_offset = 4 * C;
    for (int n = 0; n < N; ++n) {
        for (int idx = 0; idx < C; ++idx) {
            i = xact[idx];
            f = xact[idx + f_offset];
            o = xact[idx + o_offset];
            g = xact[idx + c_offset];
            //  BPTT compute the dc_{t-1} at the time of t
            //  dc_{t-1} =  dl / d(h_{t}) * d(h_{t}) / d(c_{t}) * d(c_{t}) / d(c_{t-1})
            //                  + d(c_{t+1}) / d(c_{t}) * d(c_{t}) / d(c_{t-1})
            //           =  (dl / d(h_{t}) * d(h_{t}) / d(c_{t}) + d(c_{t+1}) / d(c_{t}))
            //                  * d(c_{t}) / d(c_{t-1})
            tanh_c = tanh(c[idx]);
            dcx_sum_term = dh[idx] * o * (1 - tanh_c * tanh_c) + dc[idx];
            dcx[idx] = dcx_sum_term * f;
            dx[idx] = dcx_sum_term * g * i * (1 - i);
            dx[idx + f_offset] = dcx_sum_term * cx[idx] * f * (1 - f);
            dx[idx + o_offset] = dh[idx] * tanh_c * o * (1 - o);
            dx[idx + c_offset] = dcx_sum_term * i * (1 - g * g);
        }
        cx += C; xact += x_offset; c += C; dc += C; dh += C;
        dcx += C; dx += x_offset;
    }
}

}  // namespace kernel

}  // namepsace dragon