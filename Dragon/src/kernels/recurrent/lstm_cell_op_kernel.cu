#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! LSTMCell <T = float32, Device = CUDA> */

template <typename T>
__global__ void _LSTMCellAct(
    const int               count,
    const int               c_offset,
    const int               x_offset,
    T*                      xact) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int offset = idx % x_offset;
        xact[idx] = offset < c_offset ?
            ((T)1 / ((T)1 + exp(-xact[idx])))
                : tanh(xact[idx]);
    }
}

template <typename T>
__global__ void _LSTMCellGate(
    const int               count,
    const int               hidden_size,
    const int               o_offset, // 2 * hidden_size
    const int               c_offset, // 3 * hidden_size
    const int               x_offset, // 4 * hidden_size
    const T*                cx,
    const T*                xact,
    T*                      c,
    T*                      h) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int n = idx / hidden_size;
        const int offset = idx % hidden_size;
        const T* x  = xact + n * x_offset;
        const T i = x[offset];
        const T f = x[offset + hidden_size];
        const T o = x[offset + o_offset];
        T c_ = x[offset + c_offset];
        c_ = c[idx] = f * cx[idx] + i * c_;
        h[idx] = o * tanh(c_);
    }
}

template <> void LSTMCell<float, CUDAContext>(
    const int               count,
    const int               N,
    const int               C,
    const float*            cx,
    float*                  xact,
    float*                  c,
    float*                  h,
    CUDAContext*            ctx) {
    const int o_offset = 2 * C,
                  c_offset = 3 * C,
                      x_offset = 4 * C;
    _LSTMCellAct<float>
        << < CUDA_BLOCKS(count * 4), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count * 4, c_offset, x_offset, xact);

    _LSTMCellGate<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, C, o_offset, c_offset, x_offset,
            cx, xact, c, h);
}

/*! LSTMCellGrad <T = float32, Device = CUDA> */

template <typename T>
__global__ void _LSTMCellGateGrad(
    const int               count,
    const int               hidden_size,
    const int               o_offset,
    const int               c_offset,
    const int               x_offset,
    const T*                cx,
    const T*                xact,
    const T*                c,
    const T*                dc,
    const T*                dh,
    T*                      dcx,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int n = idx / hidden_size;
        const int offset = idx % hidden_size;
        const T* xact_ = xact + n * x_offset;
        T* dx_ = dx + n * x_offset;
        const T i = xact_[offset];
        const T f = xact_[offset + hidden_size];
        const T o = xact_[offset + o_offset];
        const T g = xact_[offset + c_offset];
        const T tanh_c = tanh(c[idx]);
        const T dcx_sum_term =
            dh[idx] * o * (1 - tanh_c * tanh_c) + dc[idx];
        dcx[idx] = dcx_sum_term * f;
        dx_[offset] = dcx_sum_term * g;
        dx_[offset + hidden_size] = dcx_sum_term * cx[idx];
        dx_[offset + o_offset] = dh[idx] * tanh_c;
        dx_[offset + c_offset] = dcx_sum_term * i;
    }
}

template <typename T>
__global__ void _LSTMCellActGrad(
    const int               count,
    const int               c_offset,
    const int               x_offset,
    const T*                xact,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int offset = idx % x_offset;
        const T val = xact[idx];
        if (offset < c_offset) dx[idx] = dx[idx] * val * (T(1) - val);
        else dx[idx] = dx[idx] * (T(1) - val * val);
    }
}

template <> void LSTMCellGrad<float, CUDAContext>(
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
    CUDAContext*            ctx) {
    const int o_offset = 2 * C, 
                  c_offset = 3 * C,
                      x_offset = 4 * C;
    _LSTMCellGateGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, C, o_offset, c_offset, x_offset,
            cx, xact, c, dc, dh, dcx, dx);

    _LSTMCellActGrad<float>
        << < CUDA_BLOCKS(count * 4), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count * 4, c_offset, x_offset, xact, dx);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA