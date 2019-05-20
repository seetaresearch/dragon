#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = float32, Device = CUDA> */

template <typename T>
__global__ void _LSTMCellAct(
    const int               nthreads,
    const int               c_offset,
    const int               x_offset,
    T*                      actx) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const int offset = i % x_offset;
        actx[i] = offset < c_offset ?
            (T(1) / (T(1) + exp(-actx[i])))
                : tanh(actx[i]);
    }
}

template <typename T>
__global__ void _LSTMCellGate(
    const int               nthreads,
    const int               hidden_size,
    const int               o_offset,
    const int               c_offset,
    const int               x_offset,
    const T*                cx,
    const T*                actx,
    T*                      c,
    T*                      h) {
    CUDA_1D_KERNEL_LOOP(idx, nthreads) {
        const int n = idx / hidden_size;
        const int offset = idx % hidden_size;
        const T* actx_ = actx + n * x_offset;
        const T i = actx_[offset];
        const T f = actx_[offset + hidden_size];
        const T o = actx_[offset + o_offset];
        T c_ = actx_[offset + c_offset];
        c_ = c[idx] = f * cx[idx] + i * c_;
        h[idx] = o * tanh(c_);
    }
}

template <> void LSTMCell<float, CUDAContext>(
    const int               N,
    const int               C,
    const float*            cx,
    float*                  actx,
    float*                  c,
    float*                  h,
    CUDAContext*            ctx) {
    auto o_offset = 2 * C, c_offset = 3 * C,
         x_offset = 4 * C, NC = N * C;
    _LSTMCellAct
        <<< CUDA_BLOCKS(NC * 4), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        NC * 4, c_offset, x_offset, actx
    );
    _LSTMCellGate
        <<< CUDA_BLOCKS(NC), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        NC, C, o_offset, c_offset,
        x_offset, cx, actx, c, h
    );
}

/* <T = float32, Device = CUDA> */

template <typename T>
__global__ void _LSTMCellGateGrad(
    const int               nthreads,
    const int               hidden_size,
    const int               o_offset,
    const int               c_offset,
    const int               x_offset,
    const T*                cx,
    const T*                actx,
    const T*                c,
    const T*                dc,
    const T*                dh,
    T*                      dcx,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, nthreads) {
        const int n = idx / hidden_size;
        const int offset = idx % hidden_size;
        const T* actx_ = actx + n * x_offset;
        T* dx_ = dx + n * x_offset;
        const T i = actx_[offset];
        const T f = actx_[offset + hidden_size];
        const T o = actx_[offset + o_offset];
        const T g = actx_[offset + c_offset];
        const T tanh_c = tanh(c[idx]);
        const T dcx_sum_term =
            dh[idx] * o * (T(1) - tanh_c * tanh_c) + dc[idx];
        dcx[idx] = dcx_sum_term * f;
        dx_[offset] = dcx_sum_term * g;
        dx_[offset + hidden_size] = dcx_sum_term * cx[idx];
        dx_[offset + o_offset] = dh[idx] * tanh_c;
        dx_[offset + c_offset] = dcx_sum_term * i;
    }
}

template <typename T>
__global__ void _LSTMCellActGrad(
    const int               nthreads,
    const int               c_offset,
    const int               x_offset,
    const T*                actx,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const T val = actx[i];
        const int offset = i % x_offset;
        if (offset < c_offset) {
            dx[i] = dx[i] * val * (T(1) - val);
        } else {
            dx[i] = dx[i] * (T(1) - val * val);
        }
    }
}

template <> void LSTMCellGrad<float, CUDAContext>(
    const int               N,
    const int               C,
    const float*            cx,
    const float*            actx,
    const float*            c,
    const float*            dc,
    const float*            dh,
    float*                  dcx,
    float*                  dx,
    CUDAContext*            ctx) {
    auto o_offset = 2 * C, c_offset = 3 * C,
         x_offset = 4 * C, NC = N * C;
    _LSTMCellGateGrad
        <<< CUDA_BLOCKS(NC), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        NC, C, o_offset, c_offset, x_offset,
        cx, actx, c, dc, dh, dcx, dx
    );
    _LSTMCellActGrad
        <<< CUDA_BLOCKS(NC * 4), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        NC * 4, c_offset, x_offset, actx, dx
    );
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA