#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
__global__ void _LSTMAct(const int NxCx4, const int C, T* x) {
  const int Cx3 = C * 3, Cx4 = C * 4;
  CUDA_1D_KERNEL_LOOP(i, NxCx4) {
    x[i] = i % Cx4 < Cx3 ? T(1) / (T(1) + exp(-x[i])) : tanh(x[i]);
  }
}

template <typename T>
__global__ void
_LSTMGate(const int NxC, const int C, const T* c_prev, const T* x, T* c, T* h) {
  const int Cx2 = C * 2, Cx3 = C * 3, Cx4 = C * 4;
  CUDA_1D_KERNEL_LOOP(index, NxC) {
    const int i = index / C;
    const int j = index % C;
    const T* offset_x = x + i * Cx4;
    const T i_val = offset_x[j];
    const T f_val = offset_x[j + C];
    const T o_val = offset_x[j + Cx2];
    T val = offset_x[j + Cx3];
    val = c[index] = f_val * c_prev[index] + i_val * val;
    h[index] = o_val * tanh(val);
  }
}

template <typename T>
__global__ void _LSTMGateGrad(
    const int NxC,
    const int C,
    const T* c_prev,
    const T* x,
    const T* c,
    const T* dc,
    const T* dh,
    T* dc_prev,
    T* dx) {
  const int Cx2 = C * 2, Cx3 = C * 3, Cx4 = C * 4;
  CUDA_1D_KERNEL_LOOP(index, NxC) {
    const int i = index / C;
    const int j = index % C;
    const T* offset_x = x + i * Cx4;
    T* offset_dx = dx + i * i * Cx4;
    const T i_val = offset_x[j];
    const T f_val = offset_x[j + C];
    const T o_val = offset_x[j + Cx2];
    const T g_val = offset_x[j + Cx3];
    const T tanh_c_val = tanh(c[index]);
    const T grad_val =
        dh[index] * o_val * (T(1) - tanh_c_val * tanh_c_val) + dc[index];
    dc_prev[index] = grad_val * f_val;
    offset_dx[j] = grad_val * g_val;
    offset_dx[j + C] = grad_val * c_prev[index];
    offset_dx[j + Cx2] = dh[index] * tanh_c_val;
    offset_dx[j + Cx3] = grad_val * i_val;
  }
}

template <typename T>
__global__ void _LSTMActGrad(const int NxCx4, const int C, const T* x, T* dx) {
  const int Cx3 = C * 3, Cx4 = C * 4;
  CUDA_1D_KERNEL_LOOP(i, NxCx4) {
    const T val = x[i];
    dx[i] *= (i % Cx4 < Cx3 ? val * (T(1) - val) : T(1) - val * val);
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void LSTMCell<float, CUDAContext>(
    const int N,
    const int C,
    const float* c_prev,
    float* x,
    float* c,
    float* h,
    CUDAContext* ctx) {
  const auto NxC = N * C;
  _LSTMAct<<<CUDA_BLOCKS(NxC * 4), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      NxC * 4, C, x);
  _LSTMGate<<<CUDA_BLOCKS(NxC), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      NxC, C, c_prev, x, c, h);
}

template <>
void LSTMCellGrad<float, CUDAContext>(
    const int N,
    const int C,
    const float* c_prev,
    const float* x,
    const float* c,
    const float* dc,
    const float* dh,
    float* dc_prev,
    float* dx,
    CUDAContext* ctx) {
  const auto NxC = N * C;
  _LSTMGateGrad<<<CUDA_BLOCKS(NxC), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      NxC, C, c_prev, x, c, dc, dh, dc_prev, dx);
  _LSTMActGrad<<<CUDA_BLOCKS(NxC * 4), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      NxC * 4, C, x, dx);
} // LSTMCellGrad

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
