#include "dragon/kernels/sequence/op_kernels.h"
#include "dragon/utils/device/common_openmp.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
T sigmoid(T x) {
  return T(1) / (T(1) + exp(-x));
}

} // namespace

template <>
void LSTMCell<float, CPUContext>(
    const int N,
    const int C,
    const float* cx,
    float* actx,
    float* c,
    float* h,
    CPUContext* ctx) {
  float i, f, o, c_;
  int f_offset = C, o_offset = 2 * C, c_offset = 3 * C, x_offset = 4 * C;
  for (int n = 0; n < N; ++n) {
    for (int idx = 0; idx < C; ++idx) {
      actx[idx] = i = sigmoid(actx[idx]);
      actx[idx + f_offset] = f = sigmoid(actx[idx + f_offset]);
      actx[idx + o_offset] = o = sigmoid(actx[idx + o_offset]);
      actx[idx + c_offset] = c_ = tanh(actx[idx + c_offset]);
      c_ = c[idx] = f * cx[idx] + i * c_;
      h[idx] = o * tanh(c_);
    }
    cx += C;
    actx += x_offset;
    c += C;
    h += C;
  }
}

template <>
void LSTMCellGrad<float, CPUContext>(
    const int N,
    const int C,
    const float* cx,
    const float* actx,
    const float* c,
    const float* dc,
    const float* dh,
    float* dcx,
    float* dx,
    CPUContext* ctx) {
  float i, f, o, g, tanh_c, dcx_sum_term;
  int f_offset = C, o_offset = 2 * C, c_offset = 3 * C, x_offset = 4 * C;
  for (int n = 0; n < N; ++n) {
    for (int idx = 0; idx < C; ++idx) {
      i = actx[idx];
      f = actx[idx + f_offset];
      o = actx[idx + o_offset];
      g = actx[idx + c_offset];
      // BPTT compute the dc_{t-1} at the time of t
      // dc_{t-1} =  dl / d(h_{t}) * d(h_{t}) / d(c_{t}) * d(c_{t}) / d(c_{t-1})
      //                 + d(c_{t+1}) / d(c_{t}) * d(c_{t}) / d(c_{t-1})
      //          =  (dl / d(h_{t}) * d(h_{t}) / d(c_{t}) + d(c_{t+1}) /
      //                  d(c_{t})) * d(c_{t}) / d(c_{t-1})
      tanh_c = tanh(c[idx]);
      dcx_sum_term = dh[idx] * o * (1 - tanh_c * tanh_c) + dc[idx];
      dcx[idx] = dcx_sum_term * f;
      dx[idx] = dcx_sum_term * g * i * (1 - i);
      dx[idx + f_offset] = dcx_sum_term * cx[idx] * f * (1 - f);
      dx[idx + o_offset] = dh[idx] * tanh_c * o * (1 - o);
      dx[idx + c_offset] = dcx_sum_term * i * (1 - g * g);
    }
    cx += C;
    actx += x_offset;
    c += C;
    dc += C;
    dh += C;
    dcx += C;
    dx += x_offset;
  }
} // LSTMCellGrad

} // namespace kernels

} // namespace dragon
