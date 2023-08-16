#include "dragon/kernels/normalization/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
void _LayerNorm(
    const int N,
    const int C,
    const AccT epsilon,
    const T* x,
    const AccT* gamma,
    const AccT* beta,
    AccT* mu,
    AccT* rsig,
    T* y) {
  const AccT scale = AccT(1) / AccT(C);
  for (int i = 0; i < N; ++i) {
    const int offset = i * C;
    AccT m_val = AccT(0), v_val = AccT(0);
    for (int j = 0; j < C; ++j) {
      const AccT val = convert::To<AccT>(x[offset + j]);
      m_val += val;
      v_val += val * val;
    }
    mu[i] = m_val = m_val * scale;
    v_val = std::sqrt(v_val * scale - m_val * m_val + epsilon);
    rsig[i] = v_val = AccT(1) / v_val;
    for (int j = 0; j < C; ++j) {
      AccT val = convert::To<AccT>(x[offset + j]);
      val = (val - m_val) * v_val;
      y[offset + j] = convert::To<T>(val * gamma[j] + beta[j]);
    }
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T, AccT)                           \
  template <>                                                     \
  void LayerNorm<T, AccT, CPUContext>(                            \
      const int N,                                                \
      const int C,                                                \
      const float epsilon,                                        \
      const T* x,                                                 \
      const AccT* gamma,                                          \
      const AccT* beta,                                           \
      AccT* mu,                                                   \
      AccT* rsig,                                                 \
      T* y,                                                       \
      CPUContext* ctx) {                                          \
    _LayerNorm(N, C, AccT(epsilon), x, gamma, beta, mu, rsig, y); \
  }

DEFINE_KERNEL_LAUNCHER(float16, float);
DEFINE_KERNEL_LAUNCHER(bfloat16, float);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(double, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
