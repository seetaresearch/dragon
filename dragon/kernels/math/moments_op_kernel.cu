#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
__global__ void _RowwiseMoments(
    const int rows,
    const int cols,
    const T* x,
    AccT* mean,
    AccT* var) {
  __shared__ typename BlockReduce<AccT>::TempStorage m_storage;
  __shared__ typename BlockReduce<AccT>::TempStorage v_storage;
  const AccT scale = AccT(1) / AccT(rows);
  CUDA_2D_KERNEL_LOOP1(i, cols) {
    AccT m_val = AccT(0), v_val = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, rows) {
      const AccT val = convert::To<AccT>(x[j * cols + i]);
      m_val += val;
      v_val += val * val;
    }
    m_val = BlockReduce<AccT>(m_storage).Sum(m_val);
    v_val = BlockReduce<AccT>(v_storage).Sum(v_val);
    if (threadIdx.x == 0) {
      m_val = m_val * scale;
      if (mean != nullptr) mean[i] = m_val;
      var[i] = v_val * scale - m_val * m_val;
    }
  }
}

template <typename T, typename AccT>
__global__ void _ColwiseMoments(
    const int rows,
    const int cols,
    const T* x,
    AccT* mean,
    AccT* var) {
  __shared__ typename BlockReduce<AccT>::TempStorage m_storage;
  __shared__ typename BlockReduce<AccT>::TempStorage v_storage;
  const AccT scale = AccT(1) / AccT(cols);
  CUDA_2D_KERNEL_LOOP1(i, rows) {
    AccT m_val = AccT(0), v_val = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, cols) {
      const AccT val = convert::To<AccT>(x[i * cols + j]);
      m_val += val;
      v_val += val * val;
    }
    m_val = BlockReduce<AccT>(m_storage).Sum(m_val);
    v_val = BlockReduce<AccT>(v_storage).Sum(v_val);
    if (threadIdx.x == 0) {
      m_val = m_val * scale;
      if (mean != nullptr) mean[i] = m_val;
      var[i] = v_val * scale - m_val * m_val;
    }
  }
}

template <typename T, typename AccT, int D>
__global__ void _GenericMoments(
    const int rows,
    const int cols,
    const SimpleArray<int, D> X_dims,
    const SimpleArray<int, D> X_strides,
    const T* x,
    AccT* mean,
    AccT* var) {
  __shared__ typename BlockReduce<AccT>::TempStorage m_storage;
  __shared__ typename BlockReduce<AccT>::TempStorage v_storage;
  const AccT scale = AccT(1) / AccT(cols);
  CUDA_2D_KERNEL_LOOP1(i, rows) {
    AccT m_val = AccT(0), v_val = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, cols) {
      int xi = 0, c = i * cols + j;
#pragma unroll
      for (int d = D - 1; d >= 0; --d) {
        int r;
        FIXED_DIVISOR_DIV_MOD(X_dims.data[d], c, &c, &r);
        xi += r * X_strides.data[d];
      }
      const AccT val = convert::To<AccT>(x[xi]);
      m_val += val;
      v_val += val * val;
    }
    m_val = BlockReduce<AccT>(m_storage).Sum(m_val);
    v_val = BlockReduce<AccT>(v_storage).Sum(v_val);
    if (threadIdx.x == 0) {
      mean[i] = m_val = m_val * scale;
      if (mean != nullptr) mean[i] = m_val;
      var[i] = v_val * scale - m_val * m_val;
    }
  }
}

template <typename T, typename AccT, int D>
void _GenericMomentsImpl(
    const int* dims,
    const int num_axes,
    const int* axes,
    const T* x,
    AccT* mean,
    AccT* var,
    CUDAContext* ctx) {
  SimpleArray<int, D> transpose_axes;
  SimpleArray<int, D> transpose_strides;
  SimpleArray<int, D> transpose_dims;
  math::utils::TransposeAxesForReduce(D, num_axes, axes, transpose_axes.data);
  math::utils::ComputeTransposeStrides(
      D, dims, transpose_axes.data, transpose_strides.data);
  int rows = 1, cols = 1;
  const int pivot = D - num_axes;
  for (int i = 0; i < pivot; ++i) {
    rows *= dims[transpose_axes.data[i]];
  }
  for (int i = pivot; i < D; ++i) {
    cols *= dims[transpose_axes.data[i]];
  }
  for (int i = 0; i < D; ++i) {
    transpose_dims.data[i] = dims[transpose_axes.data[i]];
  }
  _GenericMoments<<<rows, CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      rows, cols, transpose_dims, transpose_strides, x, mean, var);
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T, AccT)                                      \
  template <>                                                                \
  void Moments<T, AccT, CUDAContext>(                                        \
      const int num_dims,                                                    \
      const int64_t* dims,                                                   \
      const int num_axes,                                                    \
      const int64_t* axes,                                                   \
      const T* x,                                                            \
      AccT* mean,                                                            \
      AccT* var,                                                             \
      CUDAContext* ctx) {                                                    \
    vec64_t new_dims, new_axes;                                              \
    math::utils::CollapseReduceAxes(                                         \
        num_dims, dims, num_axes, axes, new_dims, new_axes);                 \
    int num_new_dims = new_dims.size();                                      \
    int64_t rows, cols;                                                      \
    vec64_t out_dims(new_dims);                                              \
    for (auto axis : new_axes) {                                             \
      out_dims[axis] = 1;                                                    \
    }                                                                        \
    if (math::utils::IsRowwiseReduce(                                        \
            num_new_dims, new_dims.data(), out_dims.data(), &rows, &cols)) { \
      return _RowwiseMoments<<<cols, CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          rows, cols, x, mean, var);                                         \
    }                                                                        \
    if (math::utils::IsColwiseReduce(                                        \
            num_new_dims, new_dims.data(), out_dims.data(), &rows, &cols)) { \
      return _ColwiseMoments<<<rows, CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          rows, cols, x, mean, var);                                         \
    }                                                                        \
    CUDA_TENSOR_DIMS_CHECK(num_new_dims);                                    \
    DISPATCH_FUNC_BY_VALUE_WITH_TYPE_2(                                      \
        _GenericMomentsImpl,                                                 \
        T,                                                                   \
        AccT,                                                                \
        num_new_dims,                                                        \
        vec32_t({new_dims.begin(), new_dims.end()}).data(),                  \
        new_axes.size(),                                                     \
        vec32_t({new_axes.begin(), new_axes.end()}).data(),                  \
        x,                                                                   \
        mean,                                                                \
        var,                                                                 \
        ctx);                                                                \
  }

DEFINE_KERNEL_LAUNCHER(int, float);
DEFINE_KERNEL_LAUNCHER(int64_t, double);
DEFINE_KERNEL_LAUNCHER(float16, float);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(double, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
