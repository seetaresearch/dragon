#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

#define LDG(x, i) __ldg(x + i)
#define LDG2(x, i) convert::To<AccT>(__ldg(x + i))

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
      const int xi = j * cols + i;
      m_val += LDG2(x, xi);
      v_val += math::utils::Square(LDG2(x, xi));
    }
    m_val = BlockReduce<AccT>(m_storage).Sum(m_val);
    v_val = BlockReduce<AccT>(v_storage).Sum(v_val);
    if (threadIdx.x == 0) {
      const AccT mu = m_val * scale;
      mean[i] = mu;
      var[i] = v_val * scale - mu * mu;
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
      const int xi = i * cols + j;
      m_val += LDG2(x, xi);
      v_val += math::utils::Square(LDG2(x, xi));
    }
    m_val = BlockReduce<AccT>(m_storage).Sum(m_val);
    v_val = BlockReduce<AccT>(v_storage).Sum(v_val);
    if (threadIdx.x == 0) {
      const AccT mu = m_val * scale;
      mean[i] = mu;
      var[i] = v_val * scale - mu * mu;
    }
  }
}

template <typename T, typename AccT, int D>
__global__ void _GenericMoments(
    const int rows,
    const int cols,
    const int num_dims,
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
      for (int d = num_dims - 1; d >= 0; --d) {
        int r;
        FIXED_DIVISOR_DIV_MOD(X_dims.data[d], c, &c, &r);
        xi += r * X_strides.data[d];
      }
      m_val += LDG2(x, xi);
      v_val += math::utils::Square(LDG2(x, xi));
    }
    m_val = BlockReduce<AccT>(m_storage).Sum(m_val);
    v_val = BlockReduce<AccT>(v_storage).Sum(v_val);
    if (threadIdx.x == 0) {
      const AccT mu = m_val * scale;
      mean[i] = mu;
      var[i] = v_val * scale - mu * mu;
    }
  }
}

template <typename T, typename AccT>
void _Moments(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const T* x,
    AccT* mean,
    AccT* var,
    CUDAContext* ctx) {
  int rows, cols;
  vec32_t out_dims(dims, dims + num_dims);
  for (int i = 0; i < num_axes; ++i) {
    out_dims[axes[i]] = 1;
  }
  if (math::utils::IsRowwiseReduce(
          num_dims, dims, out_dims.data(), &rows, &cols)) {
    _RowwiseMoments<<<
        CUDA_2D_BLOCKS(cols),
        CUDA_THREADS,
        0,
        ctx->cuda_stream()>>>(rows, cols, x, mean, var);
    return;
  }
  if (math::utils::IsColwiseReduce(
          num_dims, dims, out_dims.data(), &rows, &cols)) {
    _ColwiseMoments<<<
        CUDA_2D_BLOCKS(rows),
        CUDA_THREADS,
        0,
        ctx->cuda_stream()>>>(rows, cols, x, mean, var);
    return;
  }
  CUDA_TENSOR_DIMS_CHECK(num_dims);
  SimpleArray<int, CUDA_TENSOR_MAX_DIMS> transpose_axes;
  SimpleArray<int, CUDA_TENSOR_MAX_DIMS> transpose_strides;
  SimpleArray<int, CUDA_TENSOR_MAX_DIMS> transpose_dims;
  math::utils::TransposeAxesForReduce(
      num_dims, num_axes, axes, transpose_axes.data);
  math::utils::ComputeTransposeStrides(
      num_dims, dims, transpose_axes.data, transpose_strides.data);
  rows = cols = 1;
  const int pivot = num_dims - num_axes;
  for (int i = 0; i < pivot; ++i) {
    rows *= dims[transpose_axes.data[i]];
  }
  for (int i = pivot; i < num_dims; ++i) {
    cols *= dims[transpose_axes.data[i]];
  }
  for (int i = 0; i < num_dims; ++i) {
    transpose_dims.data[i] = dims[transpose_axes.data[i]];
  }
  _GenericMoments<<<
      CUDA_2D_BLOCKS(rows),
      CUDA_THREADS,
      0,
      ctx->cuda_stream()>>>(
      rows, cols, num_dims, transpose_dims, transpose_strides, x, mean, var);
}

#undef LDG
#undef LDG2

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T, AccT)                        \
  template <>                                                  \
  void Moments<T, AccT, CUDAContext>(                          \
      const int num_dims,                                      \
      const int* dims,                                         \
      const int num_axes,                                      \
      const int* axes,                                         \
      const T* x,                                              \
      AccT* mean,                                              \
      AccT* var,                                               \
      CUDAContext* ctx) {                                      \
    _Moments(                                                  \
        num_dims,                                              \
        dims,                                                  \
        num_axes,                                              \
        axes,                                                  \
        reinterpret_cast<const math::ScalarType<T>::type*>(x), \
        mean,                                                  \
        var,                                                   \
        ctx);                                                  \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t, float);
DEFINE_KERNEL_LAUNCHER(int8_t, float);
DEFINE_KERNEL_LAUNCHER(int, float);
DEFINE_KERNEL_LAUNCHER(int64_t, double);
DEFINE_KERNEL_LAUNCHER(float16, float);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(double, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
