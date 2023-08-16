#include "dragon/kernels/math/op_kernels.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"

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
      mean[i] = m_val;
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
      mean[i] = m_val;
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
      mean[i] = m_val;
      var[i] = v_val * scale - m_val * m_val;
    }
  }
}

template <typename T, typename AccT, int D>
void DispatchGenericMoments(
    const int* dims,
    const int num_axes,
    const int* axes,
    const T* x,
    AccT* mean,
    AccT* var,
    CUDAContext* ctx) {
  SimpleArray<int, D> transpose_axes;
  SimpleArray<int, D> transpose_dims;
  SimpleArray<int, D> transpose_strides;
  math::utils::TransposeAxesForReduce(D, num_axes, axes, transpose_axes.data);
  math::utils::ComputeTransposeStrides(
      D, dims, transpose_axes.data, transpose_strides.data);
  int rows = 1, cols = 1;
  const int pivot = D - num_axes;
  // clang-format off
  for (int i = 0; i < pivot; ++i) rows *= dims[transpose_axes.data[i]];
  for (int i = pivot; i < D; ++i) cols *= dims[transpose_axes.data[i]];
  for (int i = 0; i < D; ++i) transpose_dims.data[i] = dims[transpose_axes.data[i]];
  // clang-format on
  _GenericMoments<<<rows, CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      rows, cols, transpose_dims, transpose_strides, x, mean, var);
}

template <typename T, typename AccT>
void DispatchMoments(
    const int num_dims,
    const int64_t* dims,
    const int num_axes,
    const int64_t* axes,
    const T* x,
    AccT* mean,
    AccT* var,
    CUDAContext* ctx) {
  int64_t rows, cols;
  vec64_t out_dims(dims, dims + num_dims);
  for (int i = 0; i < num_axes; ++i) {
    out_dims[axes[i]] = 1;
  }
  if (math::utils::IsRowwiseReduce(
          num_dims, dims, out_dims.data(), &rows, &cols)) {
    return _RowwiseMoments<<<cols, CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        rows, cols, x, mean, var);
  }
  if (math::utils::IsColwiseReduce(
          num_dims, dims, out_dims.data(), &rows, &cols)) {
    return _ColwiseMoments<<<rows, CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        rows, cols, x, mean, var);
  }
  CUDA_TENSOR_DIMS_CHECK(num_dims);
  DISPATCH_FUNC_BY_VALUE_WITH_TYPE_2(
      DispatchGenericMoments,
      T,
      AccT,
      num_dims,
      vec32_t(dims, dims + num_dims).data(),
      num_axes,
      vec32_t(axes, axes + num_axes).data(),
      x,
      mean,
      var,
      ctx);
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T, AccT)                      \
  template <>                                                \
  void Moments<T, AccT, CUDAContext>(                        \
      const int num_dims,                                    \
      const int64_t* dims,                                   \
      const int num_axes,                                    \
      const int64_t* axes,                                   \
      const T* x,                                            \
      AccT* mean,                                            \
      AccT* var,                                             \
      CUDAContext* ctx) {                                    \
    vec64_t new_dims, new_axes;                              \
    math::utils::CollapseReduceAxes(                         \
        num_dims, dims, num_axes, axes, new_dims, new_axes); \
    DispatchMoments(                                         \
        new_dims.size(),                                     \
        new_dims.data(),                                     \
        new_axes.size(),                                     \
        new_axes.data(),                                     \
        x,                                                   \
        mean,                                                \
        var,                                                 \
        ctx);                                                \
  }

DEFINE_KERNEL_LAUNCHER(float16, float);
DEFINE_KERNEL_LAUNCHER(bfloat16, float);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(double, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
