#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/conversions.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename Tx, typename Ty>
__global__ void _RowwiseMoments(
    const int rows,
    const int cols,
    const Tx* x,
    Ty* mean,
    Ty* var) {
  __shared__ typename BlockReduce<Ty>::TempStorage m_storage;
  __shared__ typename BlockReduce<Ty>::TempStorage v_storage;
  const Ty scale = Ty(1) / (Ty)rows;
  CUDA_2D_KERNEL_LOOP1(i, cols) {
    Ty m_val = 0, v_val = 0;
    CUDA_2D_KERNEL_LOOP2(j, rows) {
      const int xi = j * cols + i;
#if __CUDA_ARCH__ >= 350
      m_val += __ldg(x + xi);
      v_val += math::utils::Square(__ldg(x + xi));
#else
      m_val += x[xi];
      v_val += math::utils::Square(x[xi]);
#endif
    }
    m_val = BlockReduce<Ty>(m_storage).Sum(m_val);
    v_val = BlockReduce<Ty>(v_storage).Sum(v_val);
    if (threadIdx.x == 0) {
      const Ty mu = m_val * scale;
      mean[i] = mu;
      var[i] = v_val * scale - mu * mu;
    }
  }
}

template <>
__global__ void _RowwiseMoments<half, float>(
    const int rows,
    const int cols,
    const half* x,
    float* mean,
    float* var) {
  __shared__ typename BlockReduce<float>::TempStorage m_storage;
  __shared__ typename BlockReduce<float>::TempStorage v_storage;
  const float scale = 1.f / (float)rows;
  CUDA_2D_KERNEL_LOOP1(i, cols) {
    float m_val = 0.f, v_val = 0.f;
    CUDA_2D_KERNEL_LOOP2(j, rows) {
      const int xi = j * cols + i;
      m_val += __half2float(__ldg(x + xi));
      v_val += math::utils::Square(__half2float(__ldg(x + xi)));
    }
    m_val = BlockReduce<float>(m_storage).Sum(m_val);
    v_val = BlockReduce<float>(v_storage).Sum(v_val);
    if (threadIdx.x == 0) {
      const float mu = m_val * scale;
      mean[i] = mu;
      var[i] = v_val * scale - mu * mu;
    }
  }
}

template <typename Tx, typename Ty>
__global__ void _ColwiseMoments(
    const int rows,
    const int cols,
    const Tx* x,
    Ty* mean,
    Ty* var) {
  __shared__ typename BlockReduce<Ty>::TempStorage m_storage;
  __shared__ typename BlockReduce<Ty>::TempStorage v_storage;
  const Ty scale = Ty(1) / (Ty)cols;
  CUDA_2D_KERNEL_LOOP1(i, rows) {
    Ty m_val = 0, v_val = 0;
    CUDA_2D_KERNEL_LOOP2(j, cols) {
      const int xi = i * cols + j;
#if __CUDA_ARCH__ >= 350
      m_val += __ldg(x + xi);
      v_val += math::utils::Square(__ldg(x + xi));
#else
      m_val += x[xi];
      v_val += math::utils::Square(x[xi]);
#endif
    }
    m_val = BlockReduce<Ty>(m_storage).Sum(m_val);
    v_val = BlockReduce<Ty>(v_storage).Sum(v_val);
    if (threadIdx.x == 0) {
      const Ty mu = m_val * scale;
      mean[i] = mu;
      var[i] = v_val * scale - mu * mu;
    }
  }
}

template <>
__global__ void _ColwiseMoments<half, float>(
    const int rows,
    const int cols,
    const half* x,
    float* mean,
    float* var) {
  __shared__ typename BlockReduce<float>::TempStorage m_storage;
  __shared__ typename BlockReduce<float>::TempStorage v_storage;
  const float scale = 1.f / (float)cols;
  CUDA_2D_KERNEL_LOOP1(i, rows) {
    float m_val = 0.f, v_val = 0.f;
    CUDA_2D_KERNEL_LOOP2(j, cols) {
      const int xi = i * cols + j;
      m_val += __half2float(__ldg(x + xi));
      v_val += math::utils::Square(__half2float(__ldg(x + xi)));
    }
    m_val = BlockReduce<float>(m_storage).Sum(m_val);
    v_val = BlockReduce<float>(v_storage).Sum(v_val);
    if (threadIdx.x == 0) {
      const float mu = m_val * scale;
      mean[i] = mu;
      var[i] = v_val * scale - mu * mu;
    }
  }
}

template <typename Tx, typename Ty, int D>
__global__ void _GenericMoments(
    const int rows,
    const int cols,
    const int num_dims,
    const SimpleArray<int, D> x_dims,
    const SimpleArray<int, D> x_strides,
    const Tx* x,
    Ty* mean,
    Ty* var) {
  __shared__ typename BlockReduce<Ty>::TempStorage m_storage;
  __shared__ typename BlockReduce<Ty>::TempStorage v_storage;
  const Ty scale = Ty(1) / (Ty)cols;
  CUDA_2D_KERNEL_LOOP1(i, rows) {
    Ty m_val = 0, v_val = 0;
    CUDA_2D_KERNEL_LOOP2(j, cols) {
      int xi = 0, c = i * cols + j;
      for (int d = num_dims - 1; d >= 0; --d) {
        int r;
        FIXED_DIVISOR_DIV_MOD(x_dims.data[d], c, &c, &r);
        xi += r * x_strides.data[d];
      }
#if __CUDA_ARCH__ >= 350
      m_val += __ldg(x + xi);
      v_val += math::utils::Square(__ldg(x + xi));
#else
      m_val += x[xi];
      v_val += math::utils::Square(x[xi]);
#endif
    }
    m_val = BlockReduce<Ty>(m_storage).Sum(m_val);
    v_val = BlockReduce<Ty>(v_storage).Sum(v_val);
    if (threadIdx.x == 0) {
      const Ty mu = m_val * scale;
      mean[i] = mu;
      var[i] = v_val * scale - mu * mu;
    }
  }
}

template <int D>
__global__ void _GenericMoments(
    const int rows,
    const int cols,
    const int num_dims,
    const SimpleArray<int, D> x_dims,
    const SimpleArray<int, D> x_strides,
    const half* x,
    float* mean,
    float* var) {
  __shared__ typename BlockReduce<float>::TempStorage m_storage;
  __shared__ typename BlockReduce<float>::TempStorage v_storage;
  const float scale = 1.f / (float)cols;
  CUDA_2D_KERNEL_LOOP1(i, rows) {
    float m_val = 0.f, v_val = 0.f;
    CUDA_2D_KERNEL_LOOP2(j, cols) {
      int xi = 0, c = i * cols + j;
      for (int d = num_dims - 1; d >= 0; --d) {
        int r;
        FIXED_DIVISOR_DIV_MOD(x_dims.data[d], c, &c, &r);
        xi += r * x_strides.data[d];
      }
#if __CUDA_ARCH__ >= 350
      m_val += __half2float(__ldg(x + xi));
      v_val += math::utils::Square(__half2float(__ldg(x + xi)));
#else
      m_val += __half2float(x[xi]);
      v_val += math::utils::Square(__half2float(x[xi]));
#endif
    }
    m_val = BlockReduce<float>(m_storage).Sum(m_val);
    v_val = BlockReduce<float>(v_storage).Sum(v_val);
    if (threadIdx.x == 0) {
      const float mu = m_val * scale;
      mean[i] = mu;
      var[i] = v_val * scale - mu * mu;
    }
  }
}

template <typename Tx, typename Ty>
void _Moments(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const Tx* x,
    Ty* mean,
    Ty* var,
    CUDAContext* ctx) {
  int rows, cols;
  vec32_t y_dims(dims, dims + num_dims);
  for (int i = 0; i < num_axes; ++i)
    y_dims[axes[i]] = 1;

  /*! Case #1: Rowwise Reduce */
  if (math::utils::IsRowwiseReduce(
          num_dims, dims, y_dims.data(), &rows, &cols)) {
    _RowwiseMoments<<<
        CUDA_2D_BLOCKS(cols),
        CUDA_THREADS,
        0,
        ctx->cuda_stream()>>>(rows, cols, x, mean, var);
    return;
  }

  /*! Case #2: Colwise Reduce */
  if (math::utils::IsColwiseReduce(
          num_dims, dims, y_dims.data(), &rows, &cols)) {
    _ColwiseMoments<<<
        CUDA_2D_BLOCKS(rows),
        CUDA_THREADS,
        0,
        ctx->cuda_stream()>>>(rows, cols, x, mean, var);
    return;
  }

  /*! Case #3: Generic Reduce */
  CUDA_TENSOR_DIMS_CHECK(num_dims);
  SimpleArray<int, CUDA_TENSOR_MAX_DIMS> axesT, stridesT, dimsT;
  math::utils::TransposeAxesForReduce(num_dims, num_axes, axes, axesT.data);
  math::utils::ComputeTransposeStrides(
      num_dims, dims, axesT.data, stridesT.data);

  rows = cols = 1;
  const int pivot = num_dims - num_axes;
  for (int i = 0; i < pivot; ++i)
    rows *= dims[axesT.data[i]];
  for (int i = pivot; i < num_dims; ++i)
    cols *= dims[axesT.data[i]];
  for (int i = 0; i < num_dims; ++i)
    dimsT.data[i] = dims[axesT.data[i]];

  _GenericMoments<<<
      CUDA_2D_BLOCKS(rows),
      CUDA_THREADS,
      0,
      ctx->cuda_stream()>>>(
      rows, cols, num_dims, dimsT, stridesT, x, mean, var);
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void Moments<float16, float, CUDAContext>(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const float16* x,
    float* mean,
    float* var,
    CUDAContext* ctx) {
  _Moments(
      num_dims,
      dims,
      num_axes,
      axes,
      reinterpret_cast<const half*>(x),
      mean,
      var,
      ctx);
}

#define DEFINE_KERNEL_LAUNCHER(Tx, Ty)                           \
  template <>                                                    \
  void Moments<Tx, Ty, CUDAContext>(                             \
      const int num_dims,                                        \
      const int* dims,                                           \
      const int num_axes,                                        \
      const int* axes,                                           \
      const Tx* x,                                               \
      Ty* mean,                                                  \
      Ty* var,                                                   \
      CUDAContext* ctx) {                                        \
    _Moments(num_dims, dims, num_axes, axes, x, mean, var, ctx); \
  }

DEFINE_KERNEL_LAUNCHER(int8_t, float);
DEFINE_KERNEL_LAUNCHER(uint8_t, float);
DEFINE_KERNEL_LAUNCHER(int, float);
DEFINE_KERNEL_LAUNCHER(int64_t, float);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(double, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
