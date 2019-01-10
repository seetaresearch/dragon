#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"
#include "utils/math_utils.h"
#include "utils/cub_device.h"
#include "utils/math_functions.h"

namespace dragon {

namespace kernel {

template <typename Tx, typename Ty>
__global__ void _ColwiseMoments(
    const int                   rows,
    const int                   cols,
    const Tx*                   x,
    Ty*                         mean,
    Ty*                         var) {
    __shared__ typename BlockReduce<Ty>::TempStorage m_storage;
    __shared__ typename BlockReduce<Ty>::TempStorage v_storage;
    const Ty scale = (Ty)1 / static_cast<Ty>(cols);
    CUDA_2D_KERNEL_LOOP1(i, rows) {
        Ty m_val = 0, v_val = 0;
        CUDA_2D_KERNEL_LOOP2(j, cols) {
            const int x_idx = i * cols + j;
#if __CUDA_ARCH__ >= 350
            m_val += __ldg(x + x_idx);
            v_val += __ldg(x + x_idx) * __ldg(x + x_idx);
#else
            m_val += x[x_idx];
            v_val += x[x_idx] * x[x_idx];
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

template<> __global__ void _ColwiseMoments<half, float>(
    const int                   rows,
    const int                   cols,
    const half*                 x,
    float*                      mean,
    float*                      var) {
#if __CUDA_ARCH__ >= 530
    __shared__ typename BlockReduce<float>::TempStorage m_storage;
    __shared__ typename BlockReduce<float>::TempStorage v_storage;
    const float scale = 1.f / cols;
    CUDA_2D_KERNEL_LOOP1(i, rows) {
        float m_val = 0.f, v_val = 0.f;
        CUDA_2D_KERNEL_LOOP2(j, cols) {
            const int x_idx = i * cols + j;
            m_val += __half2float(__ldg(x + x_idx));
            v_val += __half2float(__ldg(x + x_idx)) *
                     __half2float(__ldg(x + x_idx));
        }
        m_val = BlockReduce<float>(m_storage).Sum(m_val);
        v_val = BlockReduce<float>(v_storage).Sum(v_val);
        if (threadIdx.x == 0) {
            const float mu = m_val * scale;
            mean[i] = mu;
            var[i] = v_val * scale - mu * mu;
        }
    }
#endif
}

template <typename Tx, typename Ty>
__global__ void _RowwiseMoments(
    const int                   rows,
    const int                   cols,
    const Tx*                   x,
    Ty*                         mean,
    Ty*                         var) {
    __shared__ typename BlockReduce<Ty>::TempStorage m_storage;
    __shared__ typename BlockReduce<Ty>::TempStorage v_storage;
    const Ty scale = (Ty)1 / static_cast<Ty>(rows);
    CUDA_2D_KERNEL_LOOP1(i, cols) {
        Ty m_val = 0, v_val = 0;
        CUDA_2D_KERNEL_LOOP2(j, rows) {
            const int x_idx = j * cols + i;
#if __CUDA_ARCH__ >= 350
            m_val += __ldg(x + x_idx);
            v_val += __ldg(x + x_idx) * __ldg(x + x_idx);
#else
            m_val += x[x_idx];
            v_val += x[x_idx] * x[x_idx];
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

template<> __global__ void _RowwiseMoments<half, float>(
    const int                   rows,
    const int                   cols,
    const half*                 x,
    float*                      mean,
    float*                      var) {
#if __CUDA_ARCH__ >= 530
    __shared__ typename BlockReduce<float>::TempStorage m_storage;
    __shared__ typename BlockReduce<float>::TempStorage v_storage;
    const float scale = 1.f / rows;
    CUDA_2D_KERNEL_LOOP1(i, cols) {
        float m_val = 0.f, v_val = 0.f;
        CUDA_2D_KERNEL_LOOP2(j, rows) {
            const int x_idx = j * cols + i;
            m_val += __half2float(__ldg(x + x_idx));
            v_val += __half2float(__ldg(x + x_idx)) *
                     __half2float(__ldg(x + x_idx));
        }
        m_val = BlockReduce<float>(m_storage).Sum(m_val);
        v_val = BlockReduce<float>(v_storage).Sum(v_val);
        if (threadIdx.x == 0) {
            const float mu = m_val * scale;
            mean[i] = mu;
            var[i] = v_val * scale - mu * mu;
        }
    }
#endif
}

#define FIXED_DIVISOR_DIV_MOD(d, n, q, r) \
  do {                                    \
    const auto n_copy = n;                \
    *q = n_copy / d;                      \
    *r = n_copy % d;                      \
  } while (0)

template <typename Tx, typename Ty>
__global__ void _GenericMoments(
    const int                   ndims,
    const int                   outer_dim,
    const int                   inner_dim,
    const int*                  x_strides,
    const int*                  y_dims,
    const Tx*                   x,
    Ty*                         mean,
    Ty*                         var) {
    __shared__ typename BlockReduce<Ty>::TempStorage m_storage;
    __shared__ typename BlockReduce<Ty>::TempStorage v_storage;
    const Ty scale = (Ty)1 / static_cast<Ty>(inner_dim);
    CUDA_2D_KERNEL_LOOP1(i, outer_dim) {
        Ty m_val = 0, v_val = 0;
        CUDA_2D_KERNEL_LOOP2(j, inner_dim) {
            int x_idx = 0, y_idx = i * inner_dim + j;
#pragma unroll
            for (int d = ndims - 1; d >= 0; --d) {
                int r;
#if __CUDA_ARCH__ >= 350
                FIXED_DIVISOR_DIV_MOD(__ldg(y_dims + d), y_idx, &y_idx, &r);
                x_idx += r * __ldg(x_strides + d);
#else
                FIXED_DIVISOR_DIV_MOD(y_dims[d], y_idx, &y_idx, &r);
                x_idx += r * x_strides[d];
#endif
            }
#if __CUDA_ARCH__ >= 350
            m_val += __ldg(x + x_idx);
            v_val += __ldg(x + x_idx) * __ldg(x + x_idx);
#else
            m_val += x[x_idx];
            v_val += x[x_idx] * x[x_idx];
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

template<> __global__ void _GenericMoments<half, float>(
    const int                   ndims,
    const int                   outer_dim,
    const int                   inner_dim,
    const int*                  x_strides,
    const int*                  y_dims,
    const half*                 x,
    float*                      mean,
    float*                      var) {
#if __CUDA_ARCH__ >= 530
    __shared__ typename BlockReduce<float>::TempStorage m_storage;
    __shared__ typename BlockReduce<float>::TempStorage v_storage;
    const float scale = 1.f / inner_dim;
    CUDA_2D_KERNEL_LOOP1(i, outer_dim) {
        float m_val = 0.f, v_val = 0.f;
        CUDA_2D_KERNEL_LOOP2(j, inner_dim) {
            int x_idx = 0, y_idx = i * inner_dim + j;
#pragma unroll
            for (int d = ndims - 1; d >= 0; --d) {
                int r;
                FIXED_DIVISOR_DIV_MOD(__ldg(y_dims + d), y_idx, &y_idx, &r);
                x_idx += r * __ldg(x_strides + d);
            }
            m_val += __half2float(__ldg(x + x_idx));
            v_val += __half2float(__ldg(x + x_idx)) *
                     __half2float(__ldg(x + x_idx));
        }
        m_val = BlockReduce<float>(m_storage).Sum(m_val);
        v_val = BlockReduce<float>(v_storage).Sum(v_val);
        if (threadIdx.x == 0) {
            const float mu = m_val * scale;
            mean[i] = mu;
            var[i] = v_val * scale - mu * mu;
        }
    }
#endif
}

template <typename Tx, typename Ty>
void _Moments(
    const int               ndims,
    const int*              dims,
    const int               naxes,
    const int*              axes,
    const Tx*               x,
    Ty*                     mean,
    Ty*                     var,
    CUDAContext*            ctx) {
    vector<int> y_dimsV(dims, dims + ndims);
    for (int i = 0; i < naxes; ++i) y_dimsV[axes[i]] = 1;
    const int* x_dims = dims; const int* y_dims = y_dimsV.data();
    const int x_size = std::accumulate(x_dims,
        x_dims + ndims, 1, std::multiplies<int>());
    const int y_size = std::accumulate(y_dims,
        y_dims + ndims, 1, std::multiplies<int>());

    int rows, cols;

    /*! Case #1: Colwise Reduce */
    if (utils::IsColwiseReduce(ndims, x_dims, y_dims, &rows, &cols)) {
        _ColwiseMoments<Tx, Ty>
            << < CUDA_2D_BLOCKS(rows), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (rows, cols, x, mean, var); return;
    }

    /*! Case #2: Rowwise Reduce */
    if (utils::IsRowwiseReduce(ndims, x_dims, y_dims, &rows, &cols)) {
        _RowwiseMoments<Tx, Ty>
            << < CUDA_2D_BLOCKS(cols), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (rows, cols, x, mean, var); return;
    }

    /*! Case #3: Generic Reduce */
    vector<int> axesT(ndims), stridesT(ndims), dimsT(ndims);

    utils::ComputeTransposedAxesForReduce(
        ndims, naxes, axes, axesT.data());
    utils::ComputeTransposedStrides(
        ndims, dims, axesT.data(), stridesT.data());

    int outer_dim = 1, inner_dim = 1;
    const int pivot = ndims - naxes;
    for (int i = 0; i < pivot; ++i) outer_dim *= dims[axesT[i]];
    for (int i = pivot; i < ndims; ++i) inner_dim *= dims[axesT[i]];
    for (int i = 0; i < ndims; ++i) dimsT[i] = dims[axesT[i]];

    const int dbytes = sizeof(int) * ndims;
    int* XSS = (int*)ctx->New(dbytes), *YDS = (int*)ctx->New(dbytes);
    ctx->Memcpy<CUDAContext, CPUContext>(dbytes, XSS, stridesT.data());
    ctx->Memcpy<CUDAContext, CPUContext>(dbytes, YDS, dimsT.data());

    _GenericMoments<Tx, Ty>
        << < CUDA_2D_BLOCKS(outer_dim), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (ndims, outer_dim, inner_dim, XSS, YDS, x, mean, var);

    ctx->FinishDeviceCompution();
    ctx->Delete(XSS); ctx->Delete(YDS);
}

#define DEFINE_MOMENTS_KERNEL_LAUNCHER(Tx, Ty) \
    template <> void Moments<Tx, Ty, CUDAContext>( \
        const int               ndims, \
        const int*              dims, \
        const int               naxes, \
        const int*              axes, \
        const Tx*               x, \
        Ty*                     mean, \
        Ty*                     var, \
        CUDAContext*            ctx) { \
        _Moments<Tx, Ty>(ndims, dims, \
            naxes, axes, x, mean, var, ctx); \
    }

DEFINE_MOMENTS_KERNEL_LAUNCHER(int8_t, float);
DEFINE_MOMENTS_KERNEL_LAUNCHER(uint8_t, float);
DEFINE_MOMENTS_KERNEL_LAUNCHER(int, float);
DEFINE_MOMENTS_KERNEL_LAUNCHER(int64_t, float);
DEFINE_MOMENTS_KERNEL_LAUNCHER(float, float);
DEFINE_MOMENTS_KERNEL_LAUNCHER(double, double);

template <> void Moments<float16, float, CUDAContext>(
    const int               ndims,
    const int*              dims,
    const int               naxes,
    const int*              axes,
    const float16*          x,
    float*                  mean,
    float*                  var,
    CUDAContext*            ctx) {
    _Moments<half, float>(ndims, dims, naxes, axes,
        reinterpret_cast<const half*>(x), mean, var, ctx);
}

#undef FIXED_DIVISOR_DIV_MOD
#undef DEFINE_MOMENTS_KERNEL_LAUNCHER

}  // namespace kernel

}  // namespace dragon

#endif  // WITH_CUDA