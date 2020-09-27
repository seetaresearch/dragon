#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename IndexType, typename CoordType, int D>
__global__ void _UnravelIndex(
    const int nthreads,
    const int num_dims,
    const SimpleArray<int, D> dims,
    const IndexType* index,
    CoordType* coord) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    IndexType tmp = index[i];
    CoordType* offset_coord = coord + i * num_dims;
    for (int d = num_dims - 1; d >= 0; --d) {
      FIXED_DIVISOR_DIV_MOD(dims.data[d], tmp, &tmp, (offset_coord + d));
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(IndexType)                             \
  template <>                                                         \
  void Flagged<IndexType, CUDAContext>(                               \
      const int count,                                                \
      const uint8_t* mask,                                            \
      IndexType* index,                                               \
      int* num_selected,                                              \
      CUDAContext* ctx) {                                             \
    IndexType num_selected_host;                                      \
    auto* num_selected_dev = index + count;                           \
    size_t ws_nbytes = 0;                                             \
    cub::CountingInputIterator<int> itr(0);                           \
    cub::DeviceSelect::Flagged(                                       \
        nullptr,                                                      \
        ws_nbytes,                                                    \
        itr,                                                          \
        mask,                                                         \
        index,                                                        \
        static_cast<int64_t*>(nullptr),                               \
        count,                                                        \
        ctx->cuda_stream());                                          \
    cub::DeviceSelect::Flagged(                                       \
        ctx->workspace()->template data<CUDAContext>({ws_nbytes})[0], \
        ws_nbytes,                                                    \
        itr,                                                          \
        mask,                                                         \
        index,                                                        \
        num_selected_dev,                                             \
        count,                                                        \
        ctx->cuda_stream());                                          \
    CUDA_CHECK(cudaMemcpyAsync(                                       \
        &num_selected_host,                                           \
        num_selected_dev,                                             \
        sizeof(IndexType),                                            \
        cudaMemcpyDefault,                                            \
        ctx->cuda_stream()));                                         \
    ctx->FinishDeviceComputation();                                   \
    num_selected[0] = num_selected_host;                              \
  }

DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(IndexType, CoordType)                  \
  template <>                                                         \
  void UnravelIndex<IndexType, CoordType, CUDAContext>(               \
      const int count,                                                \
      const int num_dims,                                             \
      const int64_t* dims,                                            \
      const IndexType* index,                                         \
      CoordType* coord,                                               \
      CUDAContext* ctx) {                                             \
    CUDA_TENSOR_DIMS_CHECK(num_dims);                                 \
    SimpleArray<int, CUDA_TENSOR_MAX_DIMS> X_dims;                    \
    for (int i = 0; i < num_dims; ++i)                                \
      X_dims.data[i] = dims[i];                                       \
    _UnravelIndex<<<                                                  \
        CUDA_BLOCKS(count),                                           \
        CUDA_THREADS,                                                 \
        0,                                                            \
        ctx->cuda_stream()>>>(count, num_dims, X_dims, index, coord); \
  }

DEFINE_KERNEL_LAUNCHER(int, int);
DEFINE_KERNEL_LAUNCHER(int, int64_t);
DEFINE_KERNEL_LAUNCHER(int64_t, int);
DEFINE_KERNEL_LAUNCHER(int64_t, int64_t);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
