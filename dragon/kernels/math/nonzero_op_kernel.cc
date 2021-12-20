#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename IndexT>
void _Flagged(
    const int N,
    const uint8_t* mask,
    IndexT* index,
    int* num_selected) {
  IndexT* offset_index = index;
  for (int i = 0; i < N; ++i) {
    if (mask[i]) {
      *(offset_index++) = i;
    }
  }
  num_selected[0] = std::distance(index, offset_index);
}

template <typename IndexT, typename CoordT>
void _UnravelIndex(
    const int N,
    const int num_dims,
    const int64_t* dims,
    const IndexT* index,
    CoordT* coord) {
  IndexT tmp;
  for (int i = 0; i < N; ++i) {
    tmp = index[i];
    auto* offset_coord = coord + i * num_dims;
    for (int d = num_dims - 1; d >= 0; --d) {
      FIXED_DIVISOR_DIV_MOD(dims[d], tmp, &tmp, (offset_coord + d));
    }
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(IndexT)      \
  template <>                               \
  void Flagged<IndexT, CPUContext>(         \
      const int N,                          \
      const uint8_t* mask,                  \
      IndexT* index,                        \
      int* num_selected,                    \
      CPUContext* ctx) {                    \
    _Flagged(N, mask, index, num_selected); \
  }

DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(IndexT, CoordT)      \
  template <>                                       \
  void UnravelIndex<IndexT, CoordT, CPUContext>(    \
      const int N,                                  \
      const int num_dims,                           \
      const int64_t* dims,                          \
      const IndexT* index,                          \
      CoordT* coord,                                \
      CPUContext* ctx) {                            \
    _UnravelIndex(N, num_dims, dims, index, coord); \
  }

DEFINE_KERNEL_LAUNCHER(int, int);
DEFINE_KERNEL_LAUNCHER(int, int64_t);
DEFINE_KERNEL_LAUNCHER(int64_t, int);
DEFINE_KERNEL_LAUNCHER(int64_t, int64_t);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
