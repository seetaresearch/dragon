#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename IndexType>
void _Flagged(
    const int count,
    const uint8_t* mask,
    IndexType* index,
    int* num_selected) {
  IndexType* offset_index = index;
  for (int i = 0; i < count; ++i) {
    if (mask[i]) *(offset_index++) = i;
  }
  num_selected[0] = std::distance(index, offset_index);
}

template <typename IndexType, typename CoordType>
void _UnravelIndex(
    const int count,
    const int num_dims,
    const int64_t* dims,
    const IndexType* index,
    CoordType* coord) {
  IndexType tmp;
  for (int i = 0; i < count; ++i) {
    tmp = index[i];
    auto* offset_coord = coord + i * num_dims;
    for (int d = num_dims - 1; d >= 0; --d) {
      FIXED_DIVISOR_DIV_MOD(dims[d], tmp, &tmp, (offset_coord + d));
    }
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(IndexType)       \
  template <>                                   \
  void Flagged<IndexType, CPUContext>(          \
      const int count,                          \
      const uint8_t* mask,                      \
      IndexType* index,                         \
      int* num_selected,                        \
      CPUContext* ctx) {                        \
    _Flagged(count, mask, index, num_selected); \
  }

DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(IndexType, CoordType)    \
  template <>                                           \
  void UnravelIndex<IndexType, CoordType, CPUContext>(  \
      const int count,                                  \
      const int num_dims,                               \
      const int64_t* dims,                              \
      const IndexType* index,                           \
      CoordType* coord,                                 \
      CPUContext* ctx) {                                \
    _UnravelIndex(count, num_dims, dims, index, coord); \
  }

DEFINE_KERNEL_LAUNCHER(int, int);
DEFINE_KERNEL_LAUNCHER(int, int64_t);
DEFINE_KERNEL_LAUNCHER(int64_t, int);
DEFINE_KERNEL_LAUNCHER(int64_t, int64_t);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
