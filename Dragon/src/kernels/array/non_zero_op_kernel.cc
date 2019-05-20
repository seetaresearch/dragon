#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

#define FIXED_DIVISOR_DIV_MOD(d, n, q, r) \
    do {                                  \
        const auto n_copy = n;            \
        *q = n_copy / d;                  \
        *r = n_copy % d;                  \
    } while (0)

template <> void UnravelIndex<CPUContext>(
    const int               count,
    const int               ndims,
    const int*              dims,
    const int64_t*          x,
    int64_t*                y,
    CPUContext*             ctx) {
    int tmp, d; int64_t* Y;
    for (int i = 0; i < count; ++i) {
        tmp = x[i]; Y = y + i * ndims;
        for (d = ndims - 1; d >= 0; --d) {
            FIXED_DIVISOR_DIV_MOD(dims[d], tmp, &tmp, (Y + d));
        }
    }
}

#undef FIXED_DIVISOR_DIV_MOD

}  // namespace kernel

}  // namepsace dragon