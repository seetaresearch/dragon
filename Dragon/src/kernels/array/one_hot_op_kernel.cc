#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CPU> */

template <typename T>
void _OneHot(
    const int               count,
    const int               depth,
    const int               on_value,
    const T*                x,
    T*                      y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        const int val = (int)x[i];
        y[i * depth + val] = (T)on_value;
    }
}

/* <T = float32, Device = CPU> */

template <> void OneHot<float, CPUContext>(
    const int               count,
    const int               depth,
    const int               on_value,
    const float*            x,
    float*                  y,
    CPUContext*             ctx) {
    _OneHot(count, depth, on_value, x, y);
}

/* <T = int32, Device = CPU> */

template <> void OneHot<int, CPUContext>(
    const int               count,
    const int               depth,
    const int               on_value,
    const int*              x,
    int*                    y,
    CPUContext*             ctx) {
    _OneHot(count, depth, on_value, x, y);
}

/* <T = int64, Device = CPU> */

template <> void OneHot<int64_t, CPUContext>(
    const int               count,
    const int               depth,
    const int               on_value,
    const int64_t*          x,
    int64_t*                y,
    CPUContext*             ctx) {
    _OneHot(count, depth, on_value, x, y);
}

}  // namespace kernel

}  // namepsace dragon