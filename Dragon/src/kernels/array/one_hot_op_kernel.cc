#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! OneHot <T = ?, Device = CPU> */

template <typename T>
void _OneHot(
    const int               count,
    const int               depth,
    const int               on_value,
    const T*                x,
    T*                      y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        const int val = (int)x[i];
        y[i * depth + val] = static_cast<T>(on_value);
    }
}

/*! OneHot <T = float32, Device = CPU> */

template <> void OneHot<float, CPUContext>(
    const int               count,
    const int               depth,
    const int               on_value,
    const float*            x,
    float*                  y,
    CPUContext*             ctx) {
    _OneHot<float>(count, depth, on_value, x, y);
}

/*! OneHot <T = int32, Device = CPU> */

template <> void OneHot<int, CPUContext>(
    const int               count,
    const int               depth,
    const int               on_value,
    const int*              x,
    int*                    y,
    CPUContext*             ctx) {
    _OneHot<int>(count, depth, on_value, x, y);
}

/*! OneHot <T = int64, Device = CPU> */

template <> void OneHot<int64_t, CPUContext>(
    const int               count,
    const int               depth,
    const int               on_value,
    const int64_t*          x,
    int64_t*                y,
    CPUContext*             ctx) {
    _OneHot<int64_t>(count, depth, on_value, x, y);
}

}  // namespace kernel

}  // namepsace dragon