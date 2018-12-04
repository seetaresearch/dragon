#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! Arange <T = ?, Device = CPU> */

template <typename T>
void _Arange(
    const int               count,
    const int               start,
    const int               step,
    T*                      y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = static_cast<T>(start + i * step);
    }
}

/*! Arange <T = float32, Device = CPU> */

template<> void Arange<float, CPUContext>(
    const int               count,
    const int               start,
    const int               step,
    float*                  y,
    CPUContext*             ctx) {
    _Arange<float>(count, start, step, y);
}

/*! Arange <T = int32, Device = CPU> */

template<> void Arange<int, CPUContext>(
    const int               count,
    const int               start,
    const int               step,
    int*                    y,
    CPUContext*             ctx) {
    _Arange<int>(count, start, step, y);
}

/*! Arange <T = int64, Device = CPU> */

template<> void Arange<int64_t, CPUContext>(
    const int               count,
    const int               start,
    const int               step,
    int64_t*                y,
    CPUContext*             ctx) {
    _Arange<int64_t>(count, start, step, y);
}

}  // namespace kernel

}  // namepsace dragon