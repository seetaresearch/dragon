#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! Dropout <T = float32, Device = CPU> */

template<> void Dropout<float, CPUContext>(
    const int               count,
    float                   prob,
    float                   scale,
    const float*            x,
    uint32_t*               mask32,
    uint8_t*                mask8,
    float*                  y,
    CPUContext*             ctx) {
    math::RandomBernoulli<uint8_t, CPUContext>(
        count, 1 - prob, mask8, ctx);
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = x[i] * mask8[i] * scale;
    }
}

/*! Dropout <T = float16, Device = CPU> */

template<> void Dropout<float16, CPUContext>(
    const int               count,
    float                   prob,
    float                   scale,
    const float16*          x,
    uint32_t*               mask32,
    uint8_t*                mask8,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

/*! ApplyMask <Tx = float32, Tm = uint8, Device = CPU> */

template <typename Tx, typename Tm>
void _ApplyMask(
    const int               count,
    const float             scale,
    const Tx*               x,
    const Tm*               mask,
    Tx*                     y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = x[i] * mask[i] * scale;
    }
}

template <> void ApplyMask<float, uint8_t, CPUContext>(
    const int               count,
    const float             scale,
    const float*            x,
    const uint8_t*          mask,
    float*                  y,
    CPUContext*             ctx) {
    _ApplyMask<float, uint8_t>(count, scale, x, mask, y);
}

/*! ApplyMask <Tx = float16, Tm = uint8, Device = CPU> */

template <> void ApplyMask<float16, uint8_t, CPUContext>(
    const int               count,
    const float             scale,
    const float16*          x,
    const uint8_t*          mask,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

}  // namespace kernel

}  // namepsace dragon