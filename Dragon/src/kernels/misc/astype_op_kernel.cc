#include "utils/op_kernel.h"
#include "utils/cast.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! Astype <Ta = ?, Tb = ?, Device = CPU> */

template <typename Ta, typename Tb>
void _TypeA2B(const int count, const Ta* a, Tb* b) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        b[i] = static_cast<Tb>(a[i]);
    }
}

template <typename Ta, typename Tb>
void _TypeA2B_v2(const int count, const Ta* a, Tb* b) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        b[i] = dragon_cast<Tb, Ta>(a[i]);
    }
}

#define DEFINE_TYPE_A2B(type_a, type_b) \
    template <> void TypeA2B<type_a, type_b, CPUContext>( \
        const int           count, \
        const type_a*       a, \
        type_b*             b, \
        CPUContext*         ctx) { \
        _TypeA2B<type_a, type_b>(count, a, b); \
    }

#define DEFINE_TYPE_A2B_V2(type_a, type_b) \
    template <> void TypeA2B<type_a, type_b, CPUContext>( \
        const int           count, \
        const type_a*       a, \
        type_b*             b, \
        CPUContext*         ctx) { \
        _TypeA2B_v2<type_a, type_b>(count, a, b); \
    }

#define DEFINE_TYPE_DISABLE_FP16(type) \
    template <> void TypeA2B<float16, type, CPUContext>( \
        const int           count, \
        const float16*      a, \
        type*               b, \
        CPUContext*         ctx) { \
        CPU_FP16_NOT_SUPPORTED; \
    } \
    template <> void TypeA2B<type, float16, CPUContext>( \
        const int           count, \
        const type*         a, \
        float16*            b, \
        CPUContext*         ctx) { \
        CPU_FP16_NOT_SUPPORTED; \
    }

#define DEFINE_TYPE_A2ALL(type_a) \
    DEFINE_TYPE_A2B(type_a, float); \
    DEFINE_TYPE_A2B(type_a, double); \
    DEFINE_TYPE_A2B(type_a, int); \
    DEFINE_TYPE_A2B(type_a, int64_t); \
    DEFINE_TYPE_A2B(type_a, uint8_t);

DEFINE_TYPE_A2B_V2(float16, float);
DEFINE_TYPE_A2B_V2(float, float16);
DEFINE_TYPE_A2B_V2(float16, float16);
DEFINE_TYPE_A2ALL(float);
DEFINE_TYPE_A2ALL(double); DEFINE_TYPE_DISABLE_FP16(double);
DEFINE_TYPE_A2ALL(int); DEFINE_TYPE_DISABLE_FP16(int);
DEFINE_TYPE_A2ALL(int64_t); DEFINE_TYPE_DISABLE_FP16(int64_t);
DEFINE_TYPE_A2ALL(uint8_t); DEFINE_TYPE_DISABLE_FP16(uint8_t);

}  // namespace kernel

}  // namepsace dragon