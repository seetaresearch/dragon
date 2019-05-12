#include "utils/op_kernel.h"
#include "utils/cast.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! Astype <Ta = ?, Tb = ?, Device = CPU> */

template <typename Ta, typename Tb>
void _TypeA2B(const int count, const Ta* a, Tb* b) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        b[i] = cast::to<Tb>(a[i]);
    }
}

#define DEFINE_TYPE_A_TO_B(Ta, Tb) \
    template <> void TypeA2B<Ta, Tb, CPUContext>( \
        const int           count, \
        const Ta*           a, \
        Tb*                 b, \
        CPUContext*         ctx) { \
        _TypeA2B(count, a, b); \
    }

#define DEFINE_TYPE_FP16_DISABLED(T) \
    template <> void TypeA2B<float16, T, CPUContext>( \
        const int           count, \
        const float16*      a, \
        T*                  b, \
        CPUContext*         ctx) { \
        LOG(FATAL) << "Not Implemented: float16 -> " \
                   << TypeMetaToString(TypeMeta::Make<T>()); \
    } \
    template <> void TypeA2B<T, float16, CPUContext>( \
        const int           count, \
        const T*            a, \
        float16*            b, \
        CPUContext*         ctx) { \
        LOG(FATAL) << "Not Implemented: " \
                   << TypeMetaToString(TypeMeta::Make<T>()) \
                   << " -> float16"; \
    }

#define DEFINE_TYPE_A_TO_ALL(type_a) \
    DEFINE_TYPE_A_TO_B(type_a, bool); \
    DEFINE_TYPE_A_TO_B(type_a, int8_t); \
    DEFINE_TYPE_A_TO_B(type_a, uint8_t); \
    DEFINE_TYPE_A_TO_B(type_a, int); \
    DEFINE_TYPE_A_TO_B(type_a, int64_t); \
    DEFINE_TYPE_A_TO_B(type_a, float); \
    DEFINE_TYPE_A_TO_B(type_a, double);

DEFINE_TYPE_A_TO_B(float16, float);
DEFINE_TYPE_A_TO_B(float, float16);
DEFINE_TYPE_A_TO_B(float16, float16);
DEFINE_TYPE_A_TO_ALL(bool); DEFINE_TYPE_FP16_DISABLED(bool);
DEFINE_TYPE_A_TO_ALL(uint8_t); DEFINE_TYPE_FP16_DISABLED(uint8_t);
DEFINE_TYPE_A_TO_ALL(int8_t); DEFINE_TYPE_FP16_DISABLED(int8_t);
DEFINE_TYPE_A_TO_ALL(int); DEFINE_TYPE_FP16_DISABLED(int);
DEFINE_TYPE_A_TO_ALL(int64_t); DEFINE_TYPE_FP16_DISABLED(int64_t);
DEFINE_TYPE_A_TO_ALL(float);
DEFINE_TYPE_A_TO_ALL(double); DEFINE_TYPE_FP16_DISABLED(double);

}  // namespace kernel

}  // namepsace dragon