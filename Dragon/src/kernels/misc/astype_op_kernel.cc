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
        b[i] = cast::to<Tb>(a[i]);
    }
}

#define DEFINE_TYPE_A_TO_B(type_a, type_b) \
    template <> void TypeA2B<type_a, type_b, CPUContext>( \
        const int           count, \
        const type_a*       a, \
        type_b*             b, \
        CPUContext*         ctx) { \
        _TypeA2B<type_a, type_b>(count, a, b); \
    }

#define DEFINE_TYPE_FP16_DISABLED(type) \
    template <> void TypeA2B<float16, type, CPUContext>( \
        const int           count, \
        const float16*      a, \
        type*               b, \
        CPUContext*         ctx) { \
        LOG(FATAL) << "Not Implemented: float16 -> " \
                   << TypeMetaToString(TypeMeta::Make<type>()); \
    } \
    template <> void TypeA2B<type, float16, CPUContext>( \
        const int           count, \
        const type*         a, \
        float16*            b, \
        CPUContext*         ctx) { \
        LOG(FATAL) << "Not Implemented: " \
                   << TypeMetaToString(TypeMeta::Make<type>()) \
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