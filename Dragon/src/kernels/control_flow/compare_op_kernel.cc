 #include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! Equal <T = ?, Device = CPU> */

template <typename T>
void _EqualInteger(
    const int               count,
    const T*                a,
    const T*                b,
    bool*                   y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = a[i] == b[i] ? true : false;
    }
}

template <typename T>
void _EqualFloat(
    const int               count,
    const T*                a,
    const T*                b,
    bool*                   y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = fabs(a[i] - b[i]) < 1e-15 ? true : false;
    }
}

/*! Less <T = ?, Device = CPU> */

template <typename T>
void _Less(
    const int               count,
    const T*                a,
    const T*                b,
    bool*                   y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = a[i] < b[i] ? true : false;
    }
}

/*! Greater <T = ?, Device = CPU> */

template <typename T>
void _Greater(
    const int               count,
    const T*                a,
    const T*                b,
    bool*                   y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = a[i] > b[i] ? true : false;
    }
}

#define DEFINE_COMPARE_WARPPER(T, OP, IMPL) \
    template <> void OP<T, CPUContext>( \
        const int               count, \
        const T*                a, \
        const T*                b, \
        bool*                   y, \
        CPUContext*             ctx) { \
        IMPL<T>(count, a, b, y); \
    }

DEFINE_COMPARE_WARPPER(bool, Equal, _EqualInteger);
DEFINE_COMPARE_WARPPER(int8_t, Equal, _EqualInteger);
DEFINE_COMPARE_WARPPER(uint8_t, Equal, _EqualInteger);
DEFINE_COMPARE_WARPPER(int, Equal, _EqualInteger);
DEFINE_COMPARE_WARPPER(int64_t, Equal, _EqualInteger);
DEFINE_COMPARE_WARPPER(float, Equal, _EqualFloat);
DEFINE_COMPARE_WARPPER(double, Equal, _EqualFloat);

DEFINE_COMPARE_WARPPER(bool, Less, _Less);
DEFINE_COMPARE_WARPPER(int8_t, Less, _Less);
DEFINE_COMPARE_WARPPER(uint8_t, Less, _Less);
DEFINE_COMPARE_WARPPER(int, Less, _Less);
DEFINE_COMPARE_WARPPER(int64_t, Less, _Less);
DEFINE_COMPARE_WARPPER(float, Less, _Less);
DEFINE_COMPARE_WARPPER(double, Less, _Less);

DEFINE_COMPARE_WARPPER(bool, Greater, _Greater);
DEFINE_COMPARE_WARPPER(int8_t, Greater, _Greater);
DEFINE_COMPARE_WARPPER(uint8_t, Greater, _Greater);
DEFINE_COMPARE_WARPPER(int, Greater, _Greater);
DEFINE_COMPARE_WARPPER(int64_t, Greater, _Greater);
DEFINE_COMPARE_WARPPER(float, Greater, _Greater);
DEFINE_COMPARE_WARPPER(double, Greater, _Greater);

template <> void Equal<float16, CPUContext>(
    const int               count,
    const float16*          a,
    const float16*          b,
    bool*                   y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Less<float16, CPUContext>(
    const int               count,
    const float16*          a,
    const float16*          b,
    bool*                   y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Greater<float16, CPUContext>(
    const int               count,
    const float16*          a,
    const float16*          b,
    bool*                   y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

#undef DEFINE_COMPARE_WARPPER

}  // namespace kernel

}  // namepsace dragon