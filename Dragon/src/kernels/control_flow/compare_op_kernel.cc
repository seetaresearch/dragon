 #include "utils/op_kernel.h"
#include "utils/eigen_utils.h"

namespace dragon {

namespace kernel {

/* Kernel Launchers */

#define DEFINE_NOTZERO_KERNEL_LAUNCHER(T) \
    template <> void NotZero<T, CPUContext>( \
        const int               count, \
        const T*                x, \
        bool*                   y, \
        CPUContext*             ctx) { \
        EigenVectorArrayMap<bool>(y, count) = \
            ConstEigenVectorArrayMap<T>(x, count) != T(0); \
    }

#define DEFINE_COMPARE_KERNEL_LAUNCHER(T, OP, expr) \
    template <> void OP<T, CPUContext>( \
        const int               count, \
        const T*                a, \
        const T*                b, \
        bool*                   y, \
        CPUContext*             ctx) { \
        EigenVectorArrayMap<bool>(y, count) = \
            ConstEigenVectorArrayMap<T>(a, count) expr \
                ConstEigenVectorArrayMap<T>(b, count); \
    }

DEFINE_NOTZERO_KERNEL_LAUNCHER(bool);
DEFINE_NOTZERO_KERNEL_LAUNCHER(int8_t);
DEFINE_NOTZERO_KERNEL_LAUNCHER(uint8_t);
DEFINE_NOTZERO_KERNEL_LAUNCHER(int);
DEFINE_NOTZERO_KERNEL_LAUNCHER(int64_t);
DEFINE_NOTZERO_KERNEL_LAUNCHER(float);
DEFINE_NOTZERO_KERNEL_LAUNCHER(double);

DEFINE_COMPARE_KERNEL_LAUNCHER(bool, Equal, ==);
DEFINE_COMPARE_KERNEL_LAUNCHER(int8_t, Equal, ==);
DEFINE_COMPARE_KERNEL_LAUNCHER(uint8_t, Equal, ==);
DEFINE_COMPARE_KERNEL_LAUNCHER(int, Equal, ==);
DEFINE_COMPARE_KERNEL_LAUNCHER(int64_t, Equal, ==);
DEFINE_COMPARE_KERNEL_LAUNCHER(float, Equal, ==);
DEFINE_COMPARE_KERNEL_LAUNCHER(double, Equal, ==);

DEFINE_COMPARE_KERNEL_LAUNCHER(bool, NotEqual, !=);
DEFINE_COMPARE_KERNEL_LAUNCHER(int8_t, NotEqual, !=);
DEFINE_COMPARE_KERNEL_LAUNCHER(uint8_t, NotEqual, !=);
DEFINE_COMPARE_KERNEL_LAUNCHER(int, NotEqual, !=);
DEFINE_COMPARE_KERNEL_LAUNCHER(int64_t, NotEqual, !=);
DEFINE_COMPARE_KERNEL_LAUNCHER(float, NotEqual, != );
DEFINE_COMPARE_KERNEL_LAUNCHER(double, NotEqual, !=);

DEFINE_COMPARE_KERNEL_LAUNCHER(bool, Less, <);
DEFINE_COMPARE_KERNEL_LAUNCHER(int8_t, Less, <);
DEFINE_COMPARE_KERNEL_LAUNCHER(uint8_t, Less, <);
DEFINE_COMPARE_KERNEL_LAUNCHER(int, Less, <);
DEFINE_COMPARE_KERNEL_LAUNCHER(int64_t, Less, <);
DEFINE_COMPARE_KERNEL_LAUNCHER(float, Less, <);
DEFINE_COMPARE_KERNEL_LAUNCHER(double, Less, <);

DEFINE_COMPARE_KERNEL_LAUNCHER(bool, LessEqual, <=);
DEFINE_COMPARE_KERNEL_LAUNCHER(int8_t, LessEqual, <=);
DEFINE_COMPARE_KERNEL_LAUNCHER(uint8_t, LessEqual, <=);
DEFINE_COMPARE_KERNEL_LAUNCHER(int, LessEqual, <=);
DEFINE_COMPARE_KERNEL_LAUNCHER(int64_t, LessEqual, <=);
DEFINE_COMPARE_KERNEL_LAUNCHER(float, LessEqual, <=);
DEFINE_COMPARE_KERNEL_LAUNCHER(double, LessEqual, <=);

DEFINE_COMPARE_KERNEL_LAUNCHER(bool, Greater, >);
DEFINE_COMPARE_KERNEL_LAUNCHER(int8_t, Greater, >);
DEFINE_COMPARE_KERNEL_LAUNCHER(uint8_t, Greater, >);
DEFINE_COMPARE_KERNEL_LAUNCHER(int, Greater, >);
DEFINE_COMPARE_KERNEL_LAUNCHER(int64_t, Greater, >);
DEFINE_COMPARE_KERNEL_LAUNCHER(float, Greater, >);
DEFINE_COMPARE_KERNEL_LAUNCHER(double, Greater, >);

DEFINE_COMPARE_KERNEL_LAUNCHER(bool, GreaterEqual, >=);
DEFINE_COMPARE_KERNEL_LAUNCHER(int8_t, GreaterEqual, >=);
DEFINE_COMPARE_KERNEL_LAUNCHER(uint8_t, GreaterEqual, >=);
DEFINE_COMPARE_KERNEL_LAUNCHER(int, GreaterEqual, >=);
DEFINE_COMPARE_KERNEL_LAUNCHER(int64_t, GreaterEqual, >=);
DEFINE_COMPARE_KERNEL_LAUNCHER(float, GreaterEqual, >=);
DEFINE_COMPARE_KERNEL_LAUNCHER(double, GreaterEqual, >=);

template <> void NotZero<float16, CPUContext>(
    const int               count,
    const float16*          x,
    bool*                   y,
    CPUContext* ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void Equal<float16, CPUContext>(
    const int               count,
    const float16*          a,
    const float16*          b,
    bool*                   y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void NotEqual<float16, CPUContext>(
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

template <> void LessEqual<float16, CPUContext>(
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

template <> void GreaterEqual<float16, CPUContext>(
    const int               count,
    const float16*          a,
    const float16*          b,
    bool*                   y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

#undef DEFINE_NOTZERO_KERNEL_LAUNCHER
#undef DEFINE_COMPARE_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon