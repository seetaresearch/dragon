#ifdef WITH_SSE

#include <cmath>
#include <algorithm>
#include <iostream>

#include "utils/sse_alternative.h"

namespace dragon {

namespace sse {

template<> void Set(
    const int               n,
    const float             alpha,
    float*                  x) {
    __m128 scalar = SSE_FP32_SCALAR(alpha);
    int32_t i = 0;
    SSE_LOOP1(i, n) SSE_FP32_STORE(x + i, scalar);
    SSE_LOOP2(i, n) x[i] = alpha;
}

template<> void Set(
    const int               n,
    const int               alpha,
    int*                    x) {
    __m128i scalar = SSE_INT32_SCALAR(alpha);
    __m128i* x1 = reinterpret_cast<__m128i*>(x);
    int32_t i = 0;
    SSE_LOOP1(i, n) SSE_INT128_STORE(x1++, scalar);
    SSE_LOOP2(i, n) x[i] = alpha;
}

template<> void Add(
    const int               n,
    const float*            a,
    const float*            b,
    float*                  y) {
    __m128 x1, y1, z1;
    int32_t i = 0;
    SSE_LOOP1(i, n) {
        x1 = SSE_FP32_LOAD(a + i);
        y1 = SSE_FP32_LOAD(b + i);
        z1 = SSE_FP32_ADD(x1, y1);
        SSE_FP32_STORE(y + i, z1);
    }
    SSE_LOOP2(i, n) y[i] = a[i] + b[i];
}

template<> void Sub(
    const int               n,
    const float*            a,
    const float*            b,
    float*                  y) {
    __m128 x1, y1, z1;
    int32_t i = 0;
    SSE_LOOP1(i, n) {
        x1 = SSE_FP32_LOAD(a + i);
        y1 = SSE_FP32_LOAD(b + i);
        z1 = SSE_FP32_SUB(x1, y1);
        SSE_FP32_STORE(y + i, z1);
    }
    SSE_LOOP2(i, n) y[i] = a[i] - b[i];
}

template<> void Mul(
    const int               n,
    const float*            a,
    const float*            b,
    float*                  y) {
    __m128 x1, y1, z1;
    int32_t i = 0;
    SSE_LOOP1(i, n) {
        x1 = SSE_FP32_LOAD(a + i);
        y1 = SSE_FP32_LOAD(b + i);
        z1 = SSE_FP32_MUL(x1, y1);
        SSE_FP32_STORE(y + i, z1);
    }
    SSE_LOOP2(i, n) y[i] = a[i] * b[i];
}

template<> void Div(
    const int               n,
    const float*            a,
    const float*            b,
    float*                  y) {
    __m128 x1, y1, z1;
    int32_t i = 0;
    SSE_LOOP1(i, n) {
        x1 = SSE_FP32_LOAD(a + i);
        y1 = SSE_FP32_LOAD(b + i);
        z1 = SSE_FP32_DIV(x1, y1);
        SSE_FP32_STORE(y + i, z1);
    }
    SSE_LOOP2(i, n) y[i] = a[i] / b[i];
}

template<> void Scal(
    const int               n,
    const float             alpha,
    float*                  y) {
    __m128 y1, scalar = SSE_FP32_SCALAR(alpha);
    int32_t i = 0;
    SSE_LOOP1(i, n) {
        y1 = SSE_FP32_LOAD(y + i);
        y1 = SSE_FP32_MUL(y1, scalar);
        SSE_FP32_STORE(y + i, y1);
    }
    SSE_LOOP2(i, n) y[i] *= alpha;
}

template<> void Scale(
    const int               n,
    const float             alpha,
    const float*            x,
    float*                  y) {
    __m128 x1, scalar = SSE_FP32_SCALAR(alpha);
    int32_t i = 0;
    SSE_LOOP1(i, n) {
        x1 = SSE_FP32_LOAD(x + i);
        x1 = SSE_FP32_MUL(x1, scalar);
        SSE_FP32_STORE(y + i, x1);
    }
    SSE_LOOP2(i, n) y[i] = x[i] * alpha;
}

template<> void Axpy(
    const int               n,
    const float             alpha,
    const float*            x,
    float*                  y) {
    __m128 x1, y1, scalar = SSE_FP32_SCALAR(alpha);
    int32_t i = 0;
    SSE_LOOP1(i, n) {
        x1 = SSE_FP32_LOAD(x + i);
        y1 = SSE_FP32_LOAD(y + i);
        x1 = SSE_FP32_MUL(x1, scalar);
        y1 = SSE_FP32_ADD(x1, y1);
        SSE_FP32_STORE(y + i, y1);
    }
    SSE_LOOP2(i, n) y[i] = alpha * x[i] + y[i];
}

template<> void Axpby(
    const int               n,
    const float             alpha,
    const float*            x,
    const float             beta,
    float*                  y) {
    __m128 x1, y1, z1;
    __m128 scalar1 = SSE_FP32_SCALAR(alpha);
    __m128 scalar2 = SSE_FP32_SCALAR(beta);
    int32_t i = 0;
    SSE_LOOP1(i, n) {
        x1 = SSE_FP32_LOAD(x + i);
        y1 = SSE_FP32_LOAD(y + i);
        x1 = SSE_FP32_MUL(x1, scalar1);
        y1 = SSE_FP32_MUL(y1, scalar2);
        z1 = SSE_FP32_ADD(x1, y1);
        SSE_FP32_STORE(y + i, z1);
    }
    SSE_LOOP2(i, n) y[i] = alpha * x[i] + beta* y[i];
}

template<> float ASum(
    const int               n,
    const float*            x) {
    __m128 x1, sum = SSE_FP32_ZERO;
    int32_t i = 0;
    SSE_LOOP1(i, n) {
        x1 = SSE_FP32_LOAD(x + i);
        sum = SSE_FP32_ADD(sum, x1);
    }
    float buf[4];
    SSE_FP32_STORE(buf, sum);
    float ret = buf[0] + buf[1] + buf[2] + buf[3];
    SSE_LOOP2(i, n) ret += x[i];
    return ret;
}

template<> void AddScalar(
    const int               n,
    const float             alpha,
    float*                  y) {
    __m128 y1, scalar = SSE_FP32_SCALAR(alpha);
    int32_t i = 0;
    SSE_LOOP1(i, n) {
         y1 = SSE_FP32_LOAD(y + i);
         y1 = SSE_FP32_ADD(y1, scalar);
         SSE_FP32_STORE(y + i, y1);
     }
     SSE_LOOP2(i, n) y[i] += alpha;
}

template<> void MulScalar(
    const int               n,
    const float             alpha,
    float*                  y) {
    __m128 y1, scalar = SSE_FP32_SCALAR(alpha);
    int32_t i = 0;
    SSE_LOOP1(i, n) {
        y1 = SSE_FP32_LOAD(y + i);
        y1 = SSE_FP32_MUL(y1, scalar);
        SSE_FP32_STORE(y + i, y1);
    }
    SSE_LOOP2(i, n) y[i] *= alpha;
}

template <> float Dot(
    const int               n,
    const float*            a,
    const float*            b) {
    __m128 x1, y1, sum = SSE_FP32_ZERO;
    int32_t i = 0;
    SSE_LOOP1(i, n) {
        x1 = SSE_FP32_LOAD(a + i);
        y1 = SSE_FP32_LOAD(b + i);
        sum = SSE_FP32_ADD(sum, SSE_FP32_MUL(x1, y1));
    }
    float buf[4];
    SSE_FP32_STORE(buf, sum);
    float ret = buf[0] + buf[1] + buf[2] + buf[3];
    SSE_LOOP2(i, n) ret += a[i] * b[i];
    return ret;
}

}    // namespace sse

}    // namespace dragon

#endif    // WITH_SSE