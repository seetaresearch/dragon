/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *      <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_UTILS_SSE_DEVICE_H_
#define DRAGON_UTILS_SSE_DEVICE_H_

#ifdef WITH_SSE

#include <immintrin.h>
#include <tmmintrin.h>
#include <cstdint>

namespace dragon {

#define SSE_LOOP1(i, n) \
  for (i = 0; i < n - 4; i += 4) \

#define SSE_LOOP2(i, n) \
  for (; i < n; ++i)

#define SSE_FP32_LOAD _mm_loadu_ps
#define SSE_FP32_STORE _mm_storeu_ps
#define SSE_FP32_ADD _mm_add_ps
#define SSE_FP32_SUB _mm_sub_ps
#define SSE_FP32_MUL _mm_mul_ps
#define SSE_FP32_DIV _mm_div_ps
#define SSE_FP32_MAX _mm_max_ps
#define SSE_FP32_ZERO _mm_setzero_ps()
#define SSE_FP32_SCALAR _mm_setscalar_ps

#define SSE_INT32_SCALAR _mm_setscalar_epi
#define SSE_INT128_STORE _mm_storeu_si128

inline __m128 _mm_setscalar_ps(const float scalar) {
    return _mm_set_ps(scalar, scalar, scalar, scalar);
}

inline __m128i _mm_setscalar_epi(const int scalar) {
    return _mm_set_epi32(scalar, scalar, scalar, scalar);
}

}  // namespace dragon

#endif  // WITH_SSE

#endif  // DRAGON_UTILS_SSE_DEVICE_H_