/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_UTILS_CAST_H_
#define DRAGON_UTILS_CAST_H_

#include <cstring>

#include "dragon/core/types.h"
#include "dragon/utils/device/common_cuda.h"

namespace dragon {

#define HFLT_MAX 65504.F
#define HFLT_MIN 6.10e-5F

namespace cast {

template <typename DType, typename SType>
DType to(SType val) {
  return static_cast<DType>(val);
}

template <>
inline float16 to<float16, float>(float val) {
  float16 ret;
  unsigned* xp = reinterpret_cast<unsigned int*>(&val);
  unsigned x = *xp;
  unsigned u = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
  unsigned sign, exponent, mantissa;
  // Get rid of +NaN/-NaN case first.
  if (u > 0x7f800000) {
    ret.x = 0x7fffU;
    return ret;
  }
  sign = ((x >> 16) & 0x8000);
  // Get rid of +Inf/-Inf, +0/-0.
  if (u > 0x477fefff) {
    ret.x = sign | 0x7c00U;
    return ret;
  }
  if (u < 0x33000001) {
    ret.x = (sign | 0x0000);
    return ret;
  }
  exponent = ((u >> 23) & 0xff);
  mantissa = (u & 0x7fffff);
  if (exponent > 0x70) {
    shift = 13;
    exponent -= 0x70;
  } else {
    shift = 0x7e - exponent;
    exponent = 0;
    mantissa |= 0x800000;
  }
  lsb = (1 << shift);
  lsb_s1 = (lsb >> 1);
  lsb_m1 = (lsb - 1);
  // Round to nearest even.
  remainder = (mantissa & lsb_m1);
  mantissa >>= shift;
  if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
    ++mantissa;
    if (!(mantissa & 0x3ff)) {
      ++exponent;
      mantissa = 0;
    }
  }
  ret.x = (sign | (exponent << 10) | mantissa);
  return ret;
}

template <>
inline float to<float, float16>(float16 val) {
  unsigned sign = ((val.x >> 15) & 1);
  unsigned exponent = ((val.x >> 10) & 0x1f);
  unsigned mantissa = ((val.x & 0x3ff) << 13);

  if (exponent == 0x1f) { /* NaN or Inf */
    mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
    exponent = 0xff;
  } else if (!exponent) { /* Denorm or Zero */
    if (mantissa) {
      unsigned int msb;
      exponent = 0x71;
      do {
        msb = (mantissa & 0x400000);
        mantissa <<= 1; /* normalize */
        --exponent;
      } while (!msb);
      mantissa &= 0x7fffff; /* 1.mantissa is implicit */
    }
  } else {
    exponent += 0x70;
  }

  unsigned i = ((sign << 31) | (exponent << 23) | mantissa);
  float ret;
  memcpy(&ret, &i, sizeof(i));
  return ret;
}

#ifdef USE_CUDA

template <>
inline half to<half, float>(float val) {
#if CUDA_VERSION_MIN(9, 0, 0)
  __half_raw fp16_raw;
  fp16_raw.x = cast::to<float16>(val).x;
  return half(fp16_raw);
#else
  half fp16;
  fp16.x = dragon_cast<float16, float>(val).x;
  return fp16;
#endif
}

template <>
inline half2 to<half2, float>(float val) {
#if CUDA_VERSION_MIN(9, 0, 0)
  half fp16 = cast::to<half>(val);
  return half2(fp16, fp16);
#else
  half2 fp32;
  fp32.x = cast::to<float32>(val).x;
  return fp32;
#endif
}

template <>
inline half2 to<half2, float16>(float16 val) {
#if CUDA_VERSION_MIN(9, 0, 0)
  __half_raw fp16_raw;
  fp16_raw.x = val.x;
  return half2(half(fp16_raw), half(fp16_raw));
#else
  half2 fp32;
  fp32.x = dragon_cast<float32, float16>(val).x;
  return fp32;
#endif
}

template <>
inline half to<half, float16>(float16 val) {
#if CUDA_VERSION_MIN(9, 0, 0)
  __half_raw fp16_raw;
  fp16_raw.x = val.x;
  return fp16_raw;
#else
  half fp16;
  fp16.x = val.x;
  return fp16;
#endif
}

#endif // USE_CUDA

} // namespace cast

} // namespace dragon

#endif // DRAGON_UTILS_CAST_H_
