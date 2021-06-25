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

#ifndef DRAGON_UTILS_CONVERSIONS_H_
#define DRAGON_UTILS_CONVERSIONS_H_

#include "dragon/core/types.h"
#include "dragon/utils/device/common_cuda.h"

#if defined(__CUDACC__)
#define CONVERSIONS_DECL inline __host__ __device__
#else
#define CONVERSIONS_DECL inline
#endif

namespace dragon {

namespace convert {

template <typename DestType, typename SrcType>
CONVERSIONS_DECL DestType To(SrcType val) {
  return static_cast<DestType>(val);
}

template <>
inline float16 To<float16, float16>(float16 val) {
  return val;
}

template <>
inline float16 To<float16, float>(float val) {
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
inline float To<float, float16>(float16 val) {
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

template <>
inline float16 To<float16, double>(double val) {
  return To<float16>(static_cast<float>(val));
}

#ifdef USE_CUDA

template <>
CONVERSIONS_DECL float16 To<float16, half>(half val) {
  return float16{__half_raw(val).x};
}

template <>
CONVERSIONS_DECL half To<half, float16>(float16 val) {
  return __half_raw{val.x};
}

template <>
CONVERSIONS_DECL half2 To<half2, float16>(float16 val) {
  return half2(__half2_raw{val.x, val.x});
}

template <>
CONVERSIONS_DECL half To<half, half>(half val) {
  return val;
}

template <>
CONVERSIONS_DECL float To<float, half>(half val) {
  return __half2float(val);
}

template <>
CONVERSIONS_DECL half To<half, float>(float val) {
#if CUDA_VERSION_MIN(9, 2)
  return __float2half(val);
#else
#if defined(__CUDA_ARCH__)
#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short*>(&(var)))
  __half ret;
  asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(__HALF_TO_US(ret)) : "f"(val));
  return ret;
#undef __HALF_TO_US
#else
  return To<half>(To<float16>(val));
#endif
#endif
}

template <>
CONVERSIONS_DECL half2 To<half2, float>(float val) {
#if CUDA_VERSION_MIN(9, 2)
  return __float2half2_rn(val);
#else
#if defined(__CUDA_ARCH__)
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int*>(&(var)))
  __half2 ret;
  asm("{.reg .f16 low;\n"
      "  cvt.rn.f16.f32 low, %1;\n"
      "  mov.b32 %0, {low,low};}\n"
      : "=r"(__HALF2_TO_UI(ret))
      : "f"(val));
  return ret;
#undef __HALF2_TO_UI
#else
  return To<half2>(To<float16>(val));
#endif
#endif
}

template <>
CONVERSIONS_DECL half To<half, double>(double val) {
  return To<half>(static_cast<float>(val));
}

#endif // USE_CUDA

} // namespace convert

} // namespace dragon

#endif // DRAGON_UTILS_CONVERSIONS_H_
