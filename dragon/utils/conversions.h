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

template <typename DestType, typename SrcType>
CONVERSIONS_DECL DestType To(SrcType val1, SrcType val2);

template <>
CONVERSIONS_DECL float16 To<float16, float16>(float16 val) {
  return val;
}

template <>
CONVERSIONS_DECL bfloat16 To<bfloat16, bfloat16>(bfloat16 val) {
  return val;
}

template <>
CONVERSIONS_DECL float16 To<float16, float>(float val) {
  float16 ret;
  unsigned int x, remainder, result;
  std::memcpy(&x, &val, sizeof(val));
  unsigned int u = (x & 0x7fffffffU);
  unsigned int sign = ((x >> 16U) & 0x8000U);
  if (u >= 0x7f800000U) { // NaN/+Inf/-Inf
    remainder = 0U;
    result = ((u == 0x7f800000U) ? (sign | 0x7c00U) : 0x7fffU);
  } else if (u > 0x477fefffU) { // Overflows
    remainder = 0x80000000U;
    result = (sign | 0x7bffU);
  } else if (u >= 0x38800000U) { // Normal numbers
    remainder = u << 19U;
    u -= 0x38000000U;
    result = (sign | (u >> 13U));
  } else if (u < 0x33000001U) { // +0/-0
    remainder = u;
    result = sign;
  } else { // Denormal numbers
    const unsigned int exponent = u >> 23U;
    const unsigned int shift = 0x7eU - exponent;
    unsigned int mantissa = (u & 0x7fffffU);
    mantissa |= 0x800000U;
    remainder = mantissa << (32U - shift);
    result = (sign | (mantissa >> shift));
    result &= 0x0000FFFFU;
  }
  ret.x = static_cast<unsigned short>(result);
  if ((remainder > 0x80000000U) ||
      ((remainder == 0x80000000U) && ((ret.x & 0x1U) != 0U))) {
    ret.x++;
  }
  return ret;
}

template <>
CONVERSIONS_DECL bfloat16 To<bfloat16, float>(float val) {
  bfloat16 ret;
  unsigned int x, remainder;
  std::memcpy(&x, &val, sizeof(val));
  if ((x & 0x7fffffffU) > 0x7f800000U) {
    remainder = 0U;
    ret.x = static_cast<unsigned short>(0x7fffU);
  } else {
    remainder = x << 16U;
    ret.x = static_cast<unsigned short>(x >> 16U);
  }
  if ((remainder > 0x80000000U) ||
      ((remainder == 0x80000000U) && ((ret.x & 0x1U) != 0U))) {
    ret.x++;
  }
  return ret;
}

template <>
CONVERSIONS_DECL float To<float, float16>(float16 val) {
  float ret;
  unsigned int sign = ((static_cast<unsigned int>(val.x) >> 15U) & 1U);
  unsigned int exponent = ((static_cast<unsigned int>(val.x) >> 10U) & 0x1fU);
  unsigned int mantissa = ((static_cast<unsigned int>(val.x) & 0x3ffU) << 13U);
  if (exponent == 0x1fU) { /* NaN or Inf */
    /* discard sign of a NaN */
    sign = ((mantissa != 0U) ? (sign >> 1U) : sign);
    mantissa = ((mantissa != 0U) ? 0x7fffffU : 0U);
    exponent = 0xffU;
  } else if (exponent == 0U) { /* Denorm or Zero */
    if (mantissa != 0U) {
      unsigned int msb;
      exponent = 0x71U;
      do {
        msb = (mantissa & 0x400000U);
        mantissa <<= 1U; /* normalize */
        --exponent;
      } while (msb == 0U);
      mantissa &= 0x7fffffU; /* 1.mantissa is implicit */
    }
  } else {
    exponent += 0x70U;
  }
  const unsigned int u = ((sign << 31U) | (exponent << 23U) | mantissa);
  std::memcpy(&ret, &u, sizeof(u));
  return ret;
}

template <>
CONVERSIONS_DECL float To<float, bfloat16>(bfloat16 val) {
  float ret;
  const unsigned int u = static_cast<unsigned int>(val.x) << 16;
  std::memcpy(&ret, &u, sizeof(ret));
  return ret;
}

template <>
CONVERSIONS_DECL float16 To<float16, double>(double val) {
  return To<float16>(static_cast<float>(val));
}

template <>
CONVERSIONS_DECL bfloat16 To<bfloat16, double>(double val) {
  return To<bfloat16>(static_cast<float>(val));
}

#if defined(__CUDACC__)
template <>
CONVERSIONS_DECL float16 To<float16, half>(half val) {
  return float16{__half_raw(val).x};
}

template <>
CONVERSIONS_DECL bfloat16 To<bfloat16, nv_bfloat16>(nv_bfloat16 val) {
  return bfloat16{__nv_bfloat16_raw(val).x};
}

template <>
CONVERSIONS_DECL float To<float, half>(half val) {
  return __half2float(val);
}

template <>
CONVERSIONS_DECL float To<float, nv_bfloat16>(nv_bfloat16 val) {
  return __bfloat162float(val);
}

template <>
CONVERSIONS_DECL half To<half, half>(half val) {
  return val;
}

template <>
CONVERSIONS_DECL half To<half, float16>(float16 val) {
  return __half_raw{val.x};
}

template <>
CONVERSIONS_DECL half To<half, nv_bfloat16>(nv_bfloat16 val) {
  return __float2half(__bfloat162float(val));
}

template <>
CONVERSIONS_DECL half To<half, float>(float val) {
  return __float2half(val);
}

template <>
CONVERSIONS_DECL half To<half, double>(double val) {
  return To<half>(static_cast<float>(val));
}

template <>
CONVERSIONS_DECL nv_bfloat16 To<nv_bfloat16, half>(half val) {
  return __float2bfloat16(__half2float(val));
}

template <>
CONVERSIONS_DECL nv_bfloat16 To<nv_bfloat16, bfloat16>(bfloat16 val) {
  return __nv_bfloat16_raw{val.x};
}

template <>
CONVERSIONS_DECL nv_bfloat16 To<nv_bfloat16, nv_bfloat16>(nv_bfloat16 val) {
  return val;
}

template <>
CONVERSIONS_DECL nv_bfloat16 To<nv_bfloat16, float>(float val) {
  return __float2bfloat16(val);
}

template <>
CONVERSIONS_DECL nv_bfloat16 To<nv_bfloat16, double>(double val) {
  return To<nv_bfloat16>(static_cast<float>(val));
}

template <>
CONVERSIONS_DECL half2 To<half2, float16>(float16 val) {
  return half2(__half2_raw{val.x, val.x});
}

template <>
CONVERSIONS_DECL half2 To<half2, float>(float val) {
  return __float2half2_rn(val);
}

template <>
CONVERSIONS_DECL half2 To<half2, float>(float val1, float val2) {
  return __floats2half2_rn(val1, val2);
}

template <>
CONVERSIONS_DECL nv_bfloat162 To<nv_bfloat162, bfloat16>(bfloat16 val) {
  return nv_bfloat162(__nv_bfloat162_raw{val.x, val.x});
}

template <>
CONVERSIONS_DECL nv_bfloat162 To<nv_bfloat162, float>(float val) {
  return __float2bfloat162_rn(val);
}

template <>
CONVERSIONS_DECL nv_bfloat162 To<nv_bfloat162, float>(float val1, float val2) {
  return __floats2bfloat162_rn(val1, val2);
}

template <>
CONVERSIONS_DECL float2 To<float2, half2>(half2 val) {
  return __half22float2(val);
}

template <>
CONVERSIONS_DECL float2 To<float2, nv_bfloat162>(nv_bfloat162 val) {
#if __CUDA_ARCH__ >= 800
  return __bfloat1622float2(val);
#else
  return float2({__bfloat162float(val.x), __bfloat162float(val.y)});
#endif
}
#endif // defined(__CUDACC__)

#if defined(__mlu_func__)
template <typename DstT, typename SrcT>
__mlu_func__ void To(DstT* dst, SrcT* src, int count) {
  for (int i = 0; i < count; ++i) {
    dst[i] = DstT(src[i]);
  }
}

template <>
__mlu_func__ void To<uint8_t, int>(uint8_t* dst, int* src, int count) {
  __bang_int322uchar(dst, src, count, 0);
}

template <>
__mlu_func__ void To<uint8_t, half>(uint8_t* dst, half* src, int count) {
  __bang_half2uchar_dn(dst, src, count);
}

template <>
__mlu_func__ void To<uint8_t, float>(uint8_t* dst, float* src, int count) {
  __bang_float2uchar(dst, src, count);
}

template <>
__mlu_func__ void To<int8_t, int>(int8_t* dst, int* src, int count) {
  __bang_int322int8(dst, src, count, 0, 0);
}

template <>
__mlu_func__ void To<int8_t, float>(int8_t* dst, float* src, int count) {
  __bang_float2int8_rn(dst, src, count, 0);
}

template <>
__mlu_func__ void To<char, int>(char* dst, int* src, int count) {
  __bang_int322int8((int8_t*)dst, src, count, 0, 0);
}

template <>
__mlu_func__ void To<char, float>(char* dst, float* src, int count) {
  __bang_float2int8_rn((int8_t*)dst, src, count, 0);
}

template <>
__mlu_func__ void To<int, uint8_t>(int* dst, uint8_t* src, int count) {
  __bang_uchar2int32(dst, src, count, 0);
}

template <>
__mlu_func__ void To<int, int8_t>(int* dst, int8_t* src, int count) {
  __bang_int82int32(dst, src, count, 0, 0);
}

template <>
__mlu_func__ void To<int, char>(int* dst, char* src, int count) {
  __bang_int82int32(dst, (int8_t*)src, count, 0, 0);
}

template <>
__mlu_func__ void To<int, float>(int* dst, float* src, int count) {
  __bang_float2int32(dst, src, count, 0);
}

template <>
__mlu_func__ void To<half, uint8_t>(half* dst, uint8_t* src, int count) {
  __bang_uchar2half(dst, src, count);
}

template <>
__mlu_func__ void To<half, int8_t>(half* dst, int8_t* src, int count) {
  __bang_int82half(dst, src, count, 0);
}

template <>
__mlu_func__ void To<half, char>(half* dst, char* src, int count) {
  __bang_int82half(dst, (int8_t*)src, count, 0);
}

template <>
__mlu_func__ void To<half, int>(half* dst, int* src, int count) {
  __bang_int322half(dst, src, count, 0);
}

template <>
__mlu_func__ void To<half, float>(half* dst, float* src, int count) {
  __bang_float2half_rn(dst, src, count);
}

template <>
__mlu_func__ void To<float, uint8_t>(float* dst, uint8_t* src, int count) {
  __bang_uchar2float(dst, src, count);
}

template <>
__mlu_func__ void To<float, int8_t>(float* dst, int8_t* src, int count) {
  __bang_int82float(dst, src, count, 0);
}

template <>
__mlu_func__ void To<float, char>(float* dst, char* src, int count) {
  __bang_int82float(dst, (int8_t*)src, count, 0);
}

template <>
__mlu_func__ void To<float, int>(float* dst, int* src, int count) {
  __bang_int322float(dst, src, count, 0);
}

template <>
__mlu_func__ void To<float, half>(float* dst, half* src, int count) {
  __bang_half2float(dst, src, count);
}
#endif // defined(__mlu_func__)

} // namespace convert

} // namespace dragon

#endif // DRAGON_UTILS_CONVERSIONS_H_
