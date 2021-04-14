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

#ifndef DRAGON_UTILS_MATH_ATOMICS_H_
#define DRAGON_UTILS_MATH_ATOMICS_H_

#include "dragon/utils/math/functional.h"

namespace dragon {

namespace math {

template <typename T, class MathFunctor, size_t kBits>
struct AtomicIntegerFunctor;

template <typename T, class MathFunctor>
struct AtomicIntegerFunctor<T, MathFunctor, 1> {
#if defined(__CUDACC__)
  inline __device__ void operator()(T* address, T val) {
    size_t offset = (size_t)address & 3;
    uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
    uint32_t old = *address_as_ui;
    uint32_t shift = offset * 8;
    uint32_t old_byte;
    uint32_t newval;
    uint32_t assumed;
    do {
      assumed = old;
      old_byte = (old >> shift) & 0xff;
      newval = static_cast<uint8_t>(math_functor_(val, old_byte));
      newval = (old & ~(0x000000ff << shift)) | (newval << shift);
      old = atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
  }
#endif
  MathFunctor math_functor_;
};

template <typename T, class MathFunctor>
struct AtomicIntegerFunctor<T, MathFunctor, 8> {
#if defined(__CUDACC__)
  inline __device__ void operator()(T* address, T val) {
    unsigned long long* address_as_ui = (unsigned long long*)address;
    unsigned long long old = *address_as_ui;
    unsigned long long newval;
    unsigned long long assumed;
    do {
      assumed = old;
      newval = static_cast<unsigned long long>(math_functor_(val, old));
      old = atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
  }
#endif
  MathFunctor math_functor_;
};

template <typename T, class MathFunctor>
struct AtomicFloat16Functor {
#if defined(__CUDACC__)
  inline __device__ void operator()(T* address, T val) {
    unsigned int* address_as_ui =
        (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui, assumed;
    __half_raw result;
    do {
      assumed = old;
      result.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
      result = math_functor_(half(result), val);
      old = (size_t)address & 2 ? (old & 0xffff) | (result.x << 16)
                                : (old & 0xffff0000) | result.x;
      old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
  }
#endif
  MathFunctor math_functor_;
};

template <typename T, class MathFunctor>
struct AtomicFloat64Functor {
#if defined(__CUDACC__)
  inline __device__ void operator()(T* address, T val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
      assumed = old;
      old = atomicCAS(
          address_as_ull,
          assumed,
          __double_as_longlong(
              math_functor_(val, __longlong_as_double(assumed))));
    } while (assumed != old);
  }
#endif
  MathFunctor math_functor_;
};

namespace utils {

#if defined(__CUDACC__)
template <typename T>
inline __device__ void AtomicAnd(T* address, T val) {
  atomicAnd(address, val);
}

inline __device__ void AtomicAnd(uint8_t* address, uint8_t val) {
  AtomicIntegerFunctor<uint8_t, BitAndFunctor<uint8_t>, sizeof(uint8_t)>()(
      address, val);
}

template <typename T>
inline __device__ void AtomicAdd(T* address, T val) {
  atomicAdd(address, val);
}

inline __device__ void AtomicAdd(uint8_t* address, uint8_t val) {
  AtomicIntegerFunctor<uint8_t, PlusFunctor<uint8_t>, sizeof(uint8_t)>()(
      address, val);
}

inline __device__ void AtomicAdd(int8_t* address, int8_t val) {
  AtomicIntegerFunctor<int8_t, PlusFunctor<uint8_t>, sizeof(int8_t)>()(
      address, val);
}

inline __device__ void AtomicAdd(int64_t* address, int64_t val) {
  AtomicIntegerFunctor<int64_t, PlusFunctor<int64_t>, sizeof(int64_t)>()(
      address, val);
}

#if __CUDA_ARCH__ < 700
inline __device__ void AtomicAdd(half* address, half val) {
  AtomicFloat16Functor<half, PlusFunctor<half>>()(address, val);
}
#endif

#if __CUDA_ARCH__ < 600
inline __device__ void AtomicAdd(double* address, double val) {
  AtomicFloat64Functor<double, PlusFunctor<double>>()(address, val);
}
#endif
#endif

} // namespace utils

} // namespace math

} // namespace dragon

#endif // DRAGON_UTILS_MATH_ATOMICS_H_
