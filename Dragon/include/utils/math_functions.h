// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_UTILS_MATH_FUNCTIONS_H_
#define DRAGON_UTILS_MATH_FUNCTIONS_H_

#include <float.h>
#include <cstdint>
#include <climits>

#ifdef WITH_BLAS
extern "C" {
#include <cblas.h>
}
#else    // WITH_BLAS
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112} CBLAS_TRANSPOSE;
#endif

#include "protos/dragon.pb.h"

namespace dragon {

namespace math {

/******************** Level-0 ********************/

template <typename T, class Context>
void Set(const int n, const T alpha, T* x);

template <typename T, class Context>
void RandomUniform(const int n, const float low, const float high, T *x);

template <typename T, class Context>
void RandomNormal(const int n, const float mu, const float sigma, T* x);

template <typename T, class Context>
void RandomTruncatedNormal(const int n, 
                           const float mu,
                           const float sigma,
                           const float low,
                           const float high,
                           T* x);

template <typename T, class Context>
void RandomBernoulli(const int n, const float p, uint32_t* x);

/******************** Level-1 ********************/

template <typename T, class Context>
void Add(const int n, const T* a, const T* b, T* y);

template <typename T, class Context>
void Sub(const int n, const T* a, const T* b, T* y);

template <typename T, class Context>
void Mul(const int n, const T* a, const T* b, T* y);

template <typename T, class Context>
void Div(const int n, const T* a, const T* b, T* y);

template <typename T, class Context>
void Clip(const int n, const float low, const float high, T* x);

template <typename T, class Context>
void Exp(const int n, const T* x, T* y);

template <typename T, class Context>
void Log(const int n, const T* x, T* y);

template <typename T, class Context>
void Square(const int n, const T* x, T* y);

template <typename T, class Context>
void Sqrt(const int n, const T* x, T* y);

template <typename T, class Context>
void Pow(const int n, const float alpha, const T* x, T* y);

template <typename T, class Context>
void Inv(const int n, const float numerator, const T* x, T* y);

/******************** Level-2 ********************/

template <typename T, class Context>
void Scal(const int n, const float alpha, T* y);

template <typename T, class Context>
void Scale(const int n, const float alpha, const T* x, T* y);

template <typename T, class Context>
T StridedDot(const int n,
             const T* a,
             const int incx,
             const T* b,
             const int incy);

template <typename T, class Context>
float Dot(const int n, const T* a, const T* b);

template<typename T, class Context>
float ASum(const int n, const T *x);

template<typename T, class Context>
void AddScalar(const int n, const float alpha, T* y);

template<typename T, class Context>
void MulScalar(const int n, const float alpha, T* y);

template<typename T, class Context>
void Axpy(const int n, float alpha, const T* x, T *y);

template<typename T, class Context>
void Axpby(const int n, float alpha, const T* x, float beta, T *y);

/******************** Level-3 ********************/

template <typename T, class Context>
void Gemm(const CBLAS_TRANSPOSE transA,
          const CBLAS_TRANSPOSE transB,
          const int M,
          const int N,
          const int K,
          const float alpha,
          const T* A,
          const T* B,
          const float beta,
          T* C,
          TensorProto_DataType math_type = TensorProto_DataType_FLOAT);

template<typename T, class Context>
void Gemv(const CBLAS_TRANSPOSE transA,
          const int M,
          const int N,
          const float alpha,
          const T* A,
          const T* x,
          const float beta,
          T* y,
          TensorProto_DataType math_type = TensorProto_DataType_FLOAT);

}    // namespace math

}    // namespace dragon

#endif    // DRAGON_UTILS_MATH_FUNCTIONS_H_
