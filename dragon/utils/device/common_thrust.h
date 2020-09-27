#ifndef DRAGON_UTILS_DEVICE_COMMON_THRUST_H_
#define DRAGON_UTILS_DEVICE_COMMON_THRUST_H_

#ifdef USE_CUDA

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#endif // USE_CUDA

#endif // DRAGON_UTILS_DEVICE_COMMON_THRUST_H_
