#include "dragon/core/tensor.h"
#include "dragon/core/context_cuda.h"
#include "dragon/core/context_mps.h"

namespace dragon {

template <>
DRAGON_API const void* Tensor::raw_data<CPUContext>() {
  return memory(true)->cpu_data(nbytes(), offset_);
}

#ifdef USE_CUDA
template <>
DRAGON_API const void* Tensor::raw_data<CUDAContext>() {
  return memory(true)->cuda_data(nbytes(), offset_);
}
#endif

#ifdef USE_MPS
template <>
DRAGON_API const void* Tensor::raw_data<MPSContext>() {
  CHECK(!offset_) << "\nOffset is not supported";
  return memory(true)->mps_data(nbytes());
}
#endif

template <>
DRAGON_API void Tensor::raw_mutable_data<CPUContext>(void** data_ptr) {
  auto* mem_ptr = memory();
  *data_ptr = mem_ptr ? mem_ptr->mutable_cpu_data(nbytes()) : nullptr;
}

#ifdef USE_CUDA
template <>
DRAGON_API void Tensor::raw_mutable_data<CUDAContext>(void** data_ptr) {
  auto* mem_ptr = memory();
  *data_ptr = mem_ptr ? mem_ptr->mutable_cuda_data(nbytes()) : nullptr;
}
#endif

#ifdef USE_MPS
template <>
DRAGON_API void Tensor::raw_mutable_data<MPSContext>(void** data_ptr) {
  auto* mem_ptr = memory();
  *data_ptr = mem_ptr ? mem_ptr->mutable_mps_data(nbytes()) : nullptr;
}
#endif

} // namespace dragon
