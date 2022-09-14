#include "dragon/core/memory.h"
#include "dragon/core/context.h"
#include "dragon/core/context_cuda.h"
#include "dragon/core/context_mps.h"

namespace dragon {

void UnifiedMemory::ToCPU(size_t size) {
  switch (state_) {
    case UNINITIALIZED:
      cpu_ptr_ = CPUContext::New(size_);
      CPUContext::Memset(size_, cpu_ptr_);
      state_ = STATE_AT_CPU;
      break;
    case STATE_AT_CUDA:
#ifdef USE_CUDA
      if (cpu_ptr_ == nullptr) {
        cpu_ptr_ = CPUContext::New(size_);
      }
      CUDAContext::Memcpy<CPUContext, CUDAContext>(
          size > 0 ? size : size_, cpu_ptr_, cuda_ptr_, device_id_);
      state_ = SYNCED;
      break;
#endif
    case STATE_AT_MPS:
#ifdef USE_MPS
      if (cpu_ptr_ == nullptr) {
        auto* mps_ptr = MPSContext::NewSharedFromBuffer(mps_ptr_);
        MPSContext::Delete(mps_ptr_);
        mps_ptr_ = mps_ptr;
        cpu_ptr_ = MTLGetBufferContents(mps_ptr_);
        own_cpu_ptr_ = false;
      } else {
        auto* stream = MPSContext::objects().stream(device_id_, 0);
        MPSContext::SynchronizeResource(stream, 0, mps_ptr_);
        MPSContext::SynchronizeStream(stream);
      }
      state_ = SYNCED;
      break;
#endif
    case STATE_AT_CPU:
    case SYNCED:
      break;
  }
}

void UnifiedMemory::ToCUDA(size_t size) {
#ifdef USE_CUDA
  switch (state_) {
    case UNINITIALIZED:
      cuda_ptr_ = CUDAContext::New(size_);
      CUDAContext::Memset(size_, cuda_ptr_);
      device_id_ = CUDAContext::current_device();
      state_ = STATE_AT_CUDA;
      break;
    case STATE_AT_CPU:
      if (cuda_ptr_ == nullptr) {
        cuda_ptr_ = CUDAContext::New(size_);
        device_id_ = CUDAContext::current_device();
      }
      CUDAContext::Memcpy<CUDAContext, CPUContext>(
          size > 0 ? size : size_, cuda_ptr_, cpu_ptr_, device_id_);
      state_ = SYNCED;
      break;
    case STATE_AT_CUDA:
    case STATE_AT_MPS:
    case SYNCED:
      break;
  }
#endif
}

void UnifiedMemory::ToMPS(size_t size) {
#ifdef USE_MPS
  switch (state_) {
    case UNINITIALIZED:
      mps_ptr_ = MPSContext::New(size_);
      MPSContext::Memset(size_, mps_ptr_);
      device_id_ = MPSContext::current_device();
      state_ = STATE_AT_MPS;
      break;
    case STATE_AT_CPU:
      if (mps_ptr_ == nullptr) {
        mps_ptr_ = MPSContext::NewSharedFromBytes(size_, cpu_ptr_);
        device_id_ = MPSContext::current_device();
        cpu_ptr_ = MTLGetBufferContents(mps_ptr_);
        own_cpu_ptr_ = false;
      } else {
        MPSContext::SynchronizeResource(
            nullptr, size > 0 ? size : size_, mps_ptr_);
      }
      state_ = SYNCED;
      break;
    case STATE_AT_CUDA:
    case STATE_AT_MPS:
    case SYNCED:
      break;
  }
#endif
}

const void* UnifiedMemory::cpu_data(size_t size, size_t offset) {
  ToCPU(size);
  return (const void*)((uint8_t*)cpu_ptr_ + offset);
}

const void* UnifiedMemory::cuda_data(size_t size, size_t offset) {
#ifdef USE_CUDA
  SwitchToCUDADevice(CUDAContext::current_device());
#endif
  ToCUDA(size);
  return (const void*)((uint8_t*)cuda_ptr_ + offset);
}

const void* UnifiedMemory::mps_data(size_t size) {
#ifdef USE_MPS
  SwitchToMPSDevice(MPSContext::current_device());
#endif
  ToMPS(size);
  return (const void*)((uint8_t*)mps_ptr_);
}

void* UnifiedMemory::mutable_cpu_data(size_t size) {
  ToCPU(size);
  state_ = STATE_AT_CPU;
  return cpu_ptr_;
}

void* UnifiedMemory::mutable_cuda_data(size_t size) {
#ifdef USE_CUDA
  SwitchToCUDADevice(CUDAContext::current_device());
#endif
  ToCUDA(size);
  state_ = STATE_AT_CUDA;
  return cuda_ptr_;
}

void* UnifiedMemory::mutable_mps_data(size_t size) {
#ifdef USE_MPS
  SwitchToMPSDevice(MPSContext::current_device());
#endif
  ToMPS(size);
  state_ = STATE_AT_MPS;
  return mps_ptr_;
}

bool UnifiedMemory::set_cpu_data(void* cpu_ptr, size_t size) {
  if (mps_ptr_) {
    CPUContext::Memcpy<CPUContext, CPUContext>(
        size, mutable_cpu_data(size), cpu_ptr);
    return false; // Disabled external cpu data for MPS.
  } else {
    if (own_cpu_ptr_ && cpu_ptr_) {
      if (meta_.dtor()) {
        meta_.dtor()(cpu_ptr_, size_ / meta_.itemsize());
      }
      CPUContext::Delete(cpu_ptr_);
    }
    size_ = size;
    cpu_ptr_ = cpu_ptr;
    state_ = STATE_AT_CPU;
    own_cpu_ptr_ = false;
    return true;
  }
}

void UnifiedMemory::set_cuda_data(void* cuda_ptr, size_t size, int device_id) {
#ifdef USE_CUDA
  if (own_cuda_ptr_ && cuda_ptr_) {
    CUDAContext::Delete(cuda_ptr_);
  }
#endif
  size_ = size;
  cuda_ptr_ = cuda_ptr;
  state_ = STATE_AT_CUDA;
  own_cuda_ptr_ = false;
  device_id_ = device_id;
}

UnifiedMemory::~UnifiedMemory() {
  if (own_cpu_ptr_ && cpu_ptr_) {
    if (meta_.dtor()) {
      meta_.dtor()(cpu_ptr_, size_ / meta_.itemsize());
    }
    CPUContext::Delete(cpu_ptr_);
  }
#ifdef USE_CUDA
  if (own_cuda_ptr_ && cuda_ptr_) {
    CUDAContext::Delete(cuda_ptr_);
  }
#endif
#ifdef USE_MPS
  if (own_mps_ptr_ && mps_ptr_) {
    MPSContext::Delete(mps_ptr_);
  }
#endif
}

void UnifiedMemory::SwitchToCUDADevice(int device_id) {
#ifdef USE_CUDA
  if (cuda_ptr_) {
    if (device_id != device_id_) {
      void* new_ptr_ = nullptr;
      CUDADeviceGuard guard(device_id);
      new_ptr_ = CUDAContext::New(size_);
      CUDAContext::Memcpy<CUDAContext, CUDAContext>(
          size_, new_ptr_, cuda_ptr_, device_id_);
      if (own_cuda_ptr_) {
        CUDAContext::Delete(cuda_ptr_);
      }
      cuda_ptr_ = new_ptr_;
      device_id_ = device_id;
    }
  } else {
    CUDADeviceGuard guard(device_id);
    ToCUDA(size_);
  }
#endif
}

void UnifiedMemory::SwitchToMPSDevice(int device_id) {
#ifdef USE_MPS
  if (mps_ptr_) {
    if (device_id != device_id_) {
      NOT_IMPLEMENTED;
    }
  } else {
    MPSDeviceGuard guard(device_id);
    ToMPS(size_);
  }
#endif
}

Map<string, string> UnifiedMemory::info() const {
  static map<State, string> state_map{
      {UNINITIALIZED, "uninitialized"},
      {STATE_AT_CPU, "cpu"},
      {STATE_AT_CUDA, "cuda"},
      {STATE_AT_MPS, "mps"},
      {SYNCED, "device"},
  };
  string state_str = state_map[state_];
  if (state_str == "device") {
    if (cuda_ptr_) {
      state_str = "cuda";
    } else if (mps_ptr_) {
      state_str = "mps";
    } else {
      LOG(FATAL) << "Device activated, but got invalid mem pointer.";
    }
  }
  return {
      {"device_type", state_str},
      {"device_id", str::to(device_id_)},
  };
}

} // namespace dragon
