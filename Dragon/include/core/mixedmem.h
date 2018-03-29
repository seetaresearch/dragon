// ------------------------------------------------------------
// Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

#ifndef DRAGON_CORE_MIXEDMEM_H_
#define DRAGON_CORE_MIXEDMEM_H_

#include "typeid.h"
#include "context.h"
#include "context_cuda.h"

namespace dragon {

class MixedMemory {
 public:
    enum State { UNINITIALIZED, STATE_AT_CPU, STATE_AT_CUDA, SWITCHED, SYNCED };
    MixedMemory()
        : state_(UNINITIALIZED),
          cpu_ptr_(nullptr), cuda_ptr_(nullptr),
          nbytes_(0) {}
    MixedMemory(const TypeMeta& meta, const size_t nbytes)
        : state_(UNINITIALIZED), meta_(meta),
          cpu_ptr_(nullptr), cuda_ptr_(nullptr),
          nbytes_(nbytes) {}
    ~MixedMemory();

    const void* cpu_data();
    const void* cuda_data();
    void* mutable_cpu_data();
    void* mutable_cuda_data();
#ifdef WITH_CUDA
    void async_cuda_data(const cudaStream_t& stream);
#endif

    void SwitchToDevice();

    inline size_t nbytes() const { return nbytes_; }

    inline void* cpu_ptr() { state_ = STATE_AT_CPU; return cpu_ptr_; }
    inline void* cuda_ptr() { state_ = STATE_AT_CUDA; return cuda_ptr_; }

    inline State state() { return state_; }

 private:
    void ToCUDA();
    void ToCPU();

    void* cpu_ptr_, *cuda_ptr_;
    State state_;
    size_t nbytes_;
    TypeMeta meta_;
};

}    // namespace dragon

#endif
