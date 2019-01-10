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

#ifndef DRAGON_CORE_MIXEDMEM_H_
#define DRAGON_CORE_MIXEDMEM_H_

#include "core/context.h"
#include "core/context_cuda.h"
#include "core/context_cnml.h"

namespace dragon {

typedef enum {
    NCHW,
    NHWC,
} StorageOrder;

class MixedMemory {
 public:
    typedef enum {
        /*! \brief Initial state */
        UNINITIALIZED,
        /*! \brief Memory could be modified by CPUContext last time */
        STATE_AT_CPU,
        /*! \brief Memory could be modified by CUDAContext last time */
        STATE_AT_CUDA,
        /*! \brief Memory could be modified by CNMLContext last time */
        STATE_AT_CNML,
        /*! \brief Memory should be copied to another device next time */
        SWITCHED,
        /*! \brief Host and Device now hold the same contents */
        SYNCED,
    } State;

    /*! \brief Default Constructor */
    MixedMemory() : cpu_ptr_(nullptr),
          cuda_ptr_(nullptr), cnml_ptr_(nullptr) {}

    /*! \brief Constructor with the known meta and size */
    MixedMemory(const TypeMeta& meta, const size_t nbytes)
        : meta_(meta), nbytes_(nbytes), cpu_ptr_(nullptr),
          cuda_ptr_(nullptr), cnml_ptr_(nullptr) {}

    /*! \brief Deconstructor */
    ~MixedMemory();

    /*! \brief Return the const data pointer on CPUContext */
    const void* cpu_data();

    /*! \brief Return the const data pointer on CUDAContext */
    const void* cuda_data();

    /*! \brief Return the const data pointer on CNMLContext */
    const void* cnml_data();

    /*! \brief Return the mutable data pointer on CPUContext */
    void* mutable_cpu_data();

    /*! \brief Return the mutable data pointer on CUDAContext */
    void* mutable_cuda_data();

    /*! \brief Return the mutable data pointer on CNMLContext */
    void* mutable_cnml_data();

    /*! \brief Allocate the mlu devive memory */
    void* malloc_cnml_data();

    /*! \brief Copy the mlu device memory to the host */
    void fetch_cnml_data(void** data);

    /*! \brief Return the binding CNML cpu tensor */
    cnmlCpuTensor_t& cnml_cpu_tensor();

    /*! \brief Return the binding CNML mlu tensor */
    cnmlTensor_t& cnml_mlu_tensor();

    /*! \brief Set the cpu data pointer from external context */
    void set_cpu_data(void* cpu_ptr, size_t nbytes);
    
    /*! \brief Switch to the device set by Context before */
    void SwitchToDevice();

    /*! \brief Switch to the specified device */
    void SwitchToCUDADevice(int device_id);

    /*! \brief Return the total bytes of this memory */
    size_t nbytes() const { return nbytes_; }

    /*! \brief Return the chunks of this memory */
    size_t nchunks() const { return nchunks_; }

    /*! \brief Set the chunks of this memory */
    void set_nchunks(size_t nchunks) { nchunks_ = nchunks; }

    /*! \brief Return the state of this memory */
    State state() const { return state_; }

    /*! \brief Return or Set the storage order */
    StorageOrder order() const { return order_; }

    /*! \brief Set the storage order */
    void set_order(StorageOrder order) { order_ = order; }

    /*! \brief Return a string to describe the internal structure */
    const Map<string, string> info() const;

    /*! \brief Control the state machine to CPUContext */
    void ToCPU();

    /*! \brief Control the state machine to CUDAContext */
    void ToCUDA();

 private:
    /*! \brief The type meta to call the deconstructor */
    TypeMeta meta_;

    /*! \brief The number of total bytes */
    size_t nbytes_ = 0, nchunks_ = 1;

    /*! \brief The optional storage order */
    StorageOrder order_ = NCHW;

    /*! \brief Current memory status indicator */
    State state_ = UNINITIALIZED;

    /*! \brief Data pointers */
    void* cpu_ptr_, *cuda_ptr_, *cnml_ptr_;

    /*! \brief Whether this memory owns the cpu data pointer */
    int own_cpu_ptr_ = 1;
    
    /*! \brief Store the device id for some data pointers */
    int ptr_device_ = 0;

    /*! \brief Binding cpu tensor for CAMBRICON's CNML Library */
    cnmlCpuTensor_t cnml_cpu_tensor_ = nullptr;

    /*! \brief Binding mlu tensor for CAMBRICON's CNML Library */
    cnmlTensor_t cnml_mlu_tensor_ = nullptr;
};

}  // namespace dragon

#endif  // DRAGON_CORE_MIXEDMEM_H_