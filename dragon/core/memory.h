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

#ifndef DRAGON_CORE_MEMORY_H_
#define DRAGON_CORE_MEMORY_H_

#include "dragon/core/context.h"
#include "dragon/core/context_cnml.h"
#include "dragon/core/context_cuda.h"

namespace dragon {

typedef enum {
  NCHW,
  NHWC,
} StorageOrder;

class DRAGON_API UnifiedMemory {
 public:
  typedef enum {
    /*! \brief The initial state */
    UNINITIALIZED,
    /*! \brief Memory could be modified by CPUContext last time */
    STATE_AT_CPU,
    /*! \brief Memory could be modified by CUDAContext last time */
    STATE_AT_CUDA,
    /*! \brief Memory could be modified by CNMLContext last time */
    STATE_AT_CNML,
    /*! \brief The synced state  */
    SYNCED,
  } State;

  /*! \brief Default Constructor */
  UnifiedMemory() : cpu_ptr_(nullptr), cuda_ptr_(nullptr), cnml_ptr_(nullptr) {}

  /*! \brief Constructor with the known meta and size */
  UnifiedMemory(const TypeMeta& meta, size_t size)
      : meta_(meta),
        size_(size),
        cpu_ptr_(nullptr),
        cuda_ptr_(nullptr),
        cnml_ptr_(nullptr) {}

  /*! \brief Destructor */
  ~UnifiedMemory();

  /*! \brief Switch to the specified device */
  void SwitchToDevice(int device_id);

  /*! \brief Switch to the specified cuda device */
  void SwitchToCUDADevice(int device_id);

  /*! \brief Involve the state to CPUContext */
  void ToCPU(size_t size = 0);

  /*! \brief Involve the state to CUDAContext */
  void ToCUDA(size_t size = 0);

  /*! \brief Return the device index */
  int device_id() const {
    return device_id_;
  }

  /*! \brief Return the total number of bytes */
  size_t size() const {
    return size_;
  }

  /*! \brief Return the number of chunks */
  size_t nchunks() const {
    return nchunks_;
  }

  /*! \brief Return the storage order */
  StorageOrder order() const {
    return order_;
  }

  /*! \brief Return the memory state */
  State state() const {
    return state_;
  }

  /*! \brief Return a string to describe the internal structure */
  Map<string, string> info() const;

  /*! \brief Return the const data pointer on CPUContext */
  const void* cpu_data(size_t size = 0);

  /*! \brief Return the const data pointer on CUDAContext */
  const void* cuda_data(size_t size = 0);

  /*! \brief Return the const data pointer on CNMLContext */
  const void* cnml_data();

  /*! \brief Return the mutable data pointer on CPUContext */
  void* mutable_cpu_data(size_t size = 0);

  /*! \brief Return the mutable data pointer on CUDAContext */
  void* mutable_cuda_data(size_t size = 0);

  /*! \brief Return the mutable data pointer on CNMLContext */
  void* mutable_cnml_data();

  /*! \brief Return the binding cnml cpu tensor */
  cnmlCpuTensor_t& cnml_cpu_tensor();

  /*! \brief Return the binding cnml mlu tensor */
  cnmlTensor_t& cnml_mlu_tensor();

  /*! \brief Allocate the mlu device memory */
  void* malloc_cnml_data();

  /*! \brief Copy the mlu device memory to the host */
  void fetch_cnml_data(void** data);

  /*! \brief Set the chunks of this memory */
  void set_nchunks(size_t nchunks) {
    nchunks_ = nchunks;
  }

  /*! \brief Set the storage order */
  void set_order(StorageOrder order) {
    order_ = order;
  }

  /*! \brief Set the cpu data pointer from external context */
  void set_cpu_data(void* cpu_ptr, size_t size);

  /*! \brief Set the cuda data pointer from external context */
  void set_cuda_data(void* cuda_ptr, size_t size, int device_id);

 private:
  /*! \brief The type meta */
  TypeMeta meta_;

  /*! \brief The size and number of chunks */
  size_t size_ = 0, nchunks_ = 1;

  /*! \brief The storage order */
  StorageOrder order_ = NCHW;

  /*! \brief The current state */
  State state_ = UNINITIALIZED;

  /*! \brief The data pointers */
  void *cpu_ptr_, *cuda_ptr_, *cnml_ptr_;

  /*! \brief The ownership of data pointers */
  int own_cpu_ptr_ = 1, own_cuda_ptr_ = 1;

  /*! \brief The device index */
  int device_id_ = 0;

  /*! \brief The binding cpu tensor for cnml */
  cnmlCpuTensor_t cnml_cpu_tensor_ = nullptr;

  /*! \brief The binding mlu tensor for cnml */
  cnmlTensor_t cnml_mlu_tensor_ = nullptr;
};

} // namespace dragon

#endif // DRAGON_CORE_MEMORY_H_
