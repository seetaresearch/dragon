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
  NCHW = 0,
  NHWC = 1,
} StorageOrder;

/*!
 * \brief Memory to manage both the host and device data.
 */
class DRAGON_API UnifiedMemory {
 public:
  /*!
   * \brief The device-aware state for data mutation.
   */
  enum State {
    /*! \brief Initial state */
    UNINITIALIZED = 0,
    /*! \brief Data is mutable to cpu */
    STATE_AT_CPU = 1,
    /*! \brief Data is mutable to cuda */
    STATE_AT_CUDA = 2,
    /*! \brief Data is mutable to cnml */
    STATE_AT_CNML = 3,
    /*! \brief Data is synced between host and device */
    SYNCED = 4,
  };

  /*! \brief Default constructor */
  UnifiedMemory() {}

  /*! \brief Constructor with the type meta and size */
  UnifiedMemory(const TypeMeta& meta, size_t size) : meta_(meta), size_(size) {}

  /*! \brief Destructor */
  ~UnifiedMemory();

  /*! \brief Switch to the given device */
  void SwitchToDevice(int device);

  /*! \brief Switch to the given cuda device */
  void SwitchToCUDADevice(int device);

  /*! \brief Involve the state to CPUContext */
  void ToCPU(size_t size = 0);

  /*! \brief Involve the state to CUDAContext */
  void ToCUDA(size_t size = 0);

  /*! \brief Return the memory state */
  State state() const {
    return state_;
  }

  /*! \brief Return the total number of bytes */
  size_t size() const {
    return size_;
  }

  /*! \brief Return the total number of bytes on given device */
  size_t size(const string& device_type, int device_id) const {
    if (device_type == "cuda") {
      if (own_cuda_ptr_ && cuda_ptr_ && device_id_ == device_id) {
        return size_;
      }
    }
    return size_t(0);
  }

  /*! \brief Return the number of memory chunks */
  size_t num_chunks() const {
    return num_chunks_;
  }

  /*! \brief Return the storage order */
  StorageOrder order() const {
    return order_;
  }

  /*! \brief Return the device index */
  int device() const {
    return device_id_;
  }

  /*! \brief Return the data info */
  Map<string, string> info() const;

  /*! \brief Return the const cpu data */
  const void* cpu_data(size_t size = 0);

  /*! \brief Return the const cuda data */
  const void* cuda_data(size_t size = 0);

  /*! \brief Return the const cnml data */
  const void* cnml_data();

  /*! \brief Return the mutable cpu data */
  void* mutable_cpu_data(size_t size = 0);

  /*! \brief Return the mutable cuda data */
  void* mutable_cuda_data(size_t size = 0);

  /*! \brief Return the mutable cnml data */
  void* mutable_cnml_data();

  /*! \brief Return the binding cnml cpu tensor */
  cnmlCpuTensor_t& cnml_cpu_tensor();

  /*! \brief Return the binding cnml mlu tensor */
  cnmlTensor_t& cnml_mlu_tensor();

  /*! \brief Allocate the mlu device data */
  void* malloc_cnml_data();

  /*! \brief Copy the mlu device data to host */
  void fetch_cnml_data(void** data);

  /*! \brief Set the number of data chunks */
  void set_num_chunks(size_t num_chunks) {
    num_chunks_ = num_chunks;
  }

  /*! \brief Set the storage order */
  void set_order(StorageOrder order) {
    order_ = order;
  }

  /*! \brief Set to use an external block of cpu data */
  void set_cpu_data(void* cpu_ptr, size_t size);

  /*! \brief Set to use an external block of cuda data */
  void set_cuda_data(void* cuda_ptr, size_t size, int device);

 private:
  /*! \brief The data state */
  State state_ = UNINITIALIZED;

  /*! \brief The size and number of chunks */
  size_t size_ = 0, num_chunks_ = 1;

  /*! \brief The type meta */
  TypeMeta meta_;

  /*! \brief The storage order */
  StorageOrder order_ = NCHW;

  /*! \brief The device index */
  int device_id_ = 0;

  /*! \brief The cpu data pointer */
  void* cpu_ptr_ = nullptr;

  /*! \brief The ownership of cpu data pointer */
  bool own_cpu_ptr_ = true;

  /*! \brief The cuda data pointer */
  void* cuda_ptr_ = nullptr;

  /*! \brief The ownership of cuda data pointer */
  bool own_cuda_ptr_ = true;

  /*! \brief The cnml data pointer */
  void* cnml_ptr_ = nullptr;

  /*! \brief The binding cpu tensor for cnml */
  cnmlCpuTensor_t cnml_cpu_tensor_ = nullptr;

  /*! \brief The binding mlu tensor for cnml */
  cnmlTensor_t cnml_mlu_tensor_ = nullptr;

  DISABLE_COPY_AND_ASSIGN(UnifiedMemory);
};

} // namespace dragon

#endif // DRAGON_CORE_MEMORY_H_
