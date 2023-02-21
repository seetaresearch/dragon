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

#include "dragon/core/common.h"

namespace dragon {

/*!
 * \brief Memory to manage both the host and device data.
 */
class DRAGON_API UnifiedMemory {
 public:
  /*!
   * \brief The device-aware state for data consistency.
   */
  enum State {
    /*! \brief Initial state */
    UNINITIALIZED = 0,
    /*! \brief Data is mutable to cpu */
    STATE_AT_CPU = 1,
    /*! \brief Data is mutable to cuda */
    STATE_AT_CUDA = 2,
    /*! \brief Data is mutable to mps */
    STATE_AT_MPS = 3,
    /*! \brief Data is mutable to mlu */
    STATE_AT_MLU = 4,
    /*! \brief Data is synced between host and device */
    SYNCED = 5,
  };

  /*! \brief Constructor */
  UnifiedMemory() {}

  /*! \brief Constructor with the type meta and size */
  UnifiedMemory(const TypeMeta& meta, size_t size) : meta_(meta), size_(size) {}

  /*! \brief Destructor */
  ~UnifiedMemory();

  /*! \brief Switch to the given cuda device */
  void SwitchToCUDADevice(int device);

  /*! \brief Switch to the given mps device */
  void SwitchToMPSDevice(int device);

  /*! \brief Switch to the given mlu device */
  void SwitchToMLUDevice(int device);

  /*! \brief Set to the cpu state */
  void ToCPU(size_t size = 0);

  /*! \brief Set to the cuda state */
  void ToCUDA(size_t size = 0);

  /*! \brief Set to the mps state */
  void ToMPS(size_t size = 0);

  /*! \brief Set to the mlu state */
  void ToMLU(size_t size = 0);

  /*! \brief Return the state */
  State state() const {
    return state_;
  }

  /*! \brief Return the data size */
  size_t size() const {
    return size_;
  }

  /*! \brief Return the data size on given device */
  size_t size(const string& device_type, int device_id) const {
    if (device_type == "cuda") {
      if (own_cuda_ptr_ && cuda_ptr_ && device_id_ == device_id) return size_;
    } else if (device_type == "mlu") {
      if (own_mlu_ptr_ && mlu_ptr_ && device_id_ == device_id) return size_;
    } else if (device_type == "mps") {
      if (own_mps_ptr_ && mps_ptr_ && device_id_ == device_id) return size_;
    }
    return size_t(0);
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
  const void* cpu_data(size_t size = 0, size_t offset = 0);

  /*! \brief Return the const cuda data */
  const void* cuda_data(size_t size = 0, size_t offset = 0);

  /*! \brief Return the const mps data */
  const void* mps_data(size_t size = 0);

  /*! \brief Return the const mlu data */
  const void* mlu_data(size_t size = 0, size_t offset = 0);

  /*! \brief Return the mutable cpu data */
  void* mutable_cpu_data(size_t size = 0);

  /*! \brief Return the mutable cuda data */
  void* mutable_cuda_data(size_t size = 0);

  /*! \brief Return the mutable mps data */
  void* mutable_mps_data(size_t size = 0);

  /*! \brief Return the mutable mlu data */
  void* mutable_mlu_data(size_t size = 0);

  /*! \brief Set the storage order */
  void set_order(StorageOrder order) {
    order_ = order;
  }

  /*! \brief Set to use an external block of cpu data */
  bool set_cpu_data(void* cpu_ptr, size_t size);

  /*! \brief Set to use an external block of cuda data */
  void set_cuda_data(void* cuda_ptr, size_t size, int device);

  /*! \brief Set to use an external block of mlu data */
  void set_mlu_data(void* mlu_ptr, size_t size, int device);

 private:
  /*! \brief The data state */
  State state_ = UNINITIALIZED;

  /*! \brief The data size */
  size_t size_ = 0;

  /*! \brief The type meta */
  TypeMeta meta_;

  /*! \brief The storage order */
  StorageOrder order_ = NCHW;

  /*! \brief The device index */
  int device_id_ = 0;

  /*! \brief The cpu data pointer */
  void* cpu_ptr_ = nullptr;

  /*! \brief The cuda data pointer */
  void* cuda_ptr_ = nullptr;

  /*! \brief The mps data pointer */
  void* mps_ptr_ = nullptr;

  /*! \brief The mlu data pointer */
  void* mlu_ptr_ = nullptr;

  /*! \brief The ownership of cpu data pointer */
  bool own_cpu_ptr_ = true;

  /*! \brief The ownership of cuda data pointer */
  bool own_cuda_ptr_ = true;

  /*! \brief The ownership of mps data pointer */
  bool own_mps_ptr_ = true;

  /*! \brief The ownership of mlu data pointer */
  bool own_mlu_ptr_ = true;

  DISABLE_COPY_AND_ASSIGN(UnifiedMemory);
};

} // namespace dragon

#endif // DRAGON_CORE_MEMORY_H_
