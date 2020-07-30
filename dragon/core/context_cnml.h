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

#ifndef DRAGON_CORE_CONTEXT_CNML_H_
#define DRAGON_CORE_CONTEXT_CNML_H_

#include "dragon/core/common.h"

struct cnrtStream;
struct cnmlCpuTensor;
struct cnmlTensor;
struct cnmlFusionOp;
typedef struct cnrtStream* cnrtStream_t;
typedef struct cnmlCpuTensor* cnmlCpuTensor_t;
typedef struct cnmlTensor* cnmlTensor_t;
typedef struct cnmlFusionOp* cnmlFusionOp_t;

namespace dragon {

/*!
 * \brief The cnml device context.
 */
class CNMLContext {
 public:
  /*! \brief Default constructor */
  CNMLContext() : device_id_(0), random_seed_(DEFAULT_RNG_SEED) {}

  /*! \brief Constructor with the device index */
  explicit CNMLContext(int device)
      : device_id_(device), random_seed_(DEFAULT_RNG_SEED) {}

  /*! \brief Constructor with the device option */
  explicit CNMLContext(const DeviceOption& option)
      : device_id_(option.device_id()),
        random_seed_(
            option.has_random_seed() ? option.random_seed()
                                     : DEFAULT_RNG_SEED) {
    CHECK_EQ(option.device_type(), PROTO_CNML);
  }

  /*! \brief Allocate a block of memory */
  static void* New(size_t size) {
    return nullptr;
  }

  /*! \brief Set a memory block to the given value */
  static void Memset(size_t n, void* ptr, int value) {}

  /*! \brief Set a memory block to the given value asynchronously */
  void MemsetAsync(size_t n, void* ptr, int value) {
    Memset(n, ptr, value);
  }

  /*! \brief Copy a memory block to the destination */
  template <class DestContext, class SrcContext>
  static void Memcpy(size_t n, void* dest, const void* src) {}

  /*! \brief Copy a memory block to the destination asynchronously */
  template <class DestContext, class SrcContext>
  void MemcpyAsync(size_t n, void* dest, const void* src) {
    Memcpy<DestContext, SrcContext>(dest, src, n);
  }

  /*! \brief Deallocate a memory block */
  static void Delete(void* ptr) {}

  /*! \brief Switch to the device in current thread */
  void SwitchToDevice() {
    SwitchToDevice(0);
  }

  /*! \brief Switch to the device and select given stream in current thread */
  void SwitchToDevice(int stream) {}

  /*! \brief Wait for the dispatched computation to complete */
  void FinishDeviceComputation() {}

  /*! \brief Return the cnrt stream */
  cnrtStream_t cnrt_stream() {
    return cnrt_stream(device_id_, stream_id_);
  }

  /*! \brief Return the specified cnrt stream */
  static cnrtStream_t cnrt_stream(int device_id, int stream_id) {
    return (cnrtStream_t) nullptr;
  }

  /*! \brief Return the device index */
  int device() const {
    return device_id_;
  }

  /*! \brief Return the stream index */
  int stream() const {
    return stream_id_;
  }

 private:
  int device_id_, stream_id_ = 1, random_seed_;
  unique_ptr<std::mt19937> rand_generator_;
};

} // namespace dragon

#endif // DRAGON_CORE_CONTEXT_CNML_H_
