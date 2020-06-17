/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *    <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_CORE_CONTEXT_CNML_H_
#define DRAGON_CORE_CONTEXT_CNML_H_

/* CAMBRICON CNRT && CNML Environment */

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

class CNRTObject;

class CNMLContext {
 public:
  /*! \brief Default Constructor */
  explicit CNMLContext(const DeviceOption& option)
      : device_id_(option.device_id()),
        random_seed_(
            option.has_random_seed() ? option.random_seed()
                                     : DEFAULT_RNG_SEED) {
    CHECK_EQ(option.device_type(), PROTO_CNML);
  }

  /*! \brief Constructor with the specified device index */
  explicit CNMLContext(int device_id = 0)
      : device_id_(device_id), random_seed_(DEFAULT_RNG_SEED) {}

  /*! \brief Alloc the memory */
  static void* New(size_t nbytes);

  /*! \brief Zero-Reset the memory */
  static void Memset(size_t nbytes, void* ptr);

  /*! \brief Copy the memory */
  template <class DestContext, class SrcContext>
  static void Memcpy(size_t nbytes, void* dest, const void* src);

  /*! \brief Free the memory */
  static void Delete(void* data);

  /*! \brief Zero-Reset the memory asynchronously */
  void MemsetAsync(size_t nbytes, void* ptr) {
    Memset(nbytes, ptr);
  }

  /*! \brief Copy the memory asynchronously */
  template <class DestContext, class SrcContext>
  void MemcpyAsync(size_t nbytes, void* dest, const void* src) {
    Memcpy<DestContext, SrcContext>(dest, src, nbytes);
  }

  /*! \brief Switch to the device with the given stream */
  void SwitchToDevice(int stream_id) {}

  /*! \brief Switch to the device of this context */
  void SwitchToDevice() {
    SwitchToDevice(0);
  }

  /*! \brief Synchronize the dispatched operations */
  void FinishDeviceComputation() {}

  /*! \brief Return the specified cnrt stream */
  static cnrtStream_t cnrt_stream(int device_id, int stream_id);

  /*! \brief Return the internal cnrt stream */
  cnrtStream_t cnrt_stream() {
    return cnrt_stream(device_id_, stream_id_);
  }

  /*! \brief Return the device index */
  int device_id() const {
    return device_id_;
  }

  /*! \brief Return the stream index */
  int stream_id() const {
    return stream_id_;
  }

  /*! \brief Return the global context locker */
  static std::mutex& mutex() {
    static std::mutex m;
    return m;
  }

  /*! \brief Return the thread local cnrt object */
  static CNRTObject* obj();

  /*! \brief Set the stream index */
  void set_stream_id(int stream_id) {
    stream_id_ = stream_id;
  }

 private:
  int device_id_, stream_id_ = 1, random_seed_;
  unique_ptr<std::mt19937> rand_generator_;
};

} // namespace dragon

#endif // DRAGON_CORE_CONTEXT_CNML_H_
